# Adapted from: https://github.com/sshaoshuai/PointRCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from model.pointnet2_modules import PointnetSAModule
import model.pytorch_utils as pt_utils


USE_BN = False
DP_RATIO = 0.0
XYZ_UP_LAYER = [128, 128]

NUM_POINTS = 512
NPOINTS = [128, 32, -1]
RADIUS = [0.2, 0.4, 100]
NSAMPLE = [64, 64, 64]
MLPS = [[128, 128, 128],
        [128, 128, 256],
        [256, 256, 512]]
CLS_FC = [256, 256]
REG_FC = [256, 256]


class PointNet2Classification(nn.Module):
    """
    PointNet++

    Attributes:
        input_channels : int
            Number of channels associated with features (excludes x,y,z)
        num_classes : int
            Number of categories for classification
    
    NOTE: Batch size cannot be 1 for this network since we use BN: https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
    PointRCNN does not use BN so it's all good
    """
    def __init__(self, cfg, input_channels=3, num_classes=1):
        super(PointNet2Classification, self).__init__()

        # flags for segment_objectness head and semantic refinement head
        self.config = cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        # project XYZ into higher dimension
        self.rcnn_input_channel = 3 + channel_in
        self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + XYZ_UP_LAYER,
                                                bn=USE_BN)
        # c_out = XYZ_UP_LAYER[-1]
        # self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=USE_BN)

        # encode features
        for k in range(len(NPOINTS)):
            mlps = [channel_in] + MLPS[k]

            npoint = NPOINTS[k] if NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=USE_BN
                )
            )
            channel_in = mlps[-1]

        if cfg.USE_SEG_CLASSIFIER:
            cls_channel = 1 if num_classes == 2 else num_classes
            self.cls_layer = self._create_head(channel_in, cls_channel)

        if cfg.USE_SEM_REFINEMENT:
            cls_channel = cfg.NUM_THINGS
            self.sem_layer = self._create_head(channel_in, cls_channel)

        self.init_weights(weight_init='xavier')
    
    def _create_head(self, in_channels, cls_channels):
        cls_layers = []
        pre_channel = in_channels
        for k in range(len(CLS_FC)):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=USE_BN))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channels, activation=None))
        if DP_RATIO > 0:
            cls_layers.insert(1, nn.Dropout(DP_RATIO))
        return nn.Sequential(*cls_layers)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _break_up_pc(self, pc):
        """
        Break an input point cloud into it's position (xyz) and features
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
    
    def forward(self, pointcloud: torch.cuda.FloatTensor):
        """
        Run forward pass

        Params:
            pointcloud : torch.Tensor (B, N, 3 + input_channels)
                Input point cloud of format (x, y, z, features)
        
        Returns:
            Class label for each point cloud
        """
        xyz, features = self._break_up_pc(pointcloud)

        # xyz_input = xyz[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
        # xyz_feature = self.xyz_up_layer(xyz_input)

        # rpn_feature = xyz[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

        # merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
        # merged_feature = self.merge_down_layer(merged_feature)
        # # l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        # xyz, features = xyz, merged_feature.squeeze(dim=3)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        objectness, semantics = None, None
        if self.config.USE_SEG_CLASSIFIER:
            objectness = self.cls_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)

        if self.config.USE_SEM_REFINEMENT:
            semantics = self.sem_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, K)
        return objectness, semantics

    def loss_fn(self, pred_obj_cls, pred_sem_cls, batch, sem_weights=None, train=True):
        cls_loss, sem_loss = 0., 0.
        if self.config.USE_SEG_CLASSIFIER:
            cls_labels = batch["gt_label"].cuda().float()
            if self.config.USE_FOCAL_LOSS:
                cls_loss_func = sigmoid_focal_loss
            else:
                cls_loss_func = F.binary_cross_entropy_with_logits
            cls_loss = cls_loss_func(pred_obj_cls, cls_labels).mean()

        if self.config.USE_SEM_REFINEMENT:
            segment_label = batch["semantic_label"].cuda().long() - 1
            sem_loss_func = F.cross_entropy

            # backpropagate only for valid segments
            if train:
                inds = torch.where(batch['gt_label'] == 1)
                pred_sem_cls = pred_sem_cls[inds] 
                segment_label = segment_label[inds]
            else:
                if self.config.USE_SEG_CLASSIFIER:
                    inds = torch.where(pred_obj_cls.sigmoid() >= self.config.FG_THRESH)
                    pred_sem_cls = pred_sem_cls[inds] 
                    segment_label = segment_label[inds]
            
            if not self.config.USE_SEM_WEIGHTS:
                sem_weights = None

            # only compute loss if there are positive segments
            if len(inds):
                sem_loss = sem_loss_func(pred_sem_cls, segment_label, weight=sem_weights)
        return cls_loss + sem_loss


class Config:
    # Model config
    USE_SEM_FEATURES = False
    USE_SEG_CLASSIFIER = True
    USE_SEM_REFINEMENT = False


if __name__ == "__main__":
    point_cloud = torch.rand(1, 4096, 6).cuda()
    cfg = Config()
    model = PointNet2Classification(cfg, input_channels=3).cuda()
    output = model(point_cloud)
    print(output.shape)