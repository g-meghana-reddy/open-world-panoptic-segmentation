import argparse
import sys

sys.path.append("segment_classifier/")

import torch
from tqdm import tqdm

from dataset.segment_dataset import SegmentDataset
from model.pointnet2 import PointNet2Classification

NUM_POINTS = 1024

class Config:
    # Learning rate parameters
    LR = 2e-3
    LR_CLIP = 0.00001
    LR_DECAY = 0.5
    DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]

    # Model config
    USE_SEG_CLASSIFIER = True
    USE_SEM_FEATURES = False
    USE_SEM_REFINEMENT = False
    USE_SEM_WEIGHTS = False
    NUM_THINGS = 8

    # Optimizer parameters
    WEIGHT_DECAY = 0.0

    # batchnorm parameters
    BN_MOMENTUM = 0.9
    BN_DECAY = 0.5
    BNM_CLIP = 0.01
    BN_DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]

    GRAD_NORM_CLIP = 1.0
    FG_THRESH = 0.5

    EPOCHS = 200
    BATCH_SIZE = 512
    N_POINTS = 1024
    USE_WANDB = True

def validate(cfg, model, val_loader, sem_weights=None):
    model.eval()
    
    total_loss = 0.
    cls_gt, cls_pred, sem_gt, sem_pred = [], [], [], []
    num_batches = 0
    corr_pred = 0
    for i, batch in tqdm(enumerate(val_loader, 0), total=len(val_loader), leave=False, desc='val'):
        # optimizer.zero_grad()

        xyz = batch["xyz"].cuda().float()
        cls_labels = batch["gt_label"].cuda().float()

        sem_label = batch["semantic_label"].cuda().long() - 1

        if cfg.USE_SEM_FEATURES:
            features = batch["semantic_features"].cuda().float()
            pts_input = torch.cat([xyz, features], dim=-1)
        else:
            pts_input = xyz

        pred_cls, pred_sem = model(pts_input)
        if pred_cls is not None:
            pred_cls = pred_cls.view(-1)
        
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        loss = pure_model.loss_fn(pred_cls, pred_sem, batch, sem_weights=sem_weights, train=False)
        total_loss += loss.item()

        if cfg.USE_SEG_CLASSIFIER:
            cls_pred_label = (pred_cls.sigmoid() >= cfg.FG_THRESH).long()
            cls_gt.extend(cls_labels.long().cpu().numpy())
            cls_pred.extend(cls_pred_label.detach().cpu().numpy())

            pos_inds = cls_labels == 1
            neg_inds = cls_labels == 0
            if pos_inds.sum() != 0:
                corr_pred += cls_pred_label[pos_inds].sum()/pos_inds.sum()
                # import pdb;pdb.set_trace()
                num_batches +=1
        print(corr_pred/num_batches)

        if cfg.USE_SEM_REFINEMENT:
            sem_pred_label = torch.argmax(pred_sem, axis=-1).long()
            sem_gt.extend(sem_label.long().cpu().numpy())
            sem_pred.extend(sem_pred_label.detach().cpu().numpy())

    print(corr_pred/num_batches)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="Checkpoint to load for segment classifier", type=str, default='/project_data/ramanan/mganesin/4D-PLS/results/checkpoints/xyz_mean/checkpoints/epoch_200.pth')
    parser.add_argument('--use-sem-features', action="store_true")
    parser.add_argument('--use-sem-refinement', action="store_true")
    parser.add_argument('--use-sem-weights', action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_dir = "/project_data/ramanan/achakrav/4D-PLS/data/segment_dataset/"

    # load training config
    cfg = Config()
    cfg.USE_SEM_FEATURES = args.use_sem_features
    cfg.USE_SEM_REFINEMENT = args.use_sem_refinement
    cfg.USE_SEM_WEIGHTS = args.use_sem_weights

    if cfg.USE_SEM_FEATURES:
        feature_dims = 256
    else:
        feature_dims = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Classification(cfg, feature_dims).cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # create dataset
    valid_dataset = SegmentDataset(data_dir, split='validation', n_points=cfg.N_POINTS)

    # create samplers
    valid_sampler = None

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=16,
        pin_memory=True,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate_batch
    )
    
    sem_weights = None
    with torch.no_grad():
        validate(cfg, model, val_loader, sem_weights)