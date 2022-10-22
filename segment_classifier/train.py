import argparse
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import tqdm

from dataset.segment_dataset import SegmentDataset
from model.pointnet2 import PointNet2Classification
from utils import train_utils


class Config:
    # Learning rate parameters
    LR = 2e-3
    LR_CLIP = 0.00001
    LR_DECAY = 0.5
    DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]

    # Model config
    USE_SEM_FEATURES = False
    USE_SEM_REFINEMENT = False

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


def parse_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('-o', '--output-dir', type=str, default="results/")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('--use-sem-features', action="store_true")
    return parser.parse_args()

# __C.TRAIN.MOMS = [0.95, 0.85]
# __C.TRAIN.DIV_FACTOR = 10.0
# __C.TRAIN.PCT_START = 0.4


def create_optimizer(cfg, model):
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    return optimizer


def create_scheduler(cfg, model, optimizer, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.LR_DECAY
        return max(cur_decay, cfg.LR_CLIP / cfg.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.BN_DECAY
        return max(cfg.BN_MOMENTUM * cur_decay, cfg.BNM_CLIP)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


def train_one_iter(cfg, model, batch, optimizer):
    model.train()

    loss_func = F.binary_cross_entropy_with_logits
    optimizer.zero_grad()
    xyz = torch.from_numpy(batch["xyz"]).cuda().float()
    cls_labels = torch.from_numpy(batch["gt_label"]).cuda().float()

    # TODO: use this later for segment refinement
    # segm_label = torch.from_numpy(batch["semantic_label"]).cuda().long()

    if cfg.USE_SEM_FEATURES:
        features = torch.from_numpy(batch["semantic_features"]).cuda().float()
        pts_input = torch.cat([xyz, features], dim=-1)
    else:
        pts_input = xyz

    pred_cls = model(pts_input)
    pred_cls = pred_cls.view(-1)
    loss = loss_func(pred_cls, cls_labels)
    loss.backward()
    clip_grad_norm_(model.parameters(), cfg.GRAD_NORM_CLIP)
    optimizer.step()
    return loss.item()


def train(cfg, model, optimizer, train_loader, val_loader=None, ckpt_dir='checkpoints/', ckpt_save_interval=5, eval_frequency=5, tb_log=None):
    lr_scheduler, bnm_scheduler = create_scheduler(cfg, model, optimizer, -1)
    it = 0
    with tqdm.trange(0, cfg.EPOCHS, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:
        for epoch in tbar:
            for idx, batch in enumerate(train_loader):
                cur_lr = lr_scheduler.get_lr()[0]

                loss = train_one_iter(cfg, model, batch, optimizer)
                it += 1

                # log to console and tensorboard
                pbar.update()
                pbar.set_postfix(dict(total_it=it))
                tbar.set_postfix(dict(loss=loss))
                tbar.refresh()

                if tb_log is not None:
                    tb_log.add_scalar('train_loss', loss, it)
                    tb_log.add_scalar('learning_rate', cur_lr, it)
            
            lr_scheduler.step()
            bnm_scheduler.step(it)

            # save trained model
            trained_epoch = epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                train_utils.save_checkpoint(
                    train_utils.checkpoint_state(model, optimizer, trained_epoch, it), filename=ckpt_name,
                )

             # eval one epoch
            if (epoch % eval_frequency) == 0:
                pbar.close()
                if val_loader is not None:
                    with torch.no_grad():
                        val_loss, val_acc = validate(cfg, model, val_loader)
                    if tb_log is not None:
                        tb_log.add_scalar('val_loss', val_loss, it)
                        tb_log.add_scalar('val_acc', val_acc, it)
            

            pbar.close()
            pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
            pbar.set_postfix(dict(total_it=it))


def validate(cfg, model, val_loader):
    model.eval()
    loss_func = F.binary_cross_entropy_with_logits
    
    total_loss = 0.
    num_correct, num_total = 0, 0
    for i, batch in tqdm.tqdm(enumerate(val_loader, 0), total=len(val_loader), leave=False, desc='val'):
        optimizer.zero_grad()

        xyz = torch.from_numpy(batch["xyz"]).cuda().float()
        cls_labels = torch.from_numpy(batch["gt_label"]).cuda().float()

        # TODO: use this later for segment refinement
        # segm_label = torch.from_numpy(batch["semantic_label"]).cuda().long()

        if cfg.USE_SEM_FEATURES:
            features = torch.from_numpy(batch["semantic_features"]).cuda().float()
            pts_input = torch.cat([xyz, features], dim=-1)
        else:
            pts_input = xyz

        pred_cls = model(pts_input)
        pred_cls = pred_cls.view(-1)
        pred_label = (pred_cls.sigmoid() >= cfg.FG_THRESH).long()

        loss = loss_func(pred_cls, cls_labels)
        total_loss += loss.item()

        num_correct += (pred_label == cls_labels).sum()
        num_total += pts_input.shape[0]
    acc = num_correct / num_total
    return loss.item(), acc.item()


if __name__ == "__main__":
    args = parse_args()
    data_dir = "/project_data/ramanan/achakrav/4D-PLS/segment_classifier/segment_dataset/"

    # load training config
    cfg = Config()
    cfg.EPOCHS = args.epochs
    cfg.USE_SEM_FEATURES = args.use_sem_features
    cfg.BATCH_SIZE = args.batch_size

    if cfg.USE_SEM_FEATURES:
        feature_dims = 256
    else:
        feature_dims = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Classification(input_channels=feature_dims, num_classes=1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = create_optimizer(cfg, model)

    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "tensorboard")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(log_dir)
    tb_log = SummaryWriter(log_dir=log_dir)

    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        it, start_epoch = train_utils.load_checkpoint(pure_model, optimizer, filename=args.ckpt)
        last_epoch = start_epoch + 1

    train_dataset = SegmentDataset(data_dir, split='training', n_points=cfg.N_POINTS)
    valid_dataset = SegmentDataset(data_dir, split='validation', n_points=cfg.N_POINTS)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_dataset.collate_batch
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=valid_dataset.collate_batch
    )
    
    train(cfg, model, optimizer, train_loader, val_loader=valid_loader, ckpt_dir=ckpt_dir, tb_log=tb_log)
