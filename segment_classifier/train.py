import argparse
import os

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import WeightedRandomSampler
import tqdm
import wandb

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
    USE_WANDB = True


def parse_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('--exp', type=str, default="xyz")
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
    xyz = batch["xyz"].cuda().float()
    cls_labels = batch["gt_label"].cuda().float()

    # TODO: use this later for segment refinement
    # segm_label = batch["semantic_label"].cuda().long()

    if cfg.USE_SEM_FEATURES:
        features = batch["semantic_features"].cuda().float()
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


def train(cfg, model, optimizer, train_loader, val_loader=None, ckpt_dir='checkpoints/', ckpt_save_interval=5, eval_frequency=5):
    lr_scheduler, bnm_scheduler = create_scheduler(cfg, model, optimizer, -1)
    it = 0
    with tqdm.trange(0, cfg.EPOCHS, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:
        for epoch in tbar:
            for idx, batch in enumerate(train_loader):
                cur_lr = lr_scheduler.get_last_lr()
                loss = train_one_iter(cfg, model, batch, optimizer)
                it += 1

                # log to console
                pbar.update()
                pbar.set_postfix(dict(total_it=it))
                tbar.set_postfix(dict(loss=loss))
                tbar.refresh()

                # log to wandb
                if cfg.USE_WANDB:
                    wandb.log({
                        "train/loss": loss,
                        "lr": cur_lr
                    }, step=it)

            lr_scheduler.step()
            bnm_scheduler.step(it)

            # save trained model
            trained_epoch = epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                ckpt_name = os.path.join(ckpt_dir, 'epoch_%d' % trained_epoch)
                train_utils.save_checkpoint(
                    train_utils.checkpoint_state(model, optimizer, trained_epoch, it), filename=ckpt_name,
                )

             # eval one epoch
            if epoch % eval_frequency == 0 and val_loader is not None:
                pbar.close()
                with torch.no_grad():
                    metric_dict = validate(cfg, model, val_loader)
                if cfg.USE_WANDB:
                    wandb.log(metric_dict, step=it)

            pbar.close()
            pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
            pbar.set_postfix(dict(total_it=it))


def validate(cfg, model, val_loader):
    model.eval()
    loss_func = F.binary_cross_entropy_with_logits
    
    total_loss = 0.
    gt, pred = [], []
    for i, batch in tqdm.tqdm(enumerate(val_loader, 0), total=len(val_loader), leave=False, desc='val'):
        optimizer.zero_grad()

        xyz = batch["xyz"].cuda().float()
        cls_labels = batch["gt_label"].cuda().float()

        # TODO: use this later for segment refinement
        # segm_label = batch["semantic_label"].cuda().long()

        if cfg.USE_SEM_FEATURES:
            features = batch["semantic_features"].cuda().float()
            pts_input = torch.cat([xyz, features], dim=-1)
        else:
            pts_input = xyz

        pred_cls = model(pts_input)
        pred_cls = pred_cls.view(-1)
        loss = loss_func(pred_cls, cls_labels)
        total_loss += loss.item()

        pred_label = (pred_cls.sigmoid() >= cfg.FG_THRESH).long()
        gt.extend(cls_labels.long().cpu().numpy())
        pred.extend(pred_label.detach().cpu().numpy())

    # compute validation metrics
    acc = accuracy_score(gt, pred)
    balanced_acc = balanced_accuracy_score(gt, pred)
    prec, recall, f1, _ = precision_recall_fscore_support(gt, pred, average="binary")
    metric_dict = {
        "val/loss": total_loss,
        "val/acc": acc,
        "val/balanced_acc": balanced_acc,
        "val/f1": f1,
        "val/prec": prec,
        "val/recall": recall,
    }
    return metric_dict


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
    
    if cfg.USE_WANDB:
        wandb.init("segment-classifier")
        wandb.run.name = args.exp
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2Classification(input_channels=feature_dims, num_classes=1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = create_optimizer(cfg, model)

    output_dir = os.path.join("results", args.exp)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(ckpt_dir)

    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        it, start_epoch = train_utils.load_checkpoint(pure_model, optimizer, filename=args.ckpt)
        last_epoch = start_epoch + 1

    # create dataset
    train_dataset = SegmentDataset(data_dir, split='training', n_points=cfg.N_POINTS)
    valid_dataset = SegmentDataset(data_dir, split='validation', n_points=cfg.N_POINTS)

    # create samplers
    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)
    valid_sampler = None

    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=16,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_batch
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=16,
        pin_memory=True,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate_batch
    )
    
    train(cfg, model, optimizer, train_loader, val_loader=valid_loader, ckpt_dir=ckpt_dir)
