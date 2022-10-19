import argparse
import torch
from torch import optim
import torch.optim.lr_scheduler as lr_sched


FG_THRESH = 0.5

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--ckpt_save_interval", type=int, default=5)
parser.add_argument('--workers', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.7)
parser.add_argument('--decay_step', type=int, default=2e4)
parser.add_argument('--weight_decay', type=float, default=0.0)

args = parser.parse_args()

# __C.TRAIN.LR = 0.002
# __C.TRAIN.LR_CLIP = 0.00001
# __C.TRAIN.LR_DECAY = 0.5
# __C.TRAIN.DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]
# __C.TRAIN.LR_WARMUP = False
# __C.TRAIN.WARMUP_MIN = 0.0002
# __C.TRAIN.WARMUP_EPOCH = 5

# __C.TRAIN.BN_MOMENTUM = 0.9
# __C.TRAIN.BN_DECAY = 0.5
# __C.TRAIN.BNM_CLIP = 0.01
# __C.TRAIN.BN_DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]

# __C.TRAIN.OPTIMIZER = 'adam'
# __C.TRAIN.WEIGHT_DECAY = 0.0  # "L2 regularization coeff [default: 0.0]"
# __C.TRAIN.MOMENTUM = 0.9

# __C.TRAIN.MOMS = [0.95, 0.85]
# __C.TRAIN.DIV_FACTOR = 10.0
# __C.TRAIN.PCT_START = 0.4

# __C.TRAIN.GRAD_NORM_CLIP = 1.0


def train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it):
    model.train()
    for it, batch in enumerate(train_loader):
        optimizer.zero_grad()

        pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)
        pred_cls = pred_cls.view(-1)

        loss = loss_func(pred_cls, cls_labels)
        loss.backward()
        # clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_it += 1

        pred_class = (torch.sigmoid(pred_cls) > FG_THRESH)
        # fg_mask = cls_labels > 0
        # correct = ((pred_class.long() == cls_labels) & fg_mask).float().sum()
        # union = fg_mask.sum().float() + (pred_class > 0).sum().float() - correct
        # iou = correct / torch.clamp(union, min=1.0)


def train(model, train_loader):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        if cur_epoch >= args.decay_step:
            cur_decay = cur_decay * args.lr_decay
        return cur_decay
        # return max(cur_decay, args.lr_clip / args.lr)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)

    total_it = 0
    for epoch in range(1, args.epochs + 1):
        lr_scheduler.step(epoch)
        total_it = train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it)
