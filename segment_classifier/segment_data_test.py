from Segment_dataset import Segment_dataset
import numpy as np
import os
import torch



if __name__ == '__main__':

    train_dataset = Segment_dataset('segment_dataset', 'training')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        drop_last=True, 
        collate_fn=lambda x: x)

    for i, batch in enumerate(train_loader):
        import pdb;pdb.set_trace()

