import torch
from torch.utils.data import WeightedRandomSampler
import time
from tqdm import tqdm

from segment_dataset import SegmentDataset


if __name__ == '__main__':
    data_dir = "/project_data/ramanan/achakrav/4D-PLS/segment_classifier/segment_dataset"

    train_dataset = SegmentDataset(data_dir, split='training')
    batch_size = 512

    sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        collate_fn=train_dataset.collate_batch)

    start_time = time.time()
    class_counts = torch.zeros(2)
    for i, batch in tqdm(enumerate(train_loader)):
        class_counts[0] += (batch["gt_label"] == 0).sum()
        class_counts[1] += (batch["gt_label"] == 1).sum()
        # if class_counts[1] > 0:
            # import pdb; pdb.set_trace()
    end_time = time.time()
    print("Time taken:", end_time-start_time)
    print(class_counts)
