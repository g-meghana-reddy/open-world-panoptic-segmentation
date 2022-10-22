import torch
import time
from tqdm import tqdm

from segment_dataset import SegmentDataset


if __name__ == '__main__':
    data_dir = "/project_data/ramanan/achakrav/4D-PLS/segment_classifier/segment_dataset"
    train_dataset = SegmentDataset(data_dir, split='training')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        collate_fn=train_dataset.collate_batch)

    start_time = time.time()
    for i, batch in tqdm(enumerate(train_loader)):
        # import pdb; pdb.set_trace()
        pass
    end_time = time.time()
    print("Time taken:", end_time-start_time)
