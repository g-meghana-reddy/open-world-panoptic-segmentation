import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(self, dataset_path, split='training', n_points=1024, num_things=8):
        self.path = dataset_path
        self.split = split
        self.n_points = n_points
        self.num_things = num_things

        if split == "training":
            self.sequences = ["{:02d}".format(i) for i in range(11) if i != 8]
        elif split == "validation":
            self.sequences = ["{:02d}".format(8)]
        else:
            assert False, "Split not implemented"

        # compute weights for sampling during training
        self.paths = self.load_paths()
        if self.split == "training":
            indices_file = os.path.join(dataset_path, "class_indices.npz")
            if os.path.exists(indices_file):
                print("Loading precomputed indices")
                indices = dict(np.load(indices_file))
                pos_indices, neg_indices = indices["pos_indices"], indices["neg_indices"]
                sem_class_counts = indices["sem_class_counts"]
            else:
                print("Precomputing indices, one-time only!")
                pos_indices, neg_indices = [], []
                sem_class_counts = np.zeros((self.num_things, 1))
                for idx, path in enumerate(self.paths):
                    segment_data = dict(np.load(path))
                    if segment_data["gt_label"] == 1:
                        pos_indices.append(idx)
                    else:
                        neg_indices.append(idx)
                    sem_class_counts[segment_data["semantic_label"] - 1] += 1
                np.savez(
                    indices_file,
                    pos_indices=pos_indices,
                    neg_indices=neg_indices,
                    sem_class_counts=sem_class_counts
                )

            num_segments = len(self.paths)
            self.weights = torch.zeros(num_segments)
            self.weights[pos_indices] = num_segments / len(pos_indices)
            self.weights[neg_indices] = num_segments / len(neg_indices)
            self.sem_weights = np.divide(
                sem_class_counts.sum(), sem_class_counts, 
                out=np.zeros_like(sem_class_counts), where=sem_class_counts != 0)

    def load_paths(self):
        paths = []
        for seq in self.sequences:
            paths.extend([
                file for file in sorted(glob.glob(os.path.join(self.path, seq, "*.npz")))
            ])
        return paths

    def __len__(self):
        """
        Return the length of dataset
        """
        return len(self.paths)

    def __getitem__(self, index):
        segment_data = dict(np.load(self.paths[index]))

        # subsample / repeat points
        num_points = len(segment_data["xyz"])
        if num_points > self.n_points:
            chosen_idxs = np.random.choice(np.arange(num_points), self.n_points, replace=False)
        # TODO: something smarter than this
        else:
            if num_points < self.n_points:
                residual = self.n_points - num_points
                segment_data["indices"] = np.concatenate([
                    segment_data["indices"], -1 * np.ones(residual)])
                segment_data["xyz"] = np.vstack([
                    segment_data["xyz"], np.zeros((residual, 3))
                ])
                segment_data["semantic_features"] = np.vstack([
                    segment_data["semantic_features"], np.zeros((residual, 256))
                ])
            chosen_idxs = np.arange(0, self.n_points, dtype=np.int32)

        np.random.shuffle(chosen_idxs)
        segment_data["indices"] = segment_data["indices"][chosen_idxs]
        segment_data["xyz"] = segment_data["xyz"][chosen_idxs]
        segment_data["semantic_features"] = segment_data["semantic_features"][chosen_idxs]

        # Perform any data augmentation?
        return segment_data
    
    def collate_batch(self, batch):
        batch_size = len(batch)
        ans_dict = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                # skip string arrays since torch can't handle it
                if batch[0][key].dtype.kind in ("U", "S"):
                    continue
                ans_dict[key] = torch.cat(
                    [torch.from_numpy(batch[k][key][np.newaxis, ...]) for k in range(batch_size)], dim=0
                )
            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = torch.tensor(ans_dict[key], dtype=torch.int)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = torch.tensor(ans_dict[key], dtype=torch.float32)
        return ans_dict
