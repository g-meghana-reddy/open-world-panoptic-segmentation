import os
import glob
import numpy as np


class SegmentDataset:
    def __init__(self, dataset_path, split='training', n_points=1024):
        self.path = dataset_path
        self.split = split
        self.n_points = n_points
        self.segment_files = self.load_paths()

    def load_paths(self):
        paths = []
        for seq in os.listdir(self.path):
            seq_dir = os.path.join(self.path, seq)
            if not os.path.isdir(seq_dir):
                continue
            if self.split == 'training' and seq != '08':
                paths.extend([
                    file for file in sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
                ])
            elif self.split == 'validation' and seq == '08':
                paths.extend([
                    file for file in sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
                ])
        return paths

    def __len__(self):
        """
        Return the length of dataset
        """
        return len(self.segment_files)

    def __getitem__(self, index):
        segment_data = dict(np.load(self.segment_files[index]))
        
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
                ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
