import glob
import os
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(self, dataset_path, split='training', n_points=1024, task_set=1):
        self.path = dataset_path
        self.split = split
        self.n_points = n_points

        # get thing classes based on task set
        if task_set == -1:
            config_file = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti-orig.yaml"
            self.num_things = 9
        else:
            config_file = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti.yaml"
            if task_set == 1:
                self.num_things = 4
            elif task_set == 2:
                self.num_things = 6

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            if task_set == -1:
                learning_map_inv = doc['learning_map_inv']
                learning_map = doc['learning_map']
            else:
                task_set_map = doc['task_set_map']
                learning_map_inv = task_set_map[task_set]['learning_map_inv']
                learning_map = task_set_map[task_set]['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

        self.semantic_classes = max(learning_map.values()) + 1

        # data augmentation flags
        self.random_jitter = False
        self.random_flip = False
        self.random_shuffle = True
        self.rotate_to_center = True

        if split == "training":
            self.sequences = ["{:02d}".format(i) for i in range(11) if i != 8]
        elif split == "validation":
            self.sequences = ["{:02d}".format(8)]
        else:
            assert False, "Split not implemented"

        # compute weights for sampling during training
        self.paths = self.load_paths()
        if self.split == "training":
            indices_file = os.path.join(dataset_path, "class_indices_regression.npz")
            if os.path.exists(indices_file):
                print("Loading precomputed indices")
                indices = dict(np.load(indices_file, allow_pickle=True))
                pos_indices, neg_indices = indices["pos_indices"], indices["neg_indices"]
                sem_label_to_pos_idx = indices["sem_label_to_pos_idx"].item()
                sem_label_to_idx = indices["sem_label_to_idx"].item()
                sem_class_counts = indices["sem_class_counts"]
            else:
                print("Precomputing indices, one-time only!")
                pos_indices, neg_indices = [], []
                sem_label_to_pos_idx = {i: [] for i in range(self.semantic_classes)}
                sem_label_to_idx = {i: [] for i in range(self.semantic_classes)}
                sem_class_counts = np.zeros((self.semantic_classes, 1))
                for idx, path in enumerate(self.paths):
                    segment_data = dict(np.load(path))
                    sem_label = segment_data["semantic_label"].item()
                    if segment_data["gt_label"] == 1:
                        pos_indices.append(idx)
                        sem_class_counts[sem_label] += 1
                        sem_label_to_pos_idx[sem_label].append(idx)
                    else:
                        neg_indices.append(idx)
                    sem_label_to_idx[sem_label].append(idx)
                sem_label_to_pos_idx_arr = np.array(sem_label_to_pos_idx)
                sem_label_to_idx_arr = np.array(sem_label_to_idx)
                np.savez(
                    indices_file,
                    pos_indices=pos_indices,
                    neg_indices=neg_indices,
                    sem_class_counts=sem_class_counts,
                    sem_label_to_pos_idx=sem_label_to_pos_idx_arr,
                    sem_label_to_idx=sem_label_to_idx_arr,
                )

            num_segments = len(self.paths)
            self.weights = torch.zeros(num_segments)
            self.weights[pos_indices] = num_segments / len(pos_indices)
            self.weights[neg_indices] = num_segments / len(neg_indices)

            # assign each segment weights using semantic labels
            self.sem_weights = torch.zeros(num_segments)
            for cls_id, cls_idxs in sem_label_to_pos_idx.items():
                # check if thing class as per original class vocabulary
                if cls_id > 0 and cls_id < self.num_things:
                    self.sem_weights[cls_idxs] = num_segments / len(cls_idxs)

            self.sem_loss_weights = np.divide(
                sem_class_counts[1:self.num_things].sum(), sem_class_counts[1:self.num_things], 
                out=np.zeros_like(sem_class_counts[1:self.num_things]), where=sem_class_counts[1:self.num_things] != 0)

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
            chosen_idxs = np.arange(0, num_points, dtype=np.int32)
            if num_points < self.n_points:
                extra_idxs = np.random.choice(
                    chosen_idxs, self.n_points - len(chosen_idxs), replace=True)
                chosen_idxs = np.concatenate([chosen_idxs, extra_idxs], axis=0)

        # Data augmentation
        if self.random_shuffle:
            np.random.shuffle(chosen_idxs)
            segment_data["indices"] = segment_data["indices"][chosen_idxs]
            segment_data["first_frame_xyz"] = segment_data["first_frame_xyz"][chosen_idxs]
            segment_data["xyz"] = segment_data["xyz"][chosen_idxs]
            segment_data["semantic_features"] = segment_data["semantic_features"][chosen_idxs]

        segment_data["first_frame_xyz"] = self.points_augmentation(segment_data["first_frame_xyz"])
        segment_data["xyz"] = self.points_augmentation(segment_data["xyz"])

        return segment_data

    def points_augmentation(self, points):
        # standardization: zero-mean
        if self.rotate_to_center:
            _mean = points.mean(axis=0)
            _theta = np.arctan2(_mean[1], _mean[0])
            _rot = np.array([[ np.cos(_theta), np.sin(_theta)],
                             [-np.sin(_theta), np.cos(_theta)]])
            points[:,:2] = _rot.dot((points[:,:2] - _mean[None,:2]).T).T

        if self.random_jitter:
            # parameters taken from modelnet_dataset.py of the PointNet++ project
            jitter = np.clip(0.01 * np.random.randn(points.shape[0], points.shape[1]), -0.05, 0.05)
            points += jitter

        if self.random_flip:
            # NOTE: randomly flip point cloud by y-axis (velo cs)
            # the intuition is that if we drive in the middle of a road
            # we might see the same thing on either side of the road
            # such phenomenon can be simulated using flipping
            if np.random.rand() > 0.5:
                points[:,1] = -points[:,1]

        return points

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
