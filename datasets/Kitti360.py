#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling KITTI-360 dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Anirudh Chakravarthy and Meghana Ganesina - 04/04/2022
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import datetime as dt
import time
import numpy as np
import sys
import torch
from multiprocessing import Lock
import yaml

# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import *
from torch.utils.data import Sampler, get_worker_info


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class Kitti360Dataset(PointCloudDataset):
    """Class to handle SemanticKitti dataset."""

    def __init__(self, config, split='training', balance_classes=False, seqential_batch=False, return_unknowns=False):
        PointCloudDataset.__init__(self, 'Kitti360')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = 'data/Kitti360'

        # Type of task conducted on this dataset
        self.dataset_task = 'slam_segmentation'

        # Training or test set
        self.set = split

        # TODO: Validation sequence?
        # Get a list of sequences
        if not exists(
            join(self.path, 'data_3d_raw_labels',
                 '2013_05_28_drive_{:04d}_sync'.format(config.sequence), 'labels')):
            raise ValueError('Sequence does not have labels')
        # we support a single sequence inference for Kitti360, training is not supported
        self.sequences = [
            '2013_05_28_drive_{:04d}_sync'.format(config.sequence)]

        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = join(self.path, 'data_3d_raw', seq,
                             'velodyne_points', 'data')
            label_path = join(self.path, 'data_3d_raw_labels', seq, 'labels')

            frame_ids = set(vf[:-6]
                            for vf in listdir(label_path) if vf.endswith('.label'))
            frames = np.sort([
                vf[:-4] for vf in listdir(velo_path)
                if vf.endswith('.bin') and vf[:-4] in frame_ids
            ])
            self.frames.append(frames)

        self.seqential_batch = seqential_batch
        self.return_unknowns = return_unknowns

        self.task_set = config.task_set
        if self.task_set == 0:
            self.things = 3
        elif self.task_set == 1:
            self.things = 4
        elif self.task_set == 2:
            self.things = 6
        else:
            raise ValueError('No such task set: {}'.format(self.task_set))

        self.gpu_r = 1
        if config.n_test_frames > 1 and config.big_gpu:
            mem_gb = torch.cuda.get_device_properties(
                torch.device('cuda')).total_memory / (1024*1024*1024)
            self.gpu_r = mem_gb / 11.9

        # TODO: get label to names mapping and call init_labels if needed
        config_file = join(self.path, 'kitti-360.yaml')
        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            task_set_map = doc['task_set_map']
            learning_map_inv = task_set_map[self.task_set]['learning_map_inv']
            learning_map = task_set_map[self.task_set]['learning_map']
            self.learning_map = np.zeros(
                (np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros(
                (np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

            if self.task_set in (0, 1, 2):
                self.unknown_label = max(learning_map.values())

            if self.return_unknowns:
                self.unknown_label_to_names = {}
                self.unknown_label_names = []
                for k, v in learning_map.items():
                    # store names for unknown labels
                    if v == self.unknown_label:
                        label_name = all_labels[k]
                        self.unknown_label_to_names[k] = label_name
                        self.unknown_label_names.append(label_name)
                self.unknown_label_names = np.array(self.unknown_label_names)

        # Dict from labels to names
        self.label_to_names = {k: all_labels[v]
                               for k, v in learning_map_inv.items()}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Init variables
        self.calibration = np.eye(4)  # cam->velo transformation
        self.poses = []
        self.times = []
        self.all_inds = None
        self.class_proportions = None
        self.class_frames = []
        self.val_confs = []

        # Load everything
        self.load_calib_poses()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(
            np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        if seqential_batch:
            self.potentials = torch.from_numpy(
                np.zeros(self.all_inds.shape[0]))
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes = balance_classes

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius

        # shared epoch indices and classes (in case we want class balanced sampler)
        if self.set == 'training':
            N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
        else:
            N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        if self.seqential_batch:
            N = self.config.validation_size

        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        self.epoch_ins_labels = torch.from_numpy(
            np.zeros((N,), dtype=np.int32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()
        self.epoch_ins_labels.share_memory_()
        self.next_item = torch.from_numpy(np.zeros((1,), dtype=np.int64))

        self.worker_waiting = torch.tensor(
            [0 for _ in range(config.input_threads)], dtype=torch.int32)
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.frames)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        t = [time.time()]

        # Initiate concatanation lists
        c_list = []
        t_list = []  # times
        p_list = []
        f_list = []
        l_list = []
        u_list = []  # unknown labels
        ins_l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        f_inc_r_inds_list = []
        f_inc_r_mask_list = []
        val_labels_list = []
        val_unk_labels_list = []
        val_ins_labels_list = []
        val_center_label_list = []
        val_time_list = []
        batch_n = 0
        while True:

            t += [time.time()]

            with self.worker_lock:
                if self.epoch_i >= self.epoch_inds.shape[0]:
                    self.epoch_i = 0
                # Get potential minimum
                ind = int(self.epoch_inds[self.epoch_i])
                wanted_label = int(self.epoch_labels[self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1

            s_ind, f_ind = self.all_inds[ind]

            t += [time.time()]

            #########################
            # Merge n_frames together
            #########################

            # Initiate merged points
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int32)
            merged_ins_labels = np.zeros((0,), dtype=np.int32)
            merged_coords = np.zeros((0, 9), dtype=np.float32)
            if self.return_unknowns:
                merged_unk_labels = np.zeros((0,), dtype=np.int32)

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4))
            p_origin[0, 3] = 1
            pose_origin_inv = np.linalg.inv(self.poses[s_ind][0])
            pose0 = self.poses[s_ind][f_ind].dot(pose_origin_inv)
            p0 = p_origin.dot(pose0.T)[:, :3]
            p0 = np.squeeze(p0)
            o_pts = None
            o_labels = None
            o_ins_labels = None
            o_center_labels = None
            o_times = None
            if self.return_unknowns:
                o_unk_labels = None

            t += [time.time()]

            num_merged = 0
            f_inc = 0
            f_inc_points = []
            while num_merged < self.config.n_frames and f_ind - f_inc >= 0:

                # Current frame pose
                pose = self.poses[s_ind][f_ind - f_inc]

                # Select frame only if center has moved far away (more than X meter). Negative value to ignore
                X = -1.0
                if X > 0:
                    diff = p_origin.dot(
                        pose.T)[:, :3] - p_origin.dot(pose0.T)[:, :3]
                    if num_merged > 0 and np.linalg.norm(diff) < num_merged * X:
                        f_inc += 1
                        continue

                seq_path = join(self.path, 'data_3d_raw', self.sequences[s_ind],
                                'velodyne_points', 'data')
                velo_file = join(seq_path, self.frames[s_ind][f_ind] + '.bin')

                center_file = None
                if self.set == 'test':
                    label_file = None
                else:
                    label_folder = join(self.path, 'data_3d_raw_labels',
                                        self.sequences[s_ind], 'labels')
                    label_file = join(
                        label_folder, self.frames[s_ind][f_ind] + '.label')

                # Read points
                frame_points = np.fromfile(velo_file, dtype=np.float32)
                points = frame_points.reshape((-1, 4))

                center_labels = np.zeros(
                    (points.shape[0], 4), dtype=np.float32)
                if self.set == 'test':
                    # Fake labels
                    sem_labels = np.zeros((points.shape[0],), dtype=np.int32)
                    ins_labels = np.zeros((points.shape[0],), dtype=np.int32)
                    if self.return_unknowns:
                        unk_labels = np.zeros(
                            (points.shape[0],), dtype=np.int32)
                else:
                    # Read labels
                    frame_labels = np.fromfile(label_file, dtype=np.int32)
                    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                    ins_labels = frame_labels >> 16
                    ins_labels = ins_labels.astype(np.int32)
                    sem_labels[sem_labels == -1] = 0
                    sem_labels = self.learning_map[sem_labels]

                    if self.return_unknowns:
                        unk_labels = frame_labels & 0xFFFF  # semantic label in lower half

                # Apply pose (without np.dot to avoid multi-threading)
                hpoints = np.hstack(
                    (points[:, :3], np.ones_like(points[:, :1])))
                new_points = np.sum(np.expand_dims(
                    hpoints, 2) * pose.T, axis=1)

                # In case of validation, keep the original points in memory
                if self.set in ['training', 'validation', 'test'] and f_inc == 0:
                    o_pts = new_points[:, :3].astype(np.float32)
                    o_labels = sem_labels.astype(np.int32)
                    o_center_labels = center_labels
                    o_ins_labels = ins_labels.astype(np.int32)
                    if self.return_unknowns:
                        o_unk_labels = unk_labels.astype(np.int32)

                if self.set in ['training', 'validation', 'test'] and self.config.n_test_frames > 1 and f_inc > 0:
                    f_inc_points.append(new_points[:, :3].astype(np.float32))

                # In case radius smaller than 50m, chose new center on a point of the wanted class or not
                if self.in_R < 50.0 and f_inc == 0:
                    if self.balance_classes:
                        wanted_ind = np.random.choice(
                            np.where(sem_labels == wanted_label)[0])
                    else:
                        wanted_ind = np.random.choice(new_points.shape[0])
                    p0 = new_points[wanted_ind, :3]

                # Eliminate points further than config.in_radius
                mask = np.sum(
                    np.square(new_points[:, :3] - p0), axis=1) < self.in_R ** 2

                # during training
                if self.set in ['training', 'validation'] and f_inc > 0 and self.config.n_test_frames == 1:
                    # eliminate points which are not belong to any instance class for future frame

                    if self.config.sampling == 'objectness':
                        mask = ((sem_labels > 0) & (
                            sem_labels < self.things) & mask)
                    elif self.config.sampling == 'importance':
                        n_points_to_sample = np.sum(
                            (sem_labels > 0) & (sem_labels < self.things))
                        probs = (center_labels[:, 0] + 0.1)
                        idxs = np.random.choice(
                            np.arange(center_labels.shape[0]), n_points_to_sample, p=probs/np.sum(probs))
                        new_mask = np.zeros_like(mask)
                        new_mask[idxs] = 1
                        mask = (new_mask & mask)
                    else:
                        pass

                if self.set in ['validation', 'test'] and self.config.n_test_frames > 1 and f_inc > 0:
                    test_path = join('test', self.config.saving_path.split(
                        '/')[-1] + str(self.config.n_test_frames))
                    if self.set == 'validation':
                        test_path = join(test_path, 'val_probs')
                    else:
                        test_path = join(test_path, 'probs')

                    if self.config.sampling == 'objectness':

                        filename = '{:s}_{:07d}.npy'.format(
                            self.sequences[s_ind], f_ind-f_inc)
                        file_path = join(test_path, filename)
                        label_pred = None
                        counter = 0
                        while label_pred is None:
                            try:
                                label_pred = np.load(file_path)
                            except:
                                print('label cannot be read {}'.format(file_path))
                                counter += 1
                                if counter > 5:
                                    break
                                continue
                        # eliminate points which are not belong to any instance class for future frame
                        if label_pred is not None:
                            mask = ((label_pred > 0) & (
                                label_pred < self.things) & mask)
                    elif self.config.sampling == 'importance':
                        filename = '{:s}_{:07d}_c.npy'.format(
                            self.sequences[s_ind], f_ind - f_inc)
                        file_path = join(test_path, filename)
                        center_pred = None
                        counter = 0
                        while center_pred is None:
                            try:
                                center_pred = np.load(file_path)
                            except:
                                time.sleep(2)
                                counter += 1
                                if counter > 5:
                                    break
                                continue
                        if center_pred is not None:
                            n_points_to_sample = int(np.sum(mask)/10)
                            decay_ratios = np.array(
                                [np.exp(i/self.config.n_test_frames) for i in range(1, self.config.n_test_frames)])
                            decay_ratios = decay_ratios * \
                                ((self.config.n_test_frames-1) /
                                 np.sum(decay_ratios))  # normalize sums
                            if self.config.decay_sampling == 'forward':
                                n_points_to_sample = int(
                                    n_points_to_sample*decay_ratios[f_inc-1])
                            if self.config.decay_sampling == 'backward':
                                n_points_to_sample = int(
                                    n_points_to_sample * decay_ratios[-f_inc])
                            probs = (center_pred[:, 0] + 0.1)
                            idxs = np.random.choice(np.arange(center_pred.shape[0]), n_points_to_sample,
                                                    p=(probs / np.sum(probs)))
                            new_mask = np.zeros_like(mask)
                            new_mask[idxs] = 1
                            mask = (new_mask & mask)
                    else:
                        pass

                mask_inds = np.where(mask)[0].astype(np.int32)

                # Shuffle points
                rand_order = np.random.permutation(mask_inds)
                new_points = new_points[rand_order, :3]
                sem_labels = sem_labels[rand_order]
                ins_labels = ins_labels[rand_order]
                center_labels = center_labels[rand_order]
                if self.return_unknowns:
                    unk_labels = unk_labels[rand_order]
                # Place points in original frame reference to get coordinates
                if f_inc == 0:
                    new_coords = points[rand_order, :]
                else:
                    # We have to project in the first frame coordinates
                    new_coords = new_points - pose0[:3, 3]
                    new_coords = np.sum(np.expand_dims(
                        new_coords, 2) * pose0[:3, :3], axis=1)
                    new_coords = np.hstack(
                        (new_coords, points[rand_order, 3:]))

                d_coords = new_coords.shape[1]
                d_centers = center_labels.shape[1]
                times = np.ones((center_labels.shape[0], 1)) * f_inc
                times = times.astype(np.float32)
                new_coords = np.hstack((new_coords, center_labels))
                new_coords = np.hstack((new_coords, times))
                # Increment merge count

                if f_inc == 0 or (hasattr(self.config, 'stride') and f_inc % self.config.stride == 0):
                    merged_points = np.vstack((merged_points, new_points))
                    merged_labels = np.hstack((merged_labels, sem_labels))
                    merged_ins_labels = np.hstack(
                        (merged_ins_labels, ins_labels))
                    merged_coords = np.vstack((merged_coords, new_coords))
                    if self.return_unknowns:
                        merged_unk_labels = np.hstack(
                            (merged_unk_labels, unk_labels))

                num_merged += 1
                f_inc += 1

            t += [time.time()]

            #########################
            # Merge n_frames together
            #########################

            # Subsample merged frames
            in_pts, in_fts, in_lbls, in_slbls = grid_subsampling(
                merged_points,
                features=merged_coords,
                labels=merged_labels,
                ins_labels=merged_ins_labels,
                sampleDl=self.config.first_subsampling_dl)
            if self.return_unknowns:
                _, _, in_unk_lbls, _ = grid_subsampling(
                    merged_points,
                    features=merged_coords,
                    labels=merged_unk_labels,
                    ins_labels=merged_ins_labels,
                    sampleDl=self.config.first_subsampling_dl)

            t += [time.time()]

            # Number collected
            n = in_pts.shape[0]

            # Safe check
            if n < 2:
                continue

            # Randomly drop some points (augmentation process and safety for GPU memory consumption)
            if n > self.max_in_p * self.gpu_r:

                if self.config.sampling == 'density':
                    # density based sampling
                    r = self.config.first_subsampling_dl * self.config.conv_radius
                    neighbors = batch_neighbors(
                        in_pts, in_pts, [in_pts.shape[0]], [in_pts.shape[0]], r)
                    densities = np.sum(neighbors == in_pts.shape[0], 1)
                    input_inds = np.random.choice(n, size=int(
                        self.max_in_p*self.gpu_r), replace=False, p=(densities)/np.sum(densities))
                else:
                    # random sampling
                    input_inds = np.random.choice(n, size=int(
                        self.max_in_p*self.gpu_r), replace=False)

                in_pts = in_pts[input_inds, :]
                in_fts = in_fts[input_inds, :]
                in_lbls = in_lbls[input_inds, :]
                in_slbls = in_slbls[input_inds, :]
                n = input_inds.shape[0]
                if self.return_unknowns:
                    in_unk_lbls = in_unk_lbls[input_inds, :]

            in_times = in_fts[:, 8]  # hard coded last dim
            in_cts = in_fts[:, d_coords:8]
            in_fts = in_fts[:, 0:d_coords]

            t += [time.time()]

            # Before augmenting, compute reprojection inds (only for validation and test)
            if self.set in ['training', 'validation', 'test']:

                # get val_points that are in range
                radiuses = np.sum(np.square(o_pts - p0), axis=1)
                reproj_mask = radiuses < (0.99 * self.in_R) ** 2

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(
                    o_pts[reproj_mask, :], return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)

            else:
                proj_inds = np.zeros((0,))
                reproj_mask = np.zeros((0,))

            if self.set in ['training', 'validation', 'test'] and self.config.n_test_frames > 1:
                f_inc_proj_inds = []
                f_inc_reproj_mask = []
                for i in range(len(f_inc_points)):
                    # get val_points that are in range
                    radiuses = np.sum(np.square(f_inc_points[i] - p0), axis=1)
                    f_inc_reproj_mask.append(
                        radiuses < (0.99 * self.in_R) ** 2)

                    # Project predictions on the frame points
                    search_tree = KDTree(in_pts, leaf_size=100)
                    f_inc_proj_inds.append(search_tree.query(
                        f_inc_points[i][f_inc_reproj_mask[-1], :], return_distance=False))
                    f_inc_proj_inds[-1] = np.squeeze(
                        f_inc_proj_inds[-1]).astype(np.int32)

            t += [time.time()]

            if self.set in ['training', 'validation', 'test']:
                # Data augmentation
                _, scale, R = self.augmentation_transform(in_pts)
            else:
                in_pts, scale, R = self.augmentation_transform(in_pts)

            t += [time.time()]

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                in_fts[:, 3:] *= 0

            # Stack batch
            c_list += [in_cts]
            t_list += [in_times]
            p_list += [in_pts]
            f_list += [in_fts]
            l_list += [np.squeeze(in_lbls)]
            if self.return_unknowns:
                u_list += [np.squeeze(in_unk_lbls)]
            ins_l_list += [np.squeeze(in_slbls)]
            fi_list += [[s_ind, f_ind]]
            p0_list += [p0]
            s_list += [scale]
            R_list += [R]
            r_inds_list += [proj_inds]
            r_mask_list += [reproj_mask]
            if self.config.n_test_frames > 1:
                f_inc_r_inds_list += [f_inc_proj_inds]
                f_inc_r_mask_list += [f_inc_reproj_mask]
            else:
                f_inc_r_inds_list = []
                f_inc_r_mask_list = []
            val_labels_list += [o_labels]
            if self.return_unknowns:
                val_unk_labels_list += [o_unk_labels]
            val_ins_labels_list += [o_ins_labels]
            # original centers (all of them)
            val_center_label_list += [o_center_labels]

            t += [time.time()]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################
        # centers = np.concatenate(c_list, axis=0) if not self.set  in ['validation', 'test'] else np.concatenate(val_center_label_list, axis=0)
        centers = np.concatenate(c_list, axis=0)
        times = np.concatenate(t_list, axis=0)
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        if self.return_unknowns:
            unk_labels = np.concatenate(u_list, axis=0)
        # ins_labels = np.concatenate(ins_l_list, axis=0) if not self.set in ['validation', 'test'] else np.concatenate(val_ins_labels_list, axis=0)
        ins_labels = np.concatenate(ins_l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0]
                                 for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        if o_center_labels is not None:
            val_center_labels = np.concatenate(val_center_label_list, axis=0)
        else:
            val_center_labels = np.zeros_like(centers)
        if o_ins_labels is not None:
            val_ins_labels = np.concatenate(val_ins_labels_list, axis=0)
        else:
            val_ins_labels = np.zeros_like(ins_labels)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(
            stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 2:3]))
        elif self.config.in_features_dim == 3:
            # Use height + time
            if self.config.n_test_frames > 2:
                ratio = 1/(self.config.n_test_frames-1)
            else:
                ratio = 1
            stacked_features = np.hstack(
                (stacked_features, features[:, 2:3], np.expand_dims(times, axis=1)))
        elif self.config.in_features_dim == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:3]))
        elif self.config.in_features_dim == 5:
            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError(
                'Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        t += [time.time()]

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels.astype(np.int64),
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, frame_inds, frame_centers, centers, times, ins_labels.astype(np.int64), r_inds_list,
                       r_mask_list, f_inc_r_inds_list, f_inc_r_mask_list, val_labels_list, val_center_labels, val_ins_labels]
        if self.return_unknowns:
            input_list += [unk_labels.astype(np.int64), val_unk_labels_list]

        t += [time.time()]

        return [self.config.num_layers] + input_list

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        # Read Calib
        calib_file = join(self.path, 'calibration', 'calib_cam_to_velo.txt')
        self.calibration[:-1] = np.loadtxt(calib_file,
                                           dtype=np.float32).reshape(-1, 4)

        for seq in self.sequences:
            seq_folder = join(self.path, 'data_3d_raw', seq, 'velodyne_points')
            pose_folder = join(self.path, 'data_poses', seq)

            # Read times
            self.times.append(
                self.parse_timestamps(join(seq_folder, 'timestamps.txt')))

            # Read poses
            poses_f64 = self.parse_poses(
                join(pose_folder, 'cam0_to_world.txt'), self.calibration)
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

        ###################################
        # Prepare the indices of all frames
        ###################################
        seq_inds = np.hstack(
            [np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack(
            [np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

    def parse_timestamps(self, filename):
        """ read timestamps file from given filename

            Returns
            -------
            list
                list of timestamps
        """
        seq = filename.split('/')[-3]
        seq_idx = self.sequences.index(seq)
        seq_frames = set(self.frames[seq_idx])

        timestamps = []
        with open(filename, 'r') as f:
            prev_t = None
            for i, line in enumerate(f.readlines()):
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                idx = '{:010d}'.format(i)
                if idx not in seq_frames:
                    continue
                if prev_t is None:
                    delta_t = 0
                else:
                    delta_t = (t - prev_t).total_seconds() + timestamps[-1]
                timestamps.append(delta_t)
                prev_t = t
        return timestamps

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename
            Returns
            -------
            list
                list of poses (Tr[velo_t -> velo_t]) as 4x4 numpy arrays.
        """
        seq = filename.split('/')[-2]
        seq_idx = self.sequences.index(seq)
        seq_frames = set(self.frames[seq_idx])

        TrCamToVelo = calibration
        TrVeloToCam = np.linalg.inv(TrCamToVelo)
        TrWorldToCamFrame0 = None  # first frame pose (inverse)

        poses = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                values = [float(v) for v in line.strip().split()]
                idx = '{:010d}'.format(int(values[0]))
                if idx not in seq_frames:
                    continue
                TrCamToWorld = np.eye(4)
                TrCamToWorld[0, 0:4] = values[1:5]
                TrCamToWorld[1, 0:4] = values[5:9]
                TrCamToWorld[2, 0:4] = values[9:13]

                # transform Tr[cam_t->world] to Tr[velo_t->velo_0]
                if len(poses):
                    TrCamToCam = np.matmul(TrWorldToCamFrame0, TrCamToWorld)
                    pose = np.matmul(TrCamToVelo, np.matmul(
                        TrCamToCam, TrVeloToCam))
                # already at frame 0
                else:
                    TrWorldToCamFrame0 = np.linalg.inv(TrCamToWorld)
                    pose = np.eye(4)
                poses.append(pose)
        return poses
