# Usage: python 4dpls_tracking_greedy.py -t 1 -p test/val_preds_TS1 -sd results/validation/TS1/ -dc data/SemanticKitti/semantic-kitti.yaml

# python 4dpls_tracking_greedy.py -t 1 -p ../../achakrav/4D-PLS/test/val_preds_TS1 -sd tracking/TS1/ -dc data/SemanticKitti/semantic-kitti.yaml

import numpy as np
import torch
import yaml
import os
from utils.greedy_tracker import GreedyTracker
from utils.tracking_utils import *
from scipy.optimize import linear_sum_assignment
import sys
import argparse
import time
from tqdm import tqdm
import collections
from functools import reduce

import pdb
import math


def main(FLAGS):
    split = 'valid'
    dataset = 'data/SemanticKitti'
    task_set = FLAGS.task_set
    save_dir = FLAGS.save_dir

    # thing classes
    if task_set == 0:
        unknown_sem_label = 7
    elif task_set == 1:
        unknown_sem_label = 10
    elif task_set == 2:
        unknown_sem_label = 16
    else:
        unknown_sem_label = 0

    if FLAGS.baseline:
        inst_ext = 'i'
    else:
        inst_ext = 'u'

    # get number of interest classes, and the label mappings class
    with open(FLAGS.data_cfg, 'r') as stream:
        doc = yaml.safe_load(stream)

        if task_set == -1:
            learning_map_doc = doc['learning_map']
            inv_learning_map_doc = doc['learning_map_inv']
        else:
            learning_map_doc = doc['task_set_map'][task_set]['learning_map']
            inv_learning_map_doc = doc['task_set_map'][task_set]['learning_map_inv']
    
    inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), 
                                dtype=np.int32)
    for k, v in inv_learning_map_doc.items():
        inv_learning_map[k] = v

    prediction_dir =  FLAGS.predictions
    if split == 'valid':
        prediction_path = '{}/val_probs'.format(prediction_dir)
    else:
        prediction_path = '{}/probs'.format(prediction_dir)

    test_sequences = FLAGS.sequences
    poses = []
    for sequence in test_sequences:
        calib = parse_calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt"))
        poses_f64 = parse_poses(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "poses.txt"), calib)
        poses.append([pose.astype(np.float32) for pose in poses_f64])

    total_time = 0.0
    for poses_seq, sequence in zip(poses, test_sequences):
        point_names = []
        point_paths = os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "velodyne")
        # populate the label names
        seq_point_names = sorted(
            [os.path.join(point_paths, fn) for fn in os.listdir(point_paths) if fn.endswith(".bin")])

        point_names.extend(seq_point_names)

        # output directory to write label files
        seq_save_dir = '{}/sequences/{:02d}/predictions/'.format(save_dir, sequence)
        if not os.path.exists(seq_save_dir):
            os.makedirs(seq_save_dir)
        
        idxs = np.array(len(point_names))
        
        # Project the points w.r.t to the first frame
        pose0 = poses_seq[0]

        mot_tracker = GreedyTracker()
        
        for idx, point_file in tqdm(enumerate(seq_point_names)):
            
            # Load the semantic predictions
            sem_path = os.path.join(prediction_path, '{0:02d}_{1:06d}.npy'.format(sequence,idx))
            sem_labels = np.load(sem_path)
            
            # Load the unknown instance predictions
            unknown_ins_path = os.path.join(prediction_path, '{0:02d}_{1:06d}_{2:s}.npy'.format(sequence, idx, inst_ext))
            unknown_ins_labels = np.load(unknown_ins_path)
            
            
            # Load /create the tracked unknown predictions
            unknown_track_path = os.path.join(save_dir, '{0:02d}_{1:06d}_t.npy'.format(sequence,idx))
            unknown_track_labels = unknown_ins_labels.copy()
            
            # Load the points and project them to camera coordinates
            points = np.fromfile(point_file, dtype=np.float32).reshape([-1,4])
            
            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            new_points = np.sum(np.expand_dims(hpoints, 2) * poses_seq[idx].T, axis=1)

            # Project the points w.r.t the first frame coordinates (but not for first frame)
            if idx > 0:
                new_coords = new_points[:, :3] - pose0[:3, 3]
                new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                points = new_coords[:, :3]
            else:
                points = new_points[:, :3]
            
            
            points = torch.from_numpy(points)
            point_indexes = torch.arange(len(points))

            mask = sem_labels == unknown_sem_label
            unknown_inst = unknown_ins_labels[mask]
            unknown_points = points[mask]
            point_indexes_unknown = point_indexes[mask]
            
            
            
            centers, center2points = [], {}
            for ins_id in np.unique(unknown_inst):
                
                ind = np.where(unknown_inst == ins_id)[0]
                
                # If we have N = 25 points or less then just drop those instances.
                if unknown_points[ind].shape[0] < 50:
                    unknown_track_labels[point_indexes_unknown[ind]] = 0
                    continue

                # For valid instances remove outliers which are two times the distance from the median.
                # Gives the valid points and corresponding indices
#                 refined_unknown_points, mask_ind = remove_outliers(unknown_points[ind])
#                 new_ind = ind[mask_ind]
#                 outliers = np.setdiff1d(ind, new_ind)
                
                # Calculate the new median with refined points
                #center = get_median_center_from_points(refined_unknown_points)
                center = get_median_center_from_points(unknown_points[ind])
                
                # Create a dictionary of {center: corresponding points}
                center = np.stack(center)
                centers.append(center)
                center2points[center.data.tobytes()] = point_indexes_unknown[ind]
                
                # Assign 0 to all outliers
                # unknown_track_labels[outliers] = 0
            
            # If there are no unknown points then assign the ids as it is
            if len(centers) == 0:
                unknown_ins_labels = unknown_ins_labels.astype(np.int32)
                new_preds = np.left_shift(unknown_ins_labels, 16)

                sem_labels = sem_labels.astype(np.int32)
                inv_sem_labels = inv_learning_map[sem_labels]
                new_preds = np.bitwise_or(new_preds, inv_sem_labels)


                new_preds.tofile('{0:s}/{1:06d}.label'.format(seq_save_dir, idx))
                continue
            centers = np.stack(centers)
            
            # Update the tracker with the new centers
            start_time = time.time()
            trackers = mot_tracker.update(centers)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            
            # Get the tracklets and update the instance id for the corresponding points
            for (t_id, trk_idx) in enumerate(trackers):
                if trackers[trk_idx][0] == math.inf:
                    continue
                inds = center2points[trackers[trk_idx].data.tobytes()]
                unknown_track_labels[inds] = trk_idx #t_id+1
        
            unknown_track_labels = unknown_track_labels.astype(np.int32)
            new_preds = np.left_shift(unknown_track_labels, 16)

            sem_labels = sem_labels.astype(np.int32)
            inv_sem_labels = inv_learning_map[sem_labels]
            new_preds = np.bitwise_or(new_preds, inv_sem_labels)

            new_preds.tofile('{0:s}/{1:06d}.label'.format(seq_save_dir, idx))
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./4dpls_tracking_greedy.py")

    parser.add_argument(
        '--sequences', '-s',
        dest='sequences',
        type=str,
        default='8'
    )

    parser.add_argument(
        '--predictions', '-p',
        dest='predictions',
        type=str,
        required=True
    )
    
    parser.add_argument(
      '--data_cfg',
      '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s'
    )

    parser.add_argument(
        '--save_dir', '-sd',
        dest='save_dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '--task_set', '-t',
        dest='task_set',
        type=int,
        default=1,
        required=True
    )

    parser.add_argument(
        '--baseline', '-b',
        dest='baseline',
        action="store_true",
        default=False
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.sequences = [int(x) for x in FLAGS.sequences.split(',')]

    main(FLAGS)