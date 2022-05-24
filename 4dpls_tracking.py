import numpy as np
import torch
import yaml
import os
from utils.tracking_utils import *
from utils.kalman_filter import KalmanBoxTracker
from scipy.optimize import linear_sum_assignment
import sys
import argparse
import time
from tqdm import tqdm

from functools import reduce

from AB3DMOT.AB3DMOT_libs.model import AB3DMOT


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
        unknown_sem_label = 0

    # get number of interest classes, and the label mappings class
    with open(FLAGS.data_cfg, 'r') as stream:
        doc = yaml.safe_load(stream)
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

        mot_tracker = AB3DMOT(iou_threshold=1e-4) 
        for idx, point_file in tqdm(enumerate(point_names)):
            sem_path = os.path.join(prediction_path, '{0:02d}_{1:07d}.npy'.format(sequence,idx))
            unknown_ins_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_u.npy'.format(sequence,idx))
            unknown_track_path = os.path.join(save_dir, '{0:02d}_{1:07d}_t.npy'.format(sequence,idx))

            sem_labels = np.load(sem_path)
            unknown_ins_labels = np.load(unknown_ins_path)
            unknown_trk_labels = unknown_ins_labels.copy()
            points = np.fromfile(point_file, dtype=np.float32).reshape([-1,4])
            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            new_points = np.sum(np.expand_dims(hpoints, 2) * poses_seq[idx].T, axis=1)

            points = new_points[:, :3]
            points = torch.from_numpy(points)
            point_indexes = torch.arange(len(points))

            mask = sem_labels == unknown_sem_label
            unknown_inst = unknown_ins_labels[mask]
            unknown_points = points[mask]
            point_indexes_unknown = point_indexes[mask]

            bboxes, bbox2points = [], {}
            ins_ids = np.unique(unknown_inst)
            for ins_id in ins_ids:
                ind = np.where(unknown_inst == ins_id)
                point_ind = point_indexes_unknown[ind]

                if ind[0].shape[0] >= 24:
                    bbox, kalman_bbox = get_bbox_from_points(unknown_points[ind])
                    bboxes.append(kalman_bbox.numpy())
                    #bbox2points[kalman_bbox.numpy().tobytes()] = point_ind

            if len(bboxes):
                dets = np.stack(bboxes, axis=0)
            else:
                dets = np.zeros((0, 7))

            additional_info = np.zeros((dets.shape[0], 1))
            dets_all = {'dets': dets, 'info': additional_info}

            # important
            start_time = time.time()
            trackers = mot_tracker.update(dets_all)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            
#TODO: Remove Outliers!!!!!

#             save_trk_file = os.path.join('.', '%06d.txt' % idx)
#             save_bbox2point_file = os.path.join('.', '%06d_bbox2points.txt' % idx)
#             save_trk_file = open(save_trk_file, 'w')
#             save_bbox2point_file = open(save_bbox2point_file, 'w')
            for d in trackers:
                bbox3d_tmp = d[0:7].astype(np.float32) # h, w, l, x, y, z, theta in camera coord 
                id_tmp = d[7]
                ori_tmp = d[8]    
                
                x1, y1, z1, x2, y2, z2 = kalman_box_to_eight_point(bbox3d_tmp)
                poi_inds = reduce(np.intersect1d, (
                    np.where(points[:, 0] > x1),
                    np.where(points[:, 0] < x2),
                    np.where(points[:, 1] > y1),
                    np.where(points[:, 1] < y2),
                    np.where(points[:, 2] > z1),
                    np.where(points[:, 2] < z2)
                ))
                
                unknown_trk_labels[poi_inds] = id_tmp
#                 if bbox3d_tmp.tobytes() in bbox2points:
#                     ind = bbox2points[bbox3d_tmp.tobytes()]
#                     unknown_trk_labels[ind] = id_tmp
                    
#                 str_to_srite = '%f %f %f %f %f %f %f %d\n' % (
#                     bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3],
#                     bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], id_tmp)
#                 str_to_srite += str(poi_inds)
#                 save_trk_file.write(str_to_srite)
            #save_bbox2point_file.write(str(bbox2points))

            # np.save(unknown_track_path, unknown_trk_labels)
            unknown_trk_labels = unknown_trk_labels.astype(np.int32)
            new_preds = np.left_shift(unknown_trk_labels, 16)

            sem_labels = sem_labels.astype(np.int32)
            inv_sem_labels = inv_learning_map[sem_labels]
            new_preds = np.bitwise_or(new_preds, inv_sem_labels)

            new_preds.tofile('{}/sequences/{:02d}/predictions/{:06d}.label'.format(
                save_dir, sequence, idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./stitch_tracklets.py")

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

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.sequences = [int(x) for x in FLAGS.sequences.split(',')]

    main(FLAGS)