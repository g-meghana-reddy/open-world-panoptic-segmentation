import argparse
from sklearn.cluster import DBSCAN
import numpy as np
import pickle
import pdb
import time
import os
from tree_utils import flatten_scores, flatten_indices
import sys
from utils import *
import open3d as o3d
import glob
import yaml

import pdb


def evaluate(inds):
    return np.mean(objectness_objects[inds]).item()
    
    
def segment(id_, eps_list, cloud, original_indices=None, aggr_func='min'):
    if not all(eps_list[i] > eps_list[i+1] for i in range(len(eps_list)-1)):
        raise ValueError('eps_list is not sorted in descending order')
    # pick the first threshold from the list
    max_eps = eps_list[0]
    #
    if original_indices is None: original_indices = np.arange(cloud.shape[0])
    if isinstance(original_indices, list): original_indices = np.array(original_indices)
    # spatial segmentation
    dbscan = DBSCAN(max_eps, min_samples=1).fit(cloud[original_indices,:])
    labels = dbscan.labels_
    # evaluate every segment
    indices, scores = [], []
    for unique_label in np.unique(labels):
        inds = original_indices[np.flatnonzero(labels == unique_label)]
        indices.append(inds.tolist())
        scores.append(evaluate(inds))
    # return if we are done
    if len(eps_list) == 1: return indices, scores
    # expand recursively
    final_indices, final_scores = [], []
    for i, (inds, score) in enumerate(zip(indices, scores)):
        # focus on this segment
        fine_indices, fine_scores = segment(id_, eps_list[1:], cloud, inds)
        # flatten scores to get the minimum (keep structure)
        flat_fine_scores = flatten_scores(fine_scores)
        if aggr_func == 'min':
            aggr_score = np.min(flat_fine_scores)
        elif aggr_func == 'avg':
            aggr_score = np.mean(flat_fine_scores)
        elif aggr_func == 'sum':
            aggr_score = np.sum(flat_fine_scores)
        elif aggr_func == 'wavg':
            # compute a weighted average (each score is weighted by the number of points)
            flat_fine_indices = flatten_indices(fine_indices)
            sum_count, sum_score = 0, 0.0
            for indices, score in zip(flat_fine_indices, flat_fine_scores):
                sum_count += len(indices)
                sum_score += len(indices)*score
            aggr_score = float(sum_score)/sum_count
        elif aggr_func == 'd2wavg':
            # compute a weighted average (each score is weighted by the number of points)
            flat_fine_indices = flatten_indices(fine_indices)
            sum_count, sum_score = 0, 0.0
            for indices, score in zip(flat_fine_indices, flat_fine_scores):
                squared_dists = np.sum(cloud[inds,:]**2, axis=1)
                sum_count += np.sum(squared_dists)
                sum_score += np.sum(squared_dists * score)
            aggr_score = float(sum_score)/sum_count

        # COMMENTING THIS OUT BECAUSE OF ADDING SUM AS AN AGGR FUNC
        # assert(aggr_score <= 1 and aggr_score >= 0)

        # if splitting is better
        if score < aggr_score:
            final_indices.append(fine_indices)
            final_scores.append(fine_scores)
        else: # otherwise
            final_indices.append(inds)
            final_scores.append(score)
    return final_indices, final_scores


def vis_instance_o3d():
    # visualization
    pcd_objects = o3d.geometry.PointCloud()
    colors = np.zeros((len(pts_velo_cs_objects), 4))
    max_instance = len(flat_indices)
    print(f"point cloud has {max_instance + 1} clusters")
    colors_instance = plt.get_cmap("tab20")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[flat_indices[idx]] = colors_instance[idx]

    pcd_objects.points = o3d.utility.Vector3dVector(pts_velo_cs_objects[:, :3])
    pcd_objects.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_background = o3d.geometry.PointCloud()
    pcd_background.points = o3d.utility.Vector3dVector(pts_velo_cs[background_mask, :3])
    pcd_background.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([pcd_objects, pcd_background])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=2)
    parser.add_argument("-d", "--dataset", help="Dataset", default='semantic-kitti')
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, default='test/LOSP')
    parser.add_argument("-s", "--sequence", help="Sequence", type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    # if args.task_set == 0:
    #     unk_label = 7
    # elif args.task_set == 1:
    #     unk_label = 10
    # else:
    #     raise ValueError('Unknown task set: {}'.format(args.task_set))
    unk_labels = range(1, 9)
    # unk_labels = [1, 2, 3, 10]

    if args.dataset == 'semantic-kitti':
        seq = '{:02d}'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        scan_files = load_paths(scan_folder)
        objsem_folder = '/project_data/ramanan/mganesin/4D-PLS/test/4DPLS_original_params_original_repo_nframes1_1e-3_importance_None_str1_bigpug_1/val_probs'
        # objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/results/validation/val_preds_TS{}_original_params_1_frames_1e-3_importance_None_str1_bigpug_1_huseg_known/val_probs/'.format(args.task_set)
        label_folder = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/" + seq + "/labels/"
        label_files = sorted(glob.glob(label_folder + "*.label"))
        # objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/test/val_preds_4dpls_pretrained/val_probs/'

        config = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti-orig.yaml"
        with open(config, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map_doc = doc['learning_map']
            learning_map = np.zeros((np.max([k for k in learning_map_doc.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_doc.items():
                learning_map[k] = v

            inv_learning_map = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), 
                                dtype=np.int32)
            for k, v in learning_map_inv.items():
                inv_learning_map[k] = v

    elif args.dataset == 'kitti-raw':
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti-Raw/2011_09_26/'
        scan_files = glob.glob(scan_folder + '*/velodyne_points/data/*.bin')
        objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/test/val_preds_raw_TS{}/val_preds/'.format(args.task_set)
        
    elif args.dataset == 'kitti-360':
        seq = '2013_05_28_drive_{:04d}_sync'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/data_3d_raw/' + seq + '/velodyne_points/data/'
        label_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/data_3d_raw_labels/' + seq + '/labels/'
        label_files = glob.glob(label_folder + '/*')
        file_ids = [x.split('/')[-1][:-6] for x in label_files]
        scan_files = sorted([
            os.path.join(scan_folder, file_id + '.bin') for file_id in file_ids
        ])
        
        # objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/test/val_preds_TS{}_kitti360/val_probs/'.format(args.task_set)
        objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/results/validation/val_preds_TS1_kitti360_1_frames_1e-3_huseg_known_thresholded/val_probs/'#.format(args.task_set)

    objsem_files = load_paths(objsem_folder)

    sem_file_mask = []
    obj_file_mask = []
    ins_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c.' in file:
            obj_file_mask.append(idx)
        elif '_i.' in file:
            ins_file_mask.append(idx)
        elif '_e.' not in file and '_u.' not in file and '_t.' not in file and '_pots.' not in file and '.ply' not in file:
            sem_file_mask.append(idx)
    
    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]
    instance_files = objsem_files[ins_file_mask]

    assert (len(semantic_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    for idx in tqdm(range(len(objectness_files))):
        
        # load scan
        scan_file = scan_files[idx]
        pts_velo_cs = load_vertex(scan_file)
        pts_indexes = np.arange(len(pts_velo_cs))

        # load objectness
        objectness_file = objectness_files[idx]
        objectness = np.load(objectness_file)

        # labels
        label_file = semantic_files[idx]
        labels = np.load(label_file)

        # ground truths
        gt_file = label_files[idx]
        gt_label = np.fromfile(gt_file, dtype=np.int32)
        sem_gt = learning_map[gt_label & 0xFFFF]
        ins_gt = gt_label >> 16

        # instances to overwrite
        instance_file = instance_files[idx]
        instances = np.load(instance_file)
        parent_dir, ins_base = os.path.split(instance_file)
        segmented_file = os.path.join(parent_dir, ins_base.replace('_i', '_u'))

        #for unk_label in unk_labels:
        #     mask = labels == unk_label
        # background_mask = labels != unk_label
        mask = np.where(np.logical_and(labels > 0 , labels < 9))

        pts_velo_cs_objects = pts_velo_cs[mask]
        objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
        pts_indexes_objects = pts_indexes[mask]

        assert (len(pts_velo_cs_objects) == len(objectness_objects))

        if len(pts_velo_cs_objects) < 1:
            # np.save(segmented_file, instances)
            continue

        # mask out 4dpls instance predictions
        instances[mask] = 0

        # segmentation with point-net
        id_ = 0
        # eps_list = [2.0, 1.0, 0.5, 0.25]
        eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]
        indices, scores = segment(id_, eps_list_tum, pts_velo_cs_objects[:, :3])

        # flatten list(list(...(indices))) into list(indices)
        flat_indices = flatten_indices(indices)
        # map from object_indexes to pts_indexes
        mapped_indices = []
        for indexes in flat_indices:
            mapped_indices.append(pts_indexes_objects[indexes].tolist())

        # mapped_flat_indices = pts_indexes_objects
        flat_scores = flatten_scores(scores)

        # save results
        # np.savez_compressed(os.path.join(write_dir, seq + '_'+str(idx).zfill(6)),
        #                     instances=mapped_indices, segment_scores=flat_scores, allow_pickle = True)

        new_instance = instances.max() + 1
        
        for id, indices in enumerate(mapped_indices):
            # if flat_scores[idx] < 0.1:
            #     instances[indices] = 0
            # else:
            instances[indices] = new_instance + id
        
        # ========================================================
        # update the semantic predictions using match with GTs
        # copied from eval_np

        # Set up variables for coherence
        mask_extended = np.logical_and(labels > 0 , labels < 9)
        gt_mask = np.logical_and(sem_gt > 0 , sem_gt < 9)
        instance_pred_obj = instances * mask_extended.astype(np.int64)
        sem_gt_objects = sem_gt * gt_mask.astype(np.int64)
        ins_gt_objects = ins_gt * gt_mask.astype(np.int64)

        # generate the areas for each unique instance prediction
        offset = 2 ** 32
        unique_pred, counts_pred = np.unique(instance_pred_obj[instance_pred_obj > 0], return_counts=True)
        id2idx_pred = {id: idx1 for idx1, id in enumerate(unique_pred)}
        matched_pred = np.array([False] * unique_pred.shape[0])
        # print("Unique predictions:", unique_pred)

        # generate the areas for each unique instance gt_np
        unique_gt, counts_gt = np.unique(ins_gt_objects[ins_gt_objects > 0], return_counts=True)
        id2idx_gt = {id: idx1 for idx1, id in enumerate(unique_gt)}
        matched_gt = np.array([False] * unique_gt.shape[0])
        # print("Unique ground truth:", unique_gt)

        # generate intersection using offset
        valid_combos = np.logical_and(instance_pred_obj > 0, ins_gt_objects > 0)
        offset_combo = instance_pred_obj[valid_combos] + offset * ins_gt_objects[valid_combos]
        unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

        # generate an intersection map
        # count the intersections with over 0.5 IoU as TP
        gt_labels = unique_combo // offset
        pred_labels = unique_combo % offset
        gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
        pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.astype(np.float64) / unions.astype(np.float64)

        tp_indexes = ious > 0.5
        matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
        matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

        matched_pred_idxs = [id2idx_pred[id] for id in pred_labels[tp_indexes]]
        matched_gt_idxs = [id2idx_gt[id] for id in gt_labels[tp_indexes]]
        unmatched_pred_idxs = [
            idx for idx in range(unique_pred.shape[0])
            if idx not in matched_pred_idxs
        ]
        for unmatched_pred_idx in unmatched_pred_idxs:
            # Reject the segments which have lower IoU overlap 
            # with GT instance labels
            unmatched_ins = unique_pred[unmatched_pred_idx]
            unmatched_mask = instance_pred_obj == unmatched_ins
            # Ignore the rejected segments by assigning to unlabeled class
            instances[unmatched_mask] = -1.
            # labels[unmatched_mask] = 0.

        for (matched_pred_idx, matched_gt_idx) in zip(matched_pred_idxs, matched_gt_idxs):
            # find semantic label for transfer
            gt_ins = unique_gt[matched_gt_idx]
            sem_label = sem_gt_objects[ins_gt_objects == gt_ins]
            sem_label = np.bincount(sem_label).argmax()

            # transfer the label
            pred_ins = unique_pred[matched_pred_idx]
            labels[instances == pred_ins] = sem_label
        
        # ========================================================

        # Create .label files using the updated instance and semantic labels
        sem_labels = labels.astype(np.int32)
        inv_sem_labels = inv_learning_map[sem_labels]
        instances = np.left_shift(instances.astype(np.int32), 16)
        new_preds = np.bitwise_or(instances, inv_sem_labels)
        new_preds.tofile('{}/sequences/{:02d}/predictions/{:07d}.label'.format(
                args.output_dir, args.sequence, idx))
        
