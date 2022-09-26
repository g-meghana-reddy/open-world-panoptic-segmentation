import argparse
import pdb
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from tree_utils import flatten_scores, flatten_indices
from utils import *


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
    parser.add_argument("-s", "--sequence", help="Sequence", type=int, default=8)
    parser.add_argument("-o", "--obj_folder", help="Folder with object predictions", type=str, required=True)
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
    unk_labels = [1, 2, 3, 10]

    if args.dataset == 'semantic-kitti':
        seq = '{:02d}'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        scan_files = load_paths(scan_folder)
        gt_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/labels/'
        config_file = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti.yaml"

        output_dir = "results/predictions/TS{}_gt_semseg/sequences/{}/predictions".format(args.task_set, seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load objectness predictions
    obj_folder = args.obj_folder
    obj_files = load_paths(obj_folder)
    obj_file_mask = []
    for idx, file in enumerate(obj_files):
        if '_c.' in file:
            obj_file_mask.append(idx)
    objectness_files = obj_files[obj_file_mask]

    # load GT semantic seg predictions
    gt_files = load_paths(gt_folder)
    sem_file_mask = []
    for idx, file in enumerate(gt_files):
        if ".label" in file:
            sem_file_mask.append(idx)
    semantic_files = gt_files[sem_file_mask]

    assert (len(semantic_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    # open config file
    doc = yaml.safe_load(open(config_file, 'r'))
    class_remap = doc["task_set_map"][args.task_set]["learning_map"]
    inv_learning_map_doc = doc['task_set_map'][args.task_set]['learning_map_inv']

    # +100 hack making lut bigger just in case there are unknown labels
    maxkey = max(class_remap.keys())
    class_lut = np.zeros((maxkey + 100), dtype=np.int32)
    class_lut[list(class_remap.keys())] = list(class_remap.values())

    inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), 
                                dtype=np.int32)
    for k, v in inv_learning_map_doc.items():
        inv_learning_map[k] = v

    for idx in tqdm(range(len(objectness_files))):
        segmented_file = "{0:s}/{1:06d}.label".format(output_dir, idx)

        # load scan
        scan_file = scan_files[idx]
        pts_velo_cs = load_vertex(scan_file)
        pts_indexes = np.arange(len(pts_velo_cs))

        # load objectness
        objectness_file = objectness_files[idx]
        objectness = np.load(objectness_file)

        # labels
        label_file = semantic_files[idx]
        frame_labels = np.fromfile(label_file, dtype=np.uint32)
        labels = class_lut[frame_labels & 0xFFFF]
        instances = np.zeros_like(labels)

        for unk_label in unk_labels:
            mask = labels == unk_label
            background_mask = labels != unk_label

            pts_velo_cs_objects = pts_velo_cs[mask]
            objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
            pts_indexes_objects = pts_indexes[mask]

            assert (len(pts_velo_cs_objects) == len(objectness_objects))

            if len(pts_velo_cs_objects) < 1:
                np.save(segmented_file, frame_labels)
                continue

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
            new_instance = instances.max() + 1
            for idx, indices in enumerate(mapped_indices):
                instances[indices] = new_instance + idx

        # save LOSP outputs to file 
        instances = instances.astype(np.int32)
        new_preds = np.left_shift(instances, 16)

        labels = labels.astype(np.int32)
        inv_sem_labels = inv_learning_map[labels]
        new_preds = np.bitwise_or(new_preds, inv_sem_labels)
        new_preds.tofile(segmented_file)
