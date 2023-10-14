# System imports
import argparse
import sys
import os
import yaml

sys.path.append("../")

# Third-party imports
import glob
import numpy as np
from sklearn.cluster import DBSCAN

# Relative imports
from tree_utils import flatten_scores, flatten_indices
from utils import *


def evaluate(inds):
    return np.mean(objectness_objects[inds]).item()
    
    
def segment(id_, eps_list, cloud, original_indices=None, aggr_func='min'):
    if not all(eps_list[i] > eps_list[i+1] for i in range(len(eps_list)-1)):
        raise ValueError('eps_list is not sorted in descending order')
    # pick the first threshold from the list
    max_eps = eps_list[0]

    # Generate the indices if it does not exist
    if original_indices is None: original_indices = np.arange(cloud.shape[0])
    if isinstance(original_indices, list): original_indices = np.array(original_indices)
    
    # Spatial Segmentation: run DBSCAN to get clusters
    dbscan = DBSCAN(max_eps, min_samples=1).fit(cloud[original_indices,:])
    labels = dbscan.labels_

    # Evaluate every segment from the list of clusters
    indices, scores = [], []
    for unique_label in np.unique(labels):
        inds = original_indices[np.flatnonzero(labels == unique_label)]
        indices.append(inds.tolist())
        scores.append(evaluate(inds))

    # Return if we are done
    if len(eps_list) == 1: return indices, scores

    # Compute the hierarchical tree
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

        # If splitting is better the the aggr_score should be better than the current score
        if score < aggr_score:
            final_indices.append(fine_indices)
            final_scores.append(fine_scores)
        else: # otherwise
            final_indices.append(inds)
            final_scores.append(score)

    return final_indices, final_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=2)
    parser.add_argument("-d", "--dataset", help="Dataset", default='semantic-kitti')
    parser.add_argument("-s", "--sequence", help="Sequence", type=int, default=8)
    parser.add_argument(
        "-o", "--objsem_folder", help="Folder with object and semantic predictions", type=str, 
        default="/project_data/ramanan/mganesin/4D-PLS/test/4DPLS_original_params_original_repo_nframes1_1e-3_softmax/val_probs")
    parser.add_argument("-sd", "--save_dir", help="Output directory", type=str)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if args.task_set == 0:
        unk_labels = [7]
    elif args.task_set == 1:
        unk_labels = [10]
    elif args.task_set == 2:
        unk_labels = [16]
    elif args.task_set == -1:
        unk_labels = range(1,9)
    else:
        raise ValueError('Unknown task set: {}'.format(args.task_set))

    if args.dataset == 'semantic-kitti':
        seq = '{:02d}'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        scan_files = load_paths(scan_folder)

        if args.task_set == -1:
            config = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti-orig.yaml"
        elif args.task_set in (1, 2):
            config = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti.yaml"
        with open(config, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            if args.task_set == -1:
                learning_map_inv = doc['learning_map_inv']
                learning_map_doc = doc['learning_map']
            else:
                learning_map_inv = doc['task_set_map'][args.task_set]['learning_map_inv']
                learning_map_doc = doc['task_set_map'][args.task_set]['learning_map']
            learning_map = np.zeros((np.max([k for k in learning_map_doc.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_doc.items():
                learning_map[k] = v

            inv_learning_map = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), 
                                dtype=np.int32)
            for k, v in learning_map_inv.items():
                inv_learning_map[k] = v
 
    elif args.dataset == 'kitti-360':
        seq = '2013_05_28_drive_{:04d}_sync'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/data_3d_raw/' + seq + '/velodyne_points/data/'
        label_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/data_3d_raw_labels/' + seq + '/labels/'
        label_files = glob.glob(label_folder + '/*')
        file_ids = [x.split('/')[-1][:-6] for x in label_files]
        scan_files = sorted([
            os.path.join(scan_folder, file_id + '.bin') for file_id in file_ids
        ])

        if args.task_set == -1:
            config = "/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/kitti-360-orig.yaml"
        elif args.task_set in (1, 2):
            config = "/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/kitti-360.yaml"
        with open(config, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            if args.task_set == -1:
                learning_map_inv = doc['learning_map_inv']
                learning_map_doc = doc['learning_map']
            else:
                learning_map_inv = doc['task_set_map'][args.task_set]['learning_map_inv']
                learning_map_doc = doc['task_set_map'][args.task_set]['learning_map']
            learning_map = np.zeros((np.max([k for k in learning_map_doc.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_doc.items():
                learning_map[k] = v

            inv_learning_map = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), 
                                dtype=np.int32)
            for k, v in learning_map_inv.items():
                inv_learning_map[k] = v
        
    objsem_folder = args.objsem_folder
    objsem_files = load_paths(objsem_folder)

    sem_file_mask = []
    obj_file_mask = []
    ins_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c.' in file:
            obj_file_mask.append(idx)
        elif '_i.' in file:
            # Added for nframes=2
            frame_name = file.split('/')[-1]
            if len(frame_name.split('_')) < 4:
                ins_file_mask.append(idx)
        elif '_e.' not in file and '_u.' not in file and '_t.' not in file and '_pots.' not in file and '.ply' not in file and '_s.' not in file:
            # Added for nframes=2
            frame_name = file.split('/')[-1]
            if len(frame_name.split('_')) < 3:
                sem_file_mask.append(idx)
    
    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]
    instance_files = objsem_files[ins_file_mask]

    assert (len(semantic_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    for idx in tqdm(range(len(objectness_files))):
        segmented_dir = '{}/sequences/{:02d}/predictions/'.format(
            args.save_dir, args.sequence)
        if not os.path.exists(segmented_dir):
            os.makedirs(segmented_dir)
        segmented_file = os.path.join(segmented_dir, '{:07d}.label'.format(idx))

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

        # instances to overwrite
        instance_file = instance_files[idx]
        instances = np.load(instance_file)
        parent_dir, ins_base = os.path.split(instance_file)

        for unk_label in unk_labels:
            mask = labels == unk_label
            background_mask = labels != unk_label

            pts_velo_cs_objects = pts_velo_cs[mask]
            objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
            pts_indexes_objects = pts_indexes[mask]

            assert (len(pts_velo_cs_objects) == len(objectness_objects))

            if len(pts_velo_cs_objects) < 1:
                continue

            # segmentation with point-net
            id_ = 0
            eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]
            indices, scores = segment(id_, eps_list_tum, pts_velo_cs_objects[:, :3])

            flat_indices = flatten_indices(indices)
            # map from object_indexes to pts_indexes
            mapped_indices = []
            for indexes in flat_indices:
                mapped_indices.append(pts_indexes_objects[indexes].tolist())

            flat_scores = flatten_scores(scores)

            new_instance = instances.max() + 1
            for id, indices in enumerate(mapped_indices):
                instances[indices] = new_instance + id
            
        # Create .label files using the updated instance and semantic labels
        sem_labels = labels.astype(np.int32)
        inv_sem_labels = inv_learning_map[sem_labels]
        instances = np.left_shift(instances.astype(np.int32), 16)
        new_preds = np.bitwise_or(instances, inv_sem_labels)
        new_preds.tofile(segmented_file)