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

def evaluate_iou_based_objectness(pred_inds):
    # predictions
    pred_counts = pred_inds.shape[0]
    pred_indices = pts_indexes_objects[pred_inds]

    # groundtruth
    gt_indices_local = np.arange(gt_instance_indexes.shape[0])
    gt_inst_ids = gt_instance_ids[gt_indices_local]

    unique_gt_ids, unique_gt_counts = np.unique(gt_inst_ids, return_counts = True)
    
    ious = []
    for idx, gt_ins in enumerate(unique_gt_ids):
        gt_ind = np.where(gt_inst_ids == gt_ins)
        intersections = len(set(pred_indices) & set(gt_instance_indexes[gt_ind]))
        gt_counts = unique_gt_counts[idx]
        union = gt_counts + pred_counts - intersections
        iou = intersections / union
        ious.append(iou)

    ious = np.array(ious)
    max_iou = np.max(ious)
    
    return max_iou
    
    
def segment(id_, eps_list, cloud, original_indices=None, aggr_func='min'):
    if not all(eps_list[i] > eps_list[i+1] for i in range(len(eps_list)-1)):
        raise ValueError('eps_list is not sorted in descending order')
    
    # Pick the first threshold from the list
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
        scores.append(evaluate_iou_based_objectness(inds))

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



def compute_hierarchical_tree(eps_list, points_3d, original_indices= None):
    '''We compute segments from hierarchical tree and compute 
        the objectness scores to perform tree cut.'''
    
    # If we are at the end of the hierarchical tree then return the empty node.
    if len(eps_list) == 0: return []

    # Pick the first threshold from the eps_list.
    max_eps = eps_list[0]

    # Compute the original indices for each point at level 0.
    # Reassign the original_indices in further levels to keep a track of them.
    if isinstance(original_indices, list): original_indices = np.array(original_indices)

    # Perform DBSCAN on the current set of 3D points to get segments at the current level L.
    dbscan = DBSCAN(max_eps, min_samples=1).fit(points_3d[original_indices,:])
    labels = dbscan.labels_

    # Compute the indices and score per segment and store them as part of the tree node.
    segments = []
    for unique_label in np.unique(labels):

        inds = original_indices[np.flatnonzero(labels == unique_label)]
        score = evaluate_iou_based_objectness(inds)
        segment = TreeSegment(inds, score)
        segment.child_segments = compute_hierarchical_tree(eps_list[1:], points_3d, inds)
        segments.append(segment)

    return segments


def segment_tree_traverse(segment_tree, level):
    if len(segment_tree.curr_segment_data.indices) == 0:
        return

    for segment in segment_tree.child_segments:  
        segment_tree_traverse(segment, level+1)

    print("----------------------------------------------")
    print("Level: {}".format(level))
    print("Segment indices: ", segment_tree.curr_segment_data.indices)
    print("Segment score: ", segment_tree.curr_segment_data.score)
    print("----------------------------------------------")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=2)
    parser.add_argument("-d", "--dataset", help="Dataset", default='semantic-kitti')
    parser.add_argument(
        "-o", "--objsem_folder", help="Folder with object and semantic predictions", type=str, 
        default="/project_data/ramanan/mganesin/4D-PLS/test/4DPLS_original_params_original_repo_nframes1_1e-3_softmax/val_probs")
    parser.add_argument("-sd", "--save_dir", help="Output directory", type=str)
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

    objsem_folder = args.objsem_folder
    if args.dataset == 'semantic-kitti':
        seq = '{:02d}'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        scan_files = load_paths(scan_folder)
        label_folder = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/" + seq + "/labels/"
        label_files = sorted(glob.glob(label_folder + "*.label"))

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
    softmax_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c.' in file:
            obj_file_mask.append(idx)
        elif '_i.' in file:
            ins_file_mask.append(idx)
        elif '_s.' in file:
            softmax_file_mask.append(idx)
        elif '_e.' not in file and '_u.' not in file and '_t.' not in file and '_pots.' not in file and '.ply' not in file:
            sem_file_mask.append(idx)
    
    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]
    instance_files = objsem_files[ins_file_mask]
    softmax_files = objsem_files[softmax_file_mask]

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

        # softmax scores
        softmax_file = softmax_files[idx]
        softmax_scores = np.load(softmax_file)

        # ground truths
        gt_file = label_files[idx]
        gt_label = np.fromfile(gt_file, dtype=np.int32)
        sem_gt = learning_map[gt_label & 0xFFFF]
        ins_gt = gt_label >> 16

        # instances to overwrite
        instance_file = instance_files[idx]
        instances = np.load(instance_file)
        parent_dir, ins_base = os.path.split(instance_file)

        pred_mask = np.where(np.logical_and(labels > 0 , labels < 9))
        gt_mask = np.where(np.logical_and(sem_gt > 0 , sem_gt < 9))

        pts_velo_cs_objects = pts_velo_cs[pred_mask]
        objectness_objects = objectness[pred_mask]  # todo: change objectness_objects into a local variable
        pts_indexes_objects = pts_indexes[pred_mask]

        gt_instance_ids = ins_gt[gt_mask]
        gt_instance_indexes = np.arange(ins_gt.shape[0])[gt_mask]

        assert (len(pts_velo_cs_objects) == len(objectness_objects))

        if len(pts_velo_cs_objects) < 1:
            continue

        # mask out 4dpls instance predictions
        instances[pred_mask] = 0

        # segmentation with point-net
        id_ = 0
        # eps_list = [2.0, 1.0, 0.5, 0.25]
        eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]
        indices, scores = segment(id_, eps_list_tum, pts_velo_cs_objects[:, :3])

        # Use this to create the dataset for our objectness classifier
        # ========================================== #
        # original_indices = np.arange(pts_velo_cs_objects.shape[0])
        # segment_tree = TreeSegment(original_indices, 0)
        # segment_tree.child_segments = compute_hierarchical_tree(eps_list_tum, pts_velo_cs_objects, original_indices)
        # segment_tree_traverse(segment_tree, level = 0)
        # ========================================== #

        # flatten list(list(...(indices))) into list(indices)
        flat_indices = flatten_indices(indices)
        
        # map from object_indexes to pts_indexes
        mapped_indices = []
        for indexes in flat_indices:
            mapped_indices.append(pts_indexes_objects[indexes].tolist())

        flat_scores = flatten_scores(scores)

        new_instance = instances.max() + 1
        for id, indices in enumerate(mapped_indices):
            # if flat_scores[idx] < 0.1:
            #     instances[indices] = 0
            # else:
            instances[indices] = new_instance + id
            # majority semantic label in the segment is the new assignment
            labels[indices] = np.bincount(labels[indices]).argmax()

            # softmax based label transfer
            # things_softmax = softmax_scores[indices][:, :8]
            # labels[indices] = np.mean(things_softmax, axis = 0).argmax() + 1
        
        
        # Create .label files using the updated instance and semantic labels
        sem_labels = labels.astype(np.int32)
        inv_sem_labels = inv_learning_map[sem_labels]
        instances = np.left_shift(instances.astype(np.int32), 16)
        new_preds = np.bitwise_or(instances, inv_sem_labels)
        new_preds.tofile('{}/sequences/{:02d}/predictions/{:07d}.label'.format(
                args.save_dir, args.sequence, idx))
