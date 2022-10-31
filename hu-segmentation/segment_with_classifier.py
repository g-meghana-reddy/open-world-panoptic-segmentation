import argparse
import glob
import os
import sys
import yaml

sys.path.append("segment_classifier/")

import numpy as np
from sklearn.cluster import DBSCAN
import torch
from tqdm import tqdm

from segment_classifier.model.pointnet2 import PointNet2Classification
from tree_utils import flatten_indices, flatten_labels, flatten_scores
from utils import *


NUM_POINTS = 1024

class Config:
    # Model config
    USE_SEG_CLASSIFIER = True
    USE_SEM_FEATURES = False
    USE_SEM_REFINEMENT = False
    USE_SEM_WEIGHTS = False
    NUM_THINGS = 8

    FG_THRESH = 0.5
    N_POINTS = 1024


# Use the segment classifier to assign each segment a score
def evaluate(model, points, features=None):
    num_points_in_segment = points.shape[0]

    # if num_points_in_segment < 5:
    #     sem = None
    #     if model.config.USE_SEM_REFINEMENT:
    #         sem = 0
    #     return 0., sem

    # TODO: what to do about n_points?
    if num_points_in_segment > NUM_POINTS:
        chosen_idxs = np.random.choice(np.arange(num_points_in_segment), NUM_POINTS, replace=False)
    else:
        chosen_idxs = np.arange(0, num_points_in_segment, dtype=np.int32)
        if num_points_in_segment < NUM_POINTS:
            extra_idxs = np.random.choice(chosen_idxs, NUM_POINTS - len(chosen_idxs), replace=True)
            chosen_idxs = np.concatenate([chosen_idxs, extra_idxs], axis=0)
    np.random.shuffle(chosen_idxs)
    points = torch.from_numpy(points[chosen_idxs]).cuda().float()

    _mean = torch.mean(points, axis=0)
    _theta = torch.atan2(_mean[1], _mean[0]).cpu().numpy()
    _rot = torch.from_numpy(np.array([[ np.cos(_theta), np.sin(_theta)],
                        [-np.sin(_theta), np.cos(_theta)]])).cuda().float()
    
    points[:,:2] = torch.matmul(_rot, (points[:,:2] - _mean[None,:2]).T).T

    if features is not None:
        features = torch.from_numpy(features[chosen_idxs]).cuda().float()
        points = torch.cat([points, features], dim=-1)

    # get segment score using classifier
    with torch.no_grad():
        cls_, sem = model(points[None])
        score = cls_.sigmoid().cpu().numpy().item()
        if sem is not None:
            sem = sem.softmax(-1).argmax().cpu().numpy().item() + 1
    return score, sem

    
def segment(model, id_, eps_list, cloud, features=None, original_indices=None, aggr_func='min'):
    if not all(eps_list[i] > eps_list[i+1] for i in range(len(eps_list)-1)):
        raise ValueError('eps_list is not sorted in descending order')

    # pick the first threshold from the list
    max_eps = eps_list[0]
    if original_indices is None: 
        original_indices = np.arange(cloud.shape[0])
    if isinstance(original_indices, list): 
        original_indices = np.array(original_indices)

    # spatial segmentation
    dbscan = DBSCAN(max_eps, min_samples=1).fit(cloud[original_indices,:])
    labels = dbscan.labels_

    # evaluate every segment
    indices, scores, sem_labels = [], [], []
    for unique_label in np.unique(labels):
        inds = original_indices[np.flatnonzero(labels == unique_label)]
        indices.append(inds.tolist())
        segment_features = features[inds] if features is not None else None
        score, sem_label = evaluate(model, cloud[inds], segment_features)
        scores.append(score)
        sem_labels.append(sem_label)

    # return if we are done
    if len(eps_list) == 1:
        return indices, scores, sem_labels

    # expand recursively
    final_indices, final_scores, final_sem_labels = [], [], []
    for i, (inds, score, sem_label) in enumerate(zip(indices, scores, sem_labels)):
        # focus on this segment
        fine_indices, fine_scores, fine_sem_labels = segment(
            model, id_, eps_list[1:], cloud, features=features, original_indices=inds)
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
            final_sem_labels.append(fine_sem_labels)
        else: # otherwise
            final_indices.append(inds)
            final_scores.append(score)
            final_sem_labels.append(sem_label)
    return final_indices, final_scores, final_sem_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=-1)
    parser.add_argument("-d", "--dataset", help="Dataset", default='semantic-kitti')
    parser.add_argument("-s", "--sequence", help="Sequence", type=int, default=8)
    parser.add_argument("-o", "--objsem_folder", help="Folder with object and semantic predictions", type=str, default="/project_data/ramanan/mganesin/4D-PLS/test/4DPLS_original_params_original_repo_nframes1_1e-3_softmax/val_probs")
    parser.add_argument("-sd", "--save_dir", help="Save directory", type=str, default='test/LOSP')
    parser.add_argument("--ckpt", help="Checkpoint to load for segment classifier", type=str, default=None)
    parser.add_argument("--use-sem-features", help="Whether to use semantic features in classifier", action="store_true")
    parser.add_argument("--use-sem-refinement", help="Whether to use semantic refinement head", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.task_set == 0:
        unk_labels = [7]
    elif args.task_set == 1:
        unk_labels = [10]
    elif args.task_set == -1:
        unk_labels = range(1,9)
    else:
        raise ValueError('Unknown task set: {}'.format(args.task_set))

    if args.use_sem_refinement:
        in_channels = 256 # 0
        # args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/xyz_mean_focal_sem_feats_sem_refinement_weighted/checkpoints/epoch_200.pth"
        # args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/sem_feats_only_focal_sem_refinement_weighted_project_input_1024/checkpoints/epoch_200.pth"
        args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/xyz_mean_focal_sem_feats_sem_reifnement_weighted_new_indices/checkpoints/epoch_200.pth"
        # args.ckpt = "/project_data/ramanan/mganesin/4D-PLS/results/checkpoints/xyz_sem_mean_refine/checkpoints/epoch_200.pth"
    elif args.use_sem_features:
        in_channels = 256
        # args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/sem_xyz/checkpoints/epoch_200.pth"
        args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/xyz_mean_focal_sem_feats/checkpoints/epoch_50.pth"
    else:
        in_channels = 0
        args.ckpt = "/project_data/ramanan/achakrav/4D-PLS/results/checkpoints/xyz_mean_focal/checkpoints/epoch_200.pth"

    # instantiate the segment classifier
    cfg = Config()
    cfg.USE_SEM_FEATURES = args.use_sem_features
    cfg.USE_SEM_REFINEMENT = args.use_sem_refinement

    print("Loading segment classifier from checkpoint")
    classifier = PointNet2Classification(cfg, in_channels).cuda()
    ckpt = torch.load(args.ckpt)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()

    if args.dataset == 'semantic-kitti':
        seq = '{:02d}'.format(args.sequence)
        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        # scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne_first_frame/'
        scan_files = load_paths(scan_folder)

        if args.task_set == -1:
            config = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti-orig.yaml"
        else:
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
        
    objsem_folder = args.objsem_folder
    objsem_files = load_paths(objsem_folder)

    sem_file_mask = []
    obj_file_mask = []
    ins_file_mask = []
    emb_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c.' in file:
            obj_file_mask.append(idx)
        elif '_i.' in file:
            # Added for nframes=2
            frame_name = file.split('/')[-1]
            if len(frame_name.split('_')) < 4:
                ins_file_mask.append(idx)
        elif '_e' in file:
            frame_name = file.split('/')[-1]
            if len(frame_name.split('_')) < 4:
                emb_file_mask.append(idx)
        elif '_u.' not in file and '_t.' not in file and '_pots.' not in file and '.ply' not in file and '_s.' not in file:
            # Added for nframes=2
            frame_name = file.split('/')[-1]
            if len(frame_name.split('_')) < 3:
                sem_file_mask.append(idx)
    
    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]
    instance_files = objsem_files[ins_file_mask]
    embedding_files = objsem_files[emb_file_mask]

    assert (len(semantic_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    for idx in tqdm(range(len(objectness_files))):
        segmented_dir = "{}/sequences/{:02d}/predictions/".format(
            args.save_dir, args.sequence)
        if not os.path.exists(segmented_dir):
            os.makedirs(segmented_dir)

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

        # semantic features / embeddings
        if args.use_sem_features:
            embedding_file = embedding_files[idx]
            semantic_features = np.load(embedding_file)
        else:
            semantic_features = None

        if len(unk_labels) == 1:
            mask = np.where(labels == unk_labels[0])
        else:
            mask = np.where(np.logical_and(labels > 0, labels < np.max(unk_labels)))

        pts_velo_cs_objects = pts_velo_cs[mask]
        objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
        pts_indexes_objects = pts_indexes[mask]

        assert (len(pts_velo_cs_objects) == len(objectness_objects))

        if len(pts_velo_cs_objects) < 1:
            assert False

        # mask out 4dpls instance predictions
        instances[mask] = 0

        # segmentation with point-net
        id_ = 0
        # eps_list = [2.0, 1.0, 0.5, 0.25]
        eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]
        indices, scores, sem_labels = segment(
            classifier, id_, eps_list_tum,
            pts_velo_cs_objects[:, :3], features=semantic_features
        )

        # flatten list(list(...(indices))) into list(indices)
        flat_indices = flatten_indices(indices)
        # map from object_indexes to pts_indexes
        mapped_indices = []
        for indexes in flat_indices:
            mapped_indices.append(pts_indexes_objects[indexes].tolist())

        # mapped_flat_indices = pts_indexes_objects
        flat_scores = flatten_scores(scores)
        if args.use_sem_refinement:
            flat_sem_labels = flatten_labels(sem_labels)

        new_instance = instances.max() + 1
        for id_, indices in enumerate(mapped_indices):
            instances[indices] = new_instance + id_
            if args.use_sem_refinement:
                labels[indices] = flat_sem_labels[id_]

        # Create .label files using the updated instance and semantic labels
        sem_labels = labels.astype(np.int32)
        inv_sem_labels = inv_learning_map[sem_labels]
        instances = np.left_shift(instances.astype(np.int32), 16)
        new_preds = np.bitwise_or(instances, inv_sem_labels)
        new_preds.tofile('{}/{:07d}.label'.format(segmented_dir, idx))
