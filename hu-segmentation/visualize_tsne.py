from ossaudiodev import SNDCTL_SEQ_GETTIME
from sklearn.cluster import DBSCAN
from sklearn import manifold
import numpy as np
import pickle
import pdb
import time
import os
from tree_utils import flatten_scores, flatten_indices
import sys
from utils import *
import open3d as o3d

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


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


if __name__ == '__main__':
    seq = '08'
    # seq_prefix = '08_0'

    if len(sys.argv) > 2:
        seq = sys.argv[1]
        # seq_prefix = sys.argv[2]
 
    # write_dir = '/project_data/ramanan/achakrav/hu-segmentation/kitti_raw_ts1_segmented/'
    write_dir = '/project_data/ramanan/achakrav/hu-segmentation/semantic_kitti_ts1/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    # scan_folder = '/media/data/dataset/kitti-odometry/dataset/sequences/' + seq + '/velodyne'
    scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
    scan_files = load_paths(scan_folder)

    # objectness_folder = '/media/data/tmp/testsetobj'
    # objectness_files_raw = load_paths(objectness_folder)
    # objectness_files = [path for path in objectness_files_raw if seq_prefix in path]

    # semantic_folder = '/media/data/tmp/testsetsem'
    # semantic_files_raw = load_paths(semantic_folder)
    # semantic_files = [path for path in semantic_files_raw if seq_prefix in path]

    objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/val_preds_TS1/val_preds/'
    objsem_files = load_paths(objsem_folder)

    sem_file_mask = []
    obj_file_mask = []
    emb_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c' in file:
            obj_file_mask.append(idx)
        elif '_e' in file:
            emb_file_mask.append(idx)
        elif '_i' not in file and '_pots' not in file:
            sem_file_mask.append(idx)

    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]
    embedding_files = objsem_files[emb_file_mask]

    avg_embeddings = []
    
    assert (len(semantic_files) == len(objectness_files))
    assert (len(embedding_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    config_file = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti.yaml'
    with open(config_file, 'r') as stream:
        doc = yaml.safe_load(stream)
        learning_map = doc['task_set_map'][2]['learning_map']
        learning_map_arr = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            learning_map_arr[k] = v

    label_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/labels/'
    label_files = load_paths(label_folder)

    label_file_mask = []
    for idx, file in enumerate(label_files):
        if '.label' in file:
            label_file_mask.append(idx)
    
    label_files = label_files[label_file_mask]
    gt_labels = []

    assert (len(label_files) == len(scan_files))

    for idx in tqdm(range(len(objectness_files))):
        # load scan
        # frame_idx = int(os.path.basename(semantic_files[idx]).replace('.npy', '').replace('08_', ''))
        # scan_file = scan_files[frame_idx]
        scan_file = scan_files[idx]
        pts_velo_cs = load_vertex(scan_file)
        pts_indexes = np.arange(len(pts_velo_cs))

        # load objectness
        objectness_file = objectness_files[idx]
        objectness = np.load(objectness_file)

        # labels
        label_file = semantic_files[idx]
        labels = np.load(label_file)

        # add gt labels for T-SNE
        gt_file = label_files[idx]
        gt_label = np.fromfile(gt_file, dtype=np.int32)
        gt_sem_label = gt_label & 0xFFFF
        gt_sem_label = learning_map_arr[gt_sem_label]
        gt_ins_labels = gt_label >> 16
        gt_ins_labels = gt_ins_labels.astype(np.int32)

        # embeddings
        embedding_file = embedding_files[idx]
        embeddings = np.load(embedding_file)

        unk_label = 10 # for task set 1
        mask = labels == unk_label
        bg_mask = labels != unk_label

        pts_velo_cs_objects = pts_velo_cs[mask]
        objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
        pts_indexes_objects = pts_indexes[mask]
        embeddings_objects = embeddings[mask]
        gt_sem_label_objects = gt_sem_label[mask]

        assert (len(pts_velo_cs_objects) == len(embeddings_objects))

        if len(pts_velo_cs_objects) < 1:
            continue
        
        # segmentation with point-net
        id_ = 0
        # eps_list = [2.0, 1.0, 0.5, 0.25]
        eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]
        indices, scores = segment(id_, eps_list_tum, pts_velo_cs_objects[:, :3])

        # flatten list(list(...(indices))) into list(indices)
        flat_indices = flatten_indices(indices)
        # map from object_indexes to pts_indexes
        # mapped_indices = []
        for indexes in flat_indices:
            # mapped_indices.append(pts_indexes_objects[indexes].tolist())
            x1, y1, z1, _ = pts_velo_cs_objects[indexes].min(axis=0)
            x2, y2, z2, _ = pts_velo_cs_objects[indexes].max(axis=0)
            spatial_dim = np.array([abs(x2-x1), abs(y2-y1), abs(z2-z1)])
            avg_embedding = embeddings_objects[indexes].mean(axis=0)
            embedding = np.concatenate([avg_embedding, spatial_dim])
            avg_embeddings.append(embedding)

            segment_label = np.bincount(gt_sem_label_objects[indexes]).argmax()
            gt_labels.append(segment_label)

        # known segments
        pts_velo_cs_known = pts_velo_cs[bg_mask]
        embeddings_known = embeddings[bg_mask]
        gt_sem_label_known = gt_sem_label[bg_mask]
        gt_ins_label_known = gt_ins_labels[bg_mask]

        if len(embeddings_known) < 1:
            continue
        
        for ins in np.unique(gt_ins_label_known):
            if ins == 0:
                continue

            inds = np.argwhere((gt_ins_label_known == ins))[:, 0]
            x1, y1, z1, _ = pts_velo_cs_known[inds].min(axis=0)
            x2, y2, z2, _ = pts_velo_cs_known[inds].max(axis=0)
            spatial_dim = np.array([abs(x2-x1), abs(y2-y1), abs(z2-z1)])
            avg_embedding = embeddings_known[inds].mean(axis=0)
            embedding = np.concatenate([avg_embedding, spatial_dim])
            avg_embeddings.append(embedding)

            segment_label = np.bincount(gt_sem_label_known[inds]).argmax()
            gt_labels.append(segment_label)

    assert (len(avg_embeddings) == len(gt_labels))

    # for task set 1
    class_strings = [
        'unlabelled', 'car-KNOWN', 'bicycle-UNKNOWN', 'motorcycle-UNKNOWN', 'truck-KNOWN', 
        'other-vehicle-UNKNOWN', 'person-KNOWN', 'road-KNOWN', 'parking-UNKNOWN', 
        'sidewalk-KNOWN', 'other-ground-UNKNOWN', 'building-KNOWN', 'fence-KNOWN',
        'vegetation-KNOWN', 'trunk-UNKNOWN', 'terrain-KNOWN', 'pole-KNOWN', 'traffic-sign-UNKNOWN'
    ]
    hue_order = [
        'unlabelled', 'car-KNOWN', 'truck-KNOWN', 'person-KNOWN', 'road-KNOWN', 'sidewalk-KNOWN',
        'building-KNOWN', 'fence-KNOWN', 'vegetation-KNOWN', 'terrain-KNOWN', 'pole-KNOWN',
        'bicycle-UNKNOWN', 'motorcycle-UNKNOWN',  'other-vehicle-UNKNOWN', 'parking-UNKNOWN', 
        'other-ground-UNKNOWN', 'trunk-UNKNOWN', 'traffic-sign-UNKNOWN'
    ]

    method = manifold.TSNE(n_components=2, init="pca", random_state=0)
    Y = method.fit_transform(avg_embeddings)
    plt.figure(figsize=(16,10))
    df = pd.DataFrame({'X_tsne': Y[:, 0], 'Y_tsne': Y[:, 1]})
    df['label'] = [class_strings[i] for i in gt_labels] 
    sns.scatterplot(
        x='X_tsne', y='Y_tsne',
        hue='label',
        palette='tab20',
        hue_order=hue_order,
        data=df,
        alpha=0.3
    )
    plt.axis('off')
    plt.savefig('test_tsne.png')