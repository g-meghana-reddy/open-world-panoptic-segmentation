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
    dataset = 'kitti-raw' # 'semantic-kitti
    task_set = 0

    if task_set == 0:
        unk_label = 7
    elif task_set == 1:
        unk_label = 10
    else:
        assert False, 'Task set not implemented'
    
    print("*" * 80)
    print("Segmenting instances on task set: ", task_set)


    if len(sys.argv) > 2:
        seq = sys.argv[1]

    if dataset == 'kitti-raw':
        seq = '2011_09_26'
        write_dir = '/project_data/ramanan/achakrav/hu-segmentation/kitti_raw_ts{}/'.format(task_set)

        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti-Raw/{}/'.format(seq)
        scan_files = sorted(glob.glob(scan_folder + '*/velodyne_points/data/*.bin'))

        objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/val_preds_raw_TS{}/val_preds/'.format(task_set)
        objsem_files = load_paths(objsem_folder)
    else:
        seq = '08'
        write_dir = '/project_data/ramanan/achakrav/hu-segmentation/semantic_kitti_ts{}/'.format(task_set)

        scan_folder = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/' + seq + '/velodyne/'
        scan_files = load_paths(scan_folder)

        objsem_folder = '/project_data/ramanan/achakrav/4D-PLS/val_preds_TS{}/val_preds/'.format(task_set)
        objsem_files = load_paths(objsem_folder)

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    sem_file_mask = []
    obj_file_mask = []
    for idx, file in enumerate(objsem_files):
        if '_c' in file:
            obj_file_mask.append(idx)
        elif '_i' not in file and '_e' not in file and '_u' not in file and '_pots' not in file:
            sem_file_mask.append(idx)

    objectness_files = objsem_files[obj_file_mask]
    semantic_files = objsem_files[sem_file_mask]

    assert (len(semantic_files) == len(objectness_files))
    assert (len(semantic_files) == len(scan_files))

    for idx in tqdm(range(len(objectness_files))):
        # load scan
        scan_file = scan_files[idx]

        if dataset == 'kitti-raw':
            drive_dir = scan_file.split('/')[-4]
            drive_seq = seq + '/' + drive_dir + '/'
            if not os.path.exists(os.path.join(write_dir, drive_seq)):
                os.makedirs(os.path.join(write_dir, drive_seq))

        pts_velo_cs = load_vertex(scan_file)
        pts_indexes = np.arange(len(pts_velo_cs))

        # load objectness
        objectness_file = objectness_files[idx]
        objectness = np.load(objectness_file)

        # labels
        label_file = semantic_files[idx]
        labels = np.load(label_file)

        # # filter out background, this is used for raw labels
        # background = [
        #     0,   # "unlabeled", and others ignored
        #     1,   # "outlier"
        #     40,  # "road"
        #     44,  # "parking"
        #     48,  # "sidewalk"
        #     49,  # "other-ground"
        #     50,  # "building"
        #     51,  # "fence"
        #     70,  # "vegetation"
        #     71,  # "trunk"
        #     72,  # "terrain"
        # ]
        #
        # mask = []
        # background_mask = []
        # for idx, label in enumerate(labels):
        #     if label not in background:
        #         mask.append(idx)
        #     else:
        #         background_mask.append(idx)

        mask = labels == unk_label
        background_mask = labels != unk_label

        pts_velo_cs_objects = pts_velo_cs[mask]
        objectness_objects = objectness[mask]  # todo: change objectness_objects into a local variable
        pts_indexes_objects = pts_indexes[mask]

        # debug = o3d.geometry.PointCloud()
        # debug.points = o3d.utility.Vector3dVector(pts_velo_cs_objects[:, :3])
        # o3d.visualization.draw_geometries([debug])

        assert (len(pts_velo_cs_objects) == len(objectness_objects))

        if len(pts_velo_cs_objects) < 1:
            if dataset == 'kitti-raw':
                np.savez_compressed(os.path.join(write_dir, drive_seq + '_' + str(idx).zfill(6)),
                                    instances=[], segment_scores=[])
            else:
                np.savez_compressed(os.path.join(write_dir, seq + '_' + str(idx).zfill(6)),
                                    instances=[], segment_scores=[])
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

        # pdb.set_trace()
        # visualizer
        # vis_instance_o3d()

        # save results
        if dataset == 'kitti-raw':
            np.savez_compressed(os.path.join(write_dir, drive_seq + '_' + str(idx).zfill(6)),
                                instances=mapped_indices, segment_scores=flat_scores, allow_pickle = True)
        else:
            np.savez_compressed(os.path.join(write_dir, seq + '_' + str(idx).zfill(6)),
                                instances=mapped_indices, segment_scores=flat_scores, allow_pickle = True)
