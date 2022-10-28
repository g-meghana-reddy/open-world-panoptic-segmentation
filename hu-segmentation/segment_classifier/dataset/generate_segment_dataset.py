import argparse
import sys
import os
import yaml

sys.path.append('../../..')

import glob
import numpy as np
import torch
from tqdm import tqdm

from segment_utils import *

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

DATA_DIR = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/"
OUTPUT_PATH = "/project_data/ramanan/achakrav/4D-PLS/data/segment_dataset/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=-1)
    parser.add_argument("-s", "--sequence", help="Sequence", type=int, default=8)
    parser.add_argument("-o", "--objsem_folder", help="Folder with object and semantic predictions", type=str, default="/project_data/ramanan/mganesin/4D-PLS/test/4DPLS_original_params_original_repo_nframes1_1e-3_softmax/val_probs")
    args = parser.parse_args()
    return args
    

def generate_segments_per_scan(scan_file, frame_emb_preds, frame_pred_labels, frame_gt_labels, frame_gt_ins_labels, filepath, frame_xyz=None):
    '''Constructs the hierarchical tree using the thing class points and per 
        node/segment score is computed to generate the segment dataset.'''
    #****************************************************
    # Load velodyne points
    #****************************************************
    pts_velo_cs = load_vertex(scan_file)
    pts_indexes = np.arange(len(pts_velo_cs))

    #********************************************************************
    # Compute the things mask and compute labels, xyz points accordingly
    #********************************************************************
    # thing classes: [1,2,3,4,5,6,7,8]
    gt_things_mask = np.where(np.logical_and(frame_gt_labels > 0 , frame_gt_labels < 9))
    things_mask = np.where(np.logical_and(frame_pred_labels > 0 , frame_pred_labels < 9))
    gt_mask = np.where(np.logical_and(frame_gt_labels > 0 , frame_gt_labels < 9))


    # generate all labels for things only
    if frame_xyz is not None:
        first_frame_coordinates = frame_xyz[things_mask]
    else:
        first_frame_coordinates = None
    pts_velo_cs_objects = pts_velo_cs[things_mask]
    pts_indexes_objects = pts_indexes[things_mask]
    pts_embeddings_objects = frame_emb_preds[things_mask]
    gt_instance_ids_objects = frame_gt_ins_labels[gt_things_mask]
    gt_instance_indexes_objects = np.arange(frame_gt_ins_labels.shape[0])[gt_things_mask]
    gt_semantic_labels = frame_gt_labels[gt_things_mask]

    gt_file = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/08/labels/000000.label'
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

    gt_label = np.fromfile(gt_file, dtype=np.int32)
    sem_gt = learning_map[gt_label & 0xFFFF]
    ins_gt = gt_label >> 16
    gt_mask = np.where(np.logical_and(sem_gt > 0 , sem_gt < 9))
    gt_instance_ids = ins_gt[gt_mask]
    gt_instance_indexes = np.arange(ins_gt.shape[0])[gt_mask]
    print(gt_mask[0].shape)


    gt_instance_ids_objects = ins_gt[gt_mask]
    gt_instance_indexes_objects = np.arange(ins_gt.shape[0])[gt_mask]

    if len(pts_velo_cs_objects) < 1:
        return
    
    # Define the hierarchical DBSCAN thresholds
    eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]

    #********************************************************************
    # Initialize the TreeSegment class for the current scan
    #********************************************************************
    init_sem_label = np.bincount(gt_semantic_labels).argmax()
    original_indices = np.arange(pts_velo_cs_objects.shape[0])
    segment_tree = TreeSegment(original_indices, 0, init_sem_label)
    
    #********************************************************************
    # Compute the hierarchical tree of segments
    #********************************************************************
    segment_tree.child_segments = compute_hierarchical_tree(
        eps_list_tum, pts_velo_cs_objects, pts_indexes_objects, gt_instance_indexes_objects,
        gt_instance_ids_objects, gt_semantic_labels, original_indices)

    #********************************************************************
    # Traverse the computed hierarchical tree to store the segments
    #********************************************************************
    segment_tree_traverse(
        segment_tree, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, 
        filepath, 0, set(), first_frame_coordinates
    )
    return

if __name__ == '__main__':
    args = parse_args()

    seq = "{:02d}".format(args.sequence)
    scan_folder = os.path.join(DATA_DIR, "sequences", seq, "velodyne")
    gt_label_folder = os.path.join(DATA_DIR, "sequences", seq, "labels")
    scan_files = load_paths(scan_folder)
    gt_files = np.array(sorted(glob.glob(gt_label_folder + "/*.label")))

    if args.task_set == -1:
        config = os.path.join(DATA_DIR, "semantic-kitti-orig.yaml")
    else:
        config = os.path.join(DATA_DIR, "semantic-kitti.yaml")

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
    assert (len(scan_files) == len(gt_files))

    for idx in tqdm(range(len(objectness_files))):
        # load scan
        scan_file = scan_files[idx]

        # predicted semantic labels
        pred_label_file = semantic_files[idx]
        pred_sem_labels = np.load(pred_label_file)

        # load embeddings
        embedding_file = embedding_files[idx]
        embedding = np.load(embedding_file)

        # GT labels
        gt_label_file = gt_files[idx]
        gt_labels = np.fromfile(gt_label_file, dtype=np.int32)
        gt_sem_labels = learning_map[gt_labels & 0xFFFF]
        gt_ins_labels = gt_labels >> 16

        output_dir = os.path.join(OUTPUT_PATH, "{:02d}".format(args.sequence))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, "{:07d}".format(idx))

        generate_segments_per_scan(
            scan_file, embedding, pred_sem_labels, gt_sem_labels,
            gt_ins_labels, filepath, None
        )
