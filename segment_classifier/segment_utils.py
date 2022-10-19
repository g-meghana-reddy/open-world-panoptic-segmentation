import json
import numpy as np
from sklearn.cluster import DBSCAN


class Segment():
    '''Segment node of the tree.'''
    def __init__(self, indices, score):
        self.indices = indices
        self.score = score

class TreeSegment():
    '''Tree of segments as nodes.'''
    def __init__(self, indices, score):
        self.child_segments = None
        self.curr_segment_data = Segment(indices, score)

def load_vertex(file):
    '''Load the vertices from the velodyne files.'''
    frame_points = np.fromfile(file, dtype=np.float32)
    frame_points = frame_points.reshape((-1, 4))
    return frame_points[:,:3]

def evaluate_iou_based_objectness(pred_inds, pts_indexes_objects, gt_instance_indexes, gt_instance_ids):
    '''Scoring based on IoU with the GT instances.'''
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

def compute_hierarchical_tree(eps_list, points_3d, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, original_indices= None):
    '''We compute segments from hierarchical tree and compute 
        the objectness scores to perform tree cut.'''
    
    # If we are at the end of the hierarchical tree then return the empty node.
    if len(eps_list) == 0: return []

    # Pick the first threshold from the eps_list.
    max_eps = eps_list[0]

    # Compute the original indices for each point at level 0
    # Reassign the original_indices in further levels to keep a track of them.
    if isinstance(original_indices, list): original_indices = np.array(original_indices)

    # Perform DBSCAN on the current set of 3D points to get segments 
    # at the current level L
    dbscan = DBSCAN(max_eps, min_samples=1).fit(points_3d[original_indices,:])
    labels = dbscan.labels_

    # Compute the indices and score per segment and store them as part of the tree node.
    segments = []
    for unique_label in np.unique(labels):

        inds = original_indices[np.flatnonzero(labels == unique_label)]
        score = evaluate_iou_based_objectness(inds, pts_indexes_objects, gt_instance_indexes, gt_instance_ids)
        segment = TreeSegment(inds, score)
        segment.child_segments = compute_hierarchical_tree(eps_list[1:], points_3d, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, inds)
        segments.append(segment)

    return segments


def segment_tree_traverse(segment_tree, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, filepath, segment_index, visited_indices):
    '''Traversal of the hierarchical tree.'''
    if len(segment_tree.curr_segment_data.indices) == 0:
        return segment_index
    
    if len(segment_tree.curr_segment_data.indices) > 50:
        if segment_tree.curr_segment_data.score < 0.2 or segment_tree.curr_segment_data.score >= 0.7 :
            
            if tuple(segment_tree.curr_segment_data.indices.tolist()) not in visited_indices:
                # print("----------------------------------------------")
                # print("Segment indices: ", segment_tree.curr_segment_data.indices)
                # print("Segment score: ", segment_tree.curr_segment_data.score)
                # print("----------------------------------------------")
                if segment_tree.curr_segment_data.score < 0.2:
                    gt_label = 0
                else:
                    gt_label = 1

                filename = filepath + '_' + '{:07d}.json'.format(segment_index)

                segment_data = {}
                segment_data['name'] = filename.split('/')[-1][:-5]
                segment_data['indices'] = segment_tree.curr_segment_data.indices.tolist()
                segment_data['objectness_score'] = segment_tree.curr_segment_data.score
                segment_data['xyz_features'] = pts_velo_cs_objects[segment_tree.curr_segment_data.indices].tolist()
                segment_data['semantic_features'] = pts_embeddings_objects[segment_tree.curr_segment_data.indices].tolist()
                segment_data['gt_label'] = gt_label
                segment_data['semantic_label'] = int(np.bincount(gt_semantic_labels[segment_tree.curr_segment_data.indices].astype(int)).argmax())

                with open(filename, 'w') as outfile:
                    json.dump(segment_data, outfile)

                segment_index += 1
                visited_indices.add(tuple(segment_data['indices']))

    for segment in segment_tree.child_segments:  
        segment_index = segment_tree_traverse(segment, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, filepath, segment_index, visited_indices)

    return segment_index