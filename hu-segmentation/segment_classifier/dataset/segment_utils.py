import os
import numpy as np
from sklearn.cluster import DBSCAN


class Segment():
    '''Segment node of the tree.'''
    def __init__(self, indices, score, label=None):
        self.indices = indices
        self.score = score
        self.label = label


class TreeSegment():
    '''Tree of segments as nodes.'''
    def __init__(self, indices, score, label=None):
        self.child_segments = None
        self.curr_segment_data = Segment(indices, score, label)


def load_vertex(file):
    '''Load the vertices from the velodyne files.'''
    frame_points = np.fromfile(file, dtype=np.float32)
    frame_points = frame_points.reshape((-1, 4))
    return frame_points[:,:3]


def load_paths(folder):
  paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  paths.sort()
  return np.array(paths)


def evaluate_iou_based_objectness(pred_inds, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, gt_semantic_labels):
    '''Scoring based on IoU with the GT instances.'''
    # predictions
    pred_counts = pred_inds.shape[0]
    pred_indices = pts_indexes_objects[pred_inds]

    # groundtruth
    gt_indices_local = np.arange(gt_instance_indexes.shape[0])
    gt_inst_ids = gt_instance_ids[gt_indices_local]

    unique_gt_ids, unique_gt_counts = np.unique(gt_inst_ids, return_counts = True)

    ious, labels = [], []
    for idx, gt_ins in enumerate(unique_gt_ids):
        gt_ind = np.where(gt_inst_ids == gt_ins)
        label = np.bincount(gt_semantic_labels[gt_ind]).argmax()
        intersections = len(set(pred_indices) & set(gt_instance_indexes[gt_ind]))
        gt_counts = unique_gt_counts[idx]
        union = gt_counts + pred_counts - intersections
        iou = intersections / union
        ious.append(iou)
        labels.append(label)

    ious, labels = np.array(ious), np.array(labels)
    max_idx = np.argmax(ious)
    return ious[max_idx], labels[max_idx]


def compute_hierarchical_tree(eps_list, points_3d, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, gt_semantic_labels, original_indices= None):
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
        score, label = evaluate_iou_based_objectness(inds, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, gt_semantic_labels)
        segment = TreeSegment(inds, score, label)
        segment.child_segments = compute_hierarchical_tree(eps_list[1:], points_3d, pts_indexes_objects, gt_instance_indexes, gt_instance_ids, gt_semantic_labels, inds)
        segments.append(segment)

    return segments


def segment_tree_traverse(segment_tree, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, filepath, segment_index, visited_indices, first_frame_coordinates=None):
    '''Traversal of the hierarchical tree.'''
    if len(segment_tree.curr_segment_data.indices) == 0:
        return segment_index

    if len(segment_tree.curr_segment_data.indices) > 25:
        # if segment_tree.curr_segment_data.score < 0.3 or segment_tree.curr_segment_data.score >= 0.7:
        if segment_tree.curr_segment_data.score <= 1.0 or segment_tree.curr_segment_data.score >= 0.:

            if tuple(segment_tree.curr_segment_data.indices.tolist()) not in visited_indices:
                # print("----------------------------------------------")
                # print("Segment indices: ", segment_tree.curr_segment_data.indices)
                # print("Segment score: ", segment_tree.curr_segment_data.score)
                # print("----------------------------------------------")
                if segment_tree.curr_segment_data.score < 0.3:
                    gt_label = 0
                else:
                    gt_label = 1

                filename = filepath + '_' + '{:07d}.npz'.format(segment_index)
                name = filename.split('/')[-1][:-5]
                indices = segment_tree.curr_segment_data.indices
                objectness = segment_tree.curr_segment_data.score
                xyz = pts_velo_cs_objects[segment_tree.curr_segment_data.indices]
                semantic_features = pts_embeddings_objects[segment_tree.curr_segment_data.indices]
                semantic_label = int(segment_tree.curr_segment_data.label)
                if first_frame_coordinates is not None:
                    first_frame_xyz = first_frame_coordinates[segment_tree.curr_segment_data.indices]
                    np.savez(
                        filename,
                        name=name,
                        indices=indices,
                        objectness=objectness,
                        xyz=xyz,
                        first_frame_xyz=first_frame_xyz,
                        semantic_features=semantic_features,
                        gt_label=gt_label,
                        semantic_label=semantic_label,
                    )
                else:
                    np.savez(
                        filename,
                        name=name,
                        indices=indices,
                        objectness=objectness,
                        xyz=xyz,
                        semantic_features=semantic_features,
                        gt_label=gt_label,
                        semantic_label=semantic_label,
                    )

                segment_index += 1
                visited_indices.add(tuple(indices.tolist()))

    for segment in segment_tree.child_segments:  
        segment_index = segment_tree_traverse(segment, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, filepath, segment_index, visited_indices, first_frame_coordinates=first_frame_coordinates)

    return segment_index
