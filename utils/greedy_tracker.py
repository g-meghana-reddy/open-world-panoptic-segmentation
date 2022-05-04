import math
import numpy as np


class GreedyTracker():
    '''
        A center point-based tracker. Given points from two frames, 
        the instances are matched based on L2 distance of centroid points.
        Note: We are not shifting the prev frame points into curr camera coordinate.
        Assumption: No max life parameter support (any unmatched track is deleted).
    '''
    def __init__(self, threshold=10.):
        self.trackers = {}
        self.threshold = threshold

    def match(self, cost_matrix):
        r_inds, c_inds = [], []
        rows, cols = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        for (r, c) in zip(rows, cols):
            if cost_matrix[r, c] >= self.threshold:
                break
            if r not in r_inds and c not in c_inds:
                r_inds.append(r)
                c_inds.append(c)
        
        return r_inds, c_inds

    def update(self, centers):
        # associate new centers with previous centers
        if self.trackers:
            
            tracks = list(self.trackers.values())
            tracks = np.vstack(tracks)
            dist_matrix = np.linalg.norm(centers[:, None] - tracks[None], axis=-1)
            row_ind, col_ind = self.match(dist_matrix)
            # row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched_indices = np.stack([row_ind, col_ind], axis=1)
            
            unmatched_detections = []
            for d, det in enumerate(centers):
                if d not in matched_indices[:, 0]: 
                    unmatched_detections.append(d)
            unmatched_trackers = []
            for t in self.trackers:
                if t not in matched_indices[:, 1]: 
                    unmatched_trackers.append(t)
            
            # filter out matches with high cost
            # matches = []
            # for m in matched_indices:
            #     if dist_matrix[m[0], m[1]] >= self.threshold:
            #         unmatched_detections.append(m[0])
            #         unmatched_trackers.append(m[1])
            #     else:
            #         matches.append(m)
            matches = matched_indices

            # update matched trackers with assigned detections
            for (d, t) in matches:
                if t not in unmatched_trackers:
                    self.trackers[t] = centers[d]

            # create and initialise new trackers for unmatched detections
            track_keys = self.trackers.keys()
            missing_keys = list(set(range(max(track_keys) + 1)) - set(track_keys))
            max_key = max(track_keys)
            k = len(unmatched_detections) - len(missing_keys)
            missing_keys.extend(range(max_key + 1, max_key + 1 + k ))

#             idx = max(self.trackers.keys()) 
#             for i in unmatched_detections :        # a scalar of index
#                 idx += 1
#                 self.trackers[idx] = centers[i]

            missing_keys = []
            for key in self.trackers:
                if self.trackers[key][0] == math.inf:
                    missing_keys.append(key)
                    
            k = len(unmatched_detections) - len(missing_keys)
            missing_keys.extend(range(max_key + 1, max_key + 1 + k ))
            
            # remove dead tracklets
            for i in sorted(unmatched_trackers, reverse=True):
                self.trackers[i] = np.array([math.inf, math.inf, math.inf])
             
            for idx, i in zip(missing_keys, unmatched_detections):
                self.trackers[idx] = centers[i]
                
        else:
            for i in range(len(centers)):
                self.trackers[i] = centers[i]
            #self.trackers.extend(centers)
        
        #self.trackers = collections.OrderedDict(sorted(self.trackers.items()))
        return self.trackers