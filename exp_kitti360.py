import os
import glob
from plyfile import PlyData
import pdb
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

thresh = 1


'''
NOTES:
1. For static points, world coordinates are always same. So, we should try to match each point in accumulated point cloud to each point in raw point cloud.
2. In frame 0, there are ~100,000 raw points but only 26,000 unique neighbours to the accumulated point cloud. We probably need to set a threshold (< ~1). But many raw points will not be matched since accumulated point cloud is subsampled. What to do for the unmatched points, what labels to assign?
3. Maybe each raw point need not match uniquely to a point in accumulated point cloud. Using a distance threshold of 1, ~83000 points get matched. Is this the way to go?
'''


def loadCalibrationRigid(filename):
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    return np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))


'''
ANI: What is the difference between static and dynamic?
     Can we just concatenate the two? Or what exactly is the difference?
'''
kitti360_dir = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti360/'
semantic_kitti_dir = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/'

drive_dirs = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_semantics', 'train', '*')))
cam_to_world_all = sorted(glob.glob(os.path.join(kitti360_dir, 'data_poses', '*', 'cam0_to_world.txt')))

fileCameraToVelo = os.path.join(kitti360_dir, 'calibration', 'calib_cam_to_velo.txt')
TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
# TrVeloToCam0 = np.linalg.inv(TrCam0ToVelo)

# intrinsic_file = os.path.join(kitti360_dir, 'calibration', 'perspective.txt')
# R_rect = np.eye(4)
# with open(intrinsic_file) as f:
#     intrinsics = f.read().splitlines()
# for line in intrinsics:
#     line = line.split(' ')
#     if line[0] == 'R_rect_00:':
#         R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
# TrVeloToRect = np.matmul(R_rect, TrVeloToCam0)

for drive_dir in drive_dirs:
    drive = drive_dir.split('/')[-1]
    static_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_semantics', 'train', drive, 'static', '*')))
    dynamic_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_semantics', 'train', drive, 'dynamic', '*')))
    raw_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_raw', drive, 'velodyne_points', 'data', '*')))
    
    i = 0
    cam2world_drive = cam_to_world_all[drive in cam_to_world_all]
    cam2world = np.loadtxt(cam2world_drive)[i, 1:].reshape(4,4)
    world2cam = np.linalg.inv(cam2world)

    i += 1
    cam2world_next = np.loadtxt(cam2world_drive)[i]
    next_idx, cam2world_next = cam2world_next[0], cam2world_next[1:].reshape(4,4)
    world2cam_next = np.linalg.inv(cam2world_next)

    for (static_file, dynamic_file) in zip(static_files, dynamic_files):
        filename = static_file.split('/')[-1]

        # load points
        static_data = PlyData.read(static_file)
        dynamic_data = PlyData.read(dynamic_file)
        pdb.set_trace()

        # transform world points to velodyne coordinates
        static_points = np.stack([
            static_data['vertex']['x'], static_data['vertex']['y'], 
            static_data['vertex']['z'], np.ones(static_data['vertex'].count)
        ], axis=-1)
        static_points_cam = np.matmul(world2cam, static_points.T).T
        static_points_velo = np.matmul(TrCam0ToVelo, static_points_cam.T).T
        static_points_velo = static_points_velo[:, :3] / static_points_velo[:, None, 3]

        sem_labels = static_data['vertex']['semantic']
        ins_labels = static_data['vertex']['instance']

        nbrs = NearestNeighbors(n_neighbors=1).fit(static_points_velo)

#         if dynamic_data['vertex'].count:
#             dynamic_points = np.stack([
#                 dynamic_data['vertex']['x'], dynamic_data['vertex']['y'], 
#                 dynamic_data['vertex']['z'], np.ones(dynamic_data['vertex'].count)
#             ], axis=-1)

        start_frame, end_frame = filename.replace('.', '_').split('_')[:2]
        start_frame_id, end_frame_id = int(start_frame), int(end_frame)

        for frame in range(start_frame_id, end_frame_id):
            raw_points = np.fromfile(raw_files[frame], dtype=np.float32).reshape(-1, 4)
            raw_points = raw_points[:, :3]
            raw_labels = np.zeros(raw_points.shape[0], dtype=np.int32)

            # load next cam2world transformation
            if i == int(next_idx):
                cam2world, world2cam = cam2world_next, world2cam_next
                i += 1
                cam2world_next = np.loadtxt(cam2world_drive)[i]
                next_idx, cam2world_next = cam2world_next[0], cam2world_next[1:].reshape(4,4)
                world2cam_next = np.linalg.inv(cam2world_next)

                static_points_cam = np.matmul(world2cam, static_points.T).T
                static_points_velo = np.matmul(TrCam0ToVelo, static_points_cam.T).T

                nbrs = NearestNeighbors(n_neighbors=1).fit(static_points_velo)

            # match each raw point to it's closest neighbour within a threshold
            distances, indices = nbrs.kneighbors(raw_points)
            valid_inds = (distances < thresh)
            point_inds = indices[valid_inds]

            # assign the corresponding labels to that point
            raw_sem_labels = sem_labels[point_inds]
            raw_ins_labels = ins_labels[point_inds]
            new_preds = np.left_shift(raw_ins_labels, 16)
            raw_labels[valid_inds] = np.bitwise_or(new_preds, raw_sem_labels) 
            pdb.set_trace()

#             min_dist_arr = []
#             for i in range()
#             for raw_point in tqdm(raw_points):
#                 dist = np.linalg.norm(raw_point[None] - static_points_velo[:, None], axis=-1)
#                 print(dist.min())
#                 min_dist_arr.append(dist.min())
#             pdb.set_trace()

#             pointsCam = np.matmul(TrVeloToRect, raw_points.T).T
#             pointsCam = pointsCam[:, :3]
#             pdb.set_trace()



static_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_semantics', 'train', '*', 'static', '*')))
dynamic_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_semantics', 'train', '*', 'dynamic', '*')))
raw_files = sorted(glob.glob(os.path.join(kitti360_dir, 'data_3d_raw', '*', 'velodyne_points', 'data', '*')))


for (static_file, dynamic_file) in zip(static_files, dynamic_files):
    drive_dir = static_file.split('/')[-3]
    filename = static_file.split('/')[-1]
    
    start_frame, end_frame = filename.replace('.', '_').split('_')[:2]
    start_frame, end_frame = int(start_frame), int(end_frame)
    
    # load camera to world transforms
    # TODO: correct this
    cam2world_drive = cam_to_world_all[drive_dir in cam_to_world_all]
    cam2world = np.loadtxt(cam2world_drive)[0, 1:].reshape(4, 4)
    cam2world = cam2world_0

    i = 1
    cam2world_next = np.loadtxt(cam2world_drive)[i]
    next_idx, cam2world_next = cam2world_next[0], cam2world_next[1:].reshape(4,4)
    
    # load points
    static_data = PlyData.read(static_file)
    dynamic_data = PlyData.read(dynamic_file)

    for frame in range(start_frame, end_frame):
        raw_points = raw_files[frame]
        
        # load next cam2world transformation
        if i == int(next_idx):
            cam2world = cam2world_next
            i += 1
            cam2world_next = np.loadtxt(cam2world_drive)[i]
            next_idx, cam2world_next = cam2world_next[0], cam2world_next[1:].reshape(4,4)
        
        raw
        
        
        

    pdb.set_trace()
    

# for (static_file, dynamic_file, raw_file) in zip(static_files, dynamic_files, raw_files):
#     drive_dir = static_file.split('/')[-3]
    
#     cam2world = cam_to_world_all[drive_dir in cam_to_world_all]
#     cam2world = np.loadtxt(cam2world)
#     cam2world = cam2world[0, 1:].reshape(4, 4)
#     pdb.set_trace()

#     static_data = PlyData.read(static_file)
#     dynamic_data = PlyData.read(dynamic_file)
#     raw_data = np.fromfile(raw_file)

#     points_all = np.stack([
#         static_data['vertex']['x'], static_data['vertex']['y'], static_data['vertex']['z']
#     ], axis=-1)

#     cam_to_world = cam_to_world_all[drive_dir in cam_to_world_all]
#     cam_to_world = np.loadtxt(cam_to_world)
#     cam_to_world_tmp = cam_to_world[0, 1:].reshape(4, 4)
#     R_cam2world, T_cam2world = cam_to_world_tmp[:3, :3], cam_to_world_tmp[-1, :3]
#     # points_cam = world2cam(points, R_cam2world, T_cam2world, inverse=False)
#     points_cam = world2cam(points, R_cam2world, T_cam2world, inverse=True)
#     pdb.set_trace()
#     # points_cam = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
#     points_cam = np.concatenate([points_cam, np.ones((1, points_cam.shape[1]))], axis=0)

#     cam_to_velo = np.concatenate([cam_to_velo, np.zeros((1, 4))])
#     cam_to_velo[-1, -1] = 1.

#     # points_velo = cam_to_velo @ points_cam.T
#     points_velo = cam_to_velo @ points_cam
#     points_velo = points_velo[:3] / points_velo[-1, None]
#     points_velo = points_velo.T
#     pdb.set_trace()
#     # dynamic_data = PlyData.read(dynamic_file)
#     if dynamic_data['vertex']['x'].shape[0]:
#         pdb.set_trace()