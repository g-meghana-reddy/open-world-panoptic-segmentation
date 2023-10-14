#!/usr/bin/env python3
# Developed by Xieyuanli Chen on 30.10.19.
# Brief: some utilities

import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm

np.random.seed(0)


def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians

    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (
        depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # [print(x) for x in proj_x]
    # [print(y) for y in proj_y]
    # [print(d) for d in depth]

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_idx


def range_projection_vertex(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (
        depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 3), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T

    return proj_range, proj_vertex


def range_projection_o3d(pcd, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    current_points = np.asarray(pcd.points)

    # get depth of all points
    depth = np.linalg.norm(current_points, 2, axis=1)
    current_points = current_points[(depth > 0) & (
        depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_points[:, 0]
    scan_y = current_points[:, 1]
    scan_z = current_points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 3), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T

    return proj_range, proj_vertex


def wrap(x, dim):
    value = x
    if value >= dim:
        value = (value - dim)
    if value < 0:
        value = (value + dim)
    return value


def gen_normal_map(current_range, current_vertex, proj_H, proj_W):
    normal_map = np.full((proj_H, proj_W, 3), 0, dtype=np.float32)

    for x in range(proj_W):
        for y in range(proj_H - 1):
            p = current_vertex[y, x][:3]
            depth = current_range[y, x]

            if depth > 0:
                wrap_x = wrap(x + 1, proj_W)
                u = current_vertex[y, wrap_x][:3]
                u_depth = current_range[y, wrap_x]
                if u_depth <= 0:
                    continue

                v = current_vertex[y + 1, x][:3]
                v_depth = current_range[y + 1, x]
                if v_depth <= 0:
                    continue

                u_norm = (u - p) / np.linalg.norm(u - p)
                v_norm = (v - p) / np.linalg.norm(v - p)

                w = np.cross(v_norm, u_norm)
                norm = np.linalg.norm(w)
                if norm > 0:
                    normal = w / norm
                    normal_map[y, x] = normal

    return normal_map


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def euler_angles_from_rotation_matrix(R):
    '''
    From the paper by Gregory G. Slabaugh,
    Computing Euler angles from a rotation matrix
    psi, theta, phi = roll pitch yaw (x, y, z)
    '''
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return phi


def load_vertex(scan_path):
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    current_vertex = np.ones(
        (current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_points(scan_path):
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 3))
    current_points = current_vertex[:, 0:3]
    return current_points


def load_points_with_intensity(scan_path):
    current_points = np.fromfile(scan_path, dtype=np.float32)
    current_points = current_points.reshape((-1, 4))
    return current_points


def load_paths(folder):
    paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    paths.sort()
    return np.array(paths)


def gen_transformation(yaw, translation):
    """ Brief: generate transformation from given yaw angle and translation
        Input: current_range: range image
              current_vertex: point clouds
        Output: normal image
    """
    rotation = R.from_euler('zyx', [[yaw, 0, 0]], degrees=True)
    rotation = rotation.as_dcm()[0]
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = [translation[0], translation[1], translation[2]]

    return transformation


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def filter_far_points(points):
    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)
    # filter out points which are invalid or farther than 50 meters
    return points[(depth > 0) & (depth < 50)]


def test_ipbcar_depth_normal():
    # test normal function
    # grid = '/home/xieyuanlichen/my_scripts/overlap_localization/grid_virtual_frames/grid_0_0.npz'
    # visible_points = np.load(grid)['arr_0']

    scan_folder = '/media/xieyuanlichen/71c5fd91-41fe-45c8-b0f1-a62b56513f8a/dataset/' \
                  'ipb_car/first_recording/ipb_car_lidar_scans'
    scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_folder)) for f in fn]
    scan_paths.sort()
    scan_paths = np.array(scan_paths)

    time_for_normal = []
    for scan_idx in range(10):
        # load point clouds
        start_load_point = time.time()
        scan = '/media/xieyuanlichen/71c5fd91-41fe-45c8-b0f1-a62b56513f8a/dataset/ipb_car/' \
               'first_recording/ipb_car_lidar_scans/000001.bin'
        current_vertex = np.fromfile(scan_paths[scan_idx], dtype=np.float32)
        current_vertex = current_vertex.reshape((-1, 4))
        current_points = current_vertex[:, :3]
        current_vertex = np.ones(
            (current_points.shape[0], current_points.shape[1] + 1))
        current_vertex[:, :-1] = current_points

        print('Time for loading point cloud: ', time.time() - start_load_point)

        # generate depth_map and normal_map
        time_depth_map = time.time()
        current_range, current_vertex = range_projection_vertex(
            current_vertex, fov_up=16.6, fov_down=-16.6)
        print('Time for generating depth map: ', time.time() - time_depth_map)
        time_normal_map = time.time()
        normal_map = gen_normal_map(current_range, current_vertex)
        print('Time for generating normal map: ',
              time.time() - time_normal_map)
        time_for_normal.append(time.time() - time_normal_map)

        # plt.scatter(current_vertex[:, 0], current_vertex[:, 1])

        # save img without axes
        fig = plt.figure(frameon=False)
        fig.set_size_inches(9, 0.64)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(normal_map, aspect='equal')
        fig.savefig('/home/xieyuanlichen/normal')
        ax.imshow(current_range, aspect='equal',
                  cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        fig.savefig('/home/xieyuanlichen/depth')

    print(np.mean(time_for_normal))


def search_nearest_index(pose1, pose2):
    nearest_index_2 = []
    index_1 = []
    for idx in range(len(pose1)):
        dist = np.linalg.norm(pose2 - pose1[idx], axis=1)
        if min(dist) < 5:
            index_1.append(idx)
            nearest_index_2.append(np.argmin(dist))

    return np.array(index_1), np.array(nearest_index_2)


def plot_range_image(image, proj_H=64, proj_W=900, save_result=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(proj_W/100., proj_H/100.)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='equal')
    plt.show()
    if save_result:
        fig.savefig('results/normal_map')


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels


def load_poses_kitti(pose_file, calib_file):
    """ load poses in kitti format """
    # laod poses
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert poses in LiDAR coordinate system
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    new_poses = np.array(new_poses)
    poses = new_poses

    return poses


if __name__ == '__main__':
    scan_path = '/media/xieyuanlichen/71c5fd91-41fe-45c8-b0f1-a62b56513f8a/dataset/' \
                'kitti-odometry/dataset/sequences/00/velodyne/000000.bin'

    current_vertex = load_vertex(scan_path)

    current_range, current_vertex = range_projection_vertex(current_vertex)

    normal_map = gen_normal_map(current_range, current_vertex)

    plt.imshow(normal_map)
    plt.show()
