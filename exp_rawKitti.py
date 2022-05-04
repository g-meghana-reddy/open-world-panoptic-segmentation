import os
import glob
import numpy as np
import pykitti
import pdb

# Change this to the directory where you store KITTI data
raw_dir = '/project_data/ramanan/achakrav/4D-PLS/data/Kitti-Raw/'
semantic_dir = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/'

# ======================================================================
# RANDOM EXPLORATION
# ======================================================================
# Specify the dataset to load
# date = '2011_09_26'
# drive = '0001'

# sequence = '00'

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

# raw_dataset = pykitti.raw(raw_dir, date, drive)

# print(raw_dataset.timestamps[0])

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# semantic_dataset = pykitti.odometry(semantic_dir, sequence)
# print(semantic_dataset.timestamps)
# ======================================================================

for drive_dir in glob.glob(raw_dir + '/*/*/'):
    tokens = drive_dir.split('/')
    date = tokens[-3]
    drive = tokens[-2].split('_')[-2]

    dataset = pykitti.raw(raw_dir, date, drive)
    P0 = dataset.calib.P_rect_00
    P1 = dataset.calib.P_rect_10
    P2 = dataset.calib.P_rect_20
    P3 = dataset.calib.P_rect_30
    Tr = dataset.calib.T_cam0_velo
    timestamps = dataset.timestamps
    imu_poses = [x.T_w_imu for x in dataset.oxts]
    T_imu_cam0 = np.linalg.inv(dataset.calib.T_cam0_imu)

    calib_filename = os.path.join(drive_dir, 'calib.txt')
    with open(calib_filename, 'w+') as f:
        f.write('P0: '); np.savetxt(f, P0, newline=' '); f.write('\n')
        f.write('P1: '); np.savetxt(f, P1, newline=' '); f.write('\n')
        f.write('P2: '); np.savetxt(f, P2, newline=' '); f.write('\n')
        f.write('P3: '); np.savetxt(f, P3, newline=' '); f.write('\n')
        f.write('Tr: '); np.savetxt(f, Tr, newline=' '); f.write('\n')

    times_filename = os.path.join(drive_dir, 'times.txt')
    with open(times_filename, 'w+') as f:
        for idx, timestamp in enumerate(timestamps):
            if idx == 0:
                f.write(str(float(idx)))
            else:
                timediff = timestamp - timestamps[idx-1]
                timediff_sec = timediff.total_seconds()
                f.write(str(timediff_sec))
            f.write('\n')

    poses_filename = os.path.join(drive_dir, 'poses.txt')
    with open(poses_filename, 'w+') as f:
        for pose_w_imu in imu_poses:
            pose_w_cam0 = pose_w_imu.dot(T_imu_cam0)
            np.savetxt(f, pose_w_cam0[:3], newline=' ')
            f.write('\n')