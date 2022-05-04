import os
import glob
import numpy as np

# from datasets.SemanticKitti import *
# from models.architectures import KPFCNN
from utils.ply import read_ply, write_ply
# from utils.visualizer import ModelVisualizer, show_ModelNet_models

# import open3d as o3d

import pdb


if __name__ == '__main__':
    # checkpoint_dir = "../../mganesin/4D-PLS/results/4DPLS_TS2/"
    semanticKitti_dir = 'data/SemanticKitti/sequences/08/velodyne/'
    kitti360_dir = 'data/Kitti360/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data/'
    file_id = 100

    semanticKitti_file = os.path.join(semanticKitti_dir,'{0:06d}.bin'.format(file_id))
    semanticKitti_points = np.fromfile(semanticKitti_file, dtype=np.float32).reshape(-1, 4)
    semanticKitti_colors = np.array([[255., 0., 0.]]).repeat(semanticKitti_points.shape[0], 0)
    write_ply('{0:06d}'.format(file_id), [semanticKitti_points[:, :3], semanticKitti_colors], ['x', 'y', 'z', 'r', 'g', 'b'])

    kitti360_file = os.path.join(kitti360_dir, '{0:010d}.bin'.format(file_id))
    kitti360_points = np.fromfile(kitti360_file, dtype=np.float32).reshape(-1, 4)
    kitti360_colors = np.array([[0., 0., 255.]]).repeat(kitti360_points.shape[0], 0)
    write_ply('{0:010d}'.format(file_id), [kitti360_points[:, :3], kitti360_colors], ['x', 'y', 'z', 'r', 'g', 'b'])

    # semanticKitti_pcd = o3d.geometry.PointCloud()
    # semanticKitti_pcd.points = o3d.utility.Vector3dVector(semanticKitti_points[:, :3])
    # semanticKitti_colors = np.array([[1., 0., 0.]]).repeat(semanticKitti_points.shape[0], 0) # red
    # semanticKitti_pcd.colors = o3d.utility.Vector3dVector(semanticKitti_colors)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True) # works for me with False, on some systems needs to be true
    # vis.add_geometry(semanticKitti_pcd)
    # vis.update_geometry(semanticKitti_pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image('tmp.png')
    # vis.destroy_window()

    # o3d.visualization.draw_geometries(
    #     [semanticKitti_pcd],
    #     # zoom=0.3412,
    #     # front=[0.4257, -0.2125, -0.8795],
    #     # lookat=[2.6172, 2.0475, 1.532],
    #     # up=[-0.0694, -0.9768, 0.2024]
    # )
    
    # for ply_file in glob.glob(os.path.join(checkpoint_dir, "val_preds/*.ply")):
    #     data = read_ply(ply_file)
    #     points = np.vstack((data['x'], data['y'], data['z'])).T
    #     x, y, z = data['x'], data['y'], data['z']
    #     pred = data['pre']
    #     truth = data['gt']
    #     pdb.set_trace()
        # mlab.points3d(x, y, z, pred)
        # show_ModelNet_models(points)
    