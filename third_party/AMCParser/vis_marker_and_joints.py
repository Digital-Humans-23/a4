"""This script is to visualize the marker and joints together in CMU Mocap
"""



import os, glob, sys
import open3d as o3d
import scipy.io as sio
import numpy as np




markers_file = '/home/yzhang/Downloads/c3d_matlab/markers_01_01.mat'
marker_locs = sio.loadmat(markers_file)['Markers']/1000

joints_file = '/mnt/hdd/datasets/CMU/allasfamc/all_asfamc/subjects/01/motion_00000.pkl'
joint_locs = np.load(joints_file, allow_pickle=True)['joints_locs']
ROT_POSITIVE_X = np.array([  [1.0000000,  0.0000000,  0.0000000],
                             [0.0000000,  0.0000000,  -1.0000000],
                             [0.0000000, 1.0000000,  0.0000000]])

ROT_POSITIVE_Z = np.array([  [0.0000000, -1.0000000,  0.0000000],
                             [1.0000000,  0.0000000,  0.0000000],
                             [0.0000000,  0.0000000,  1.0000000]])




joint_locs = np.einsum('ij, tpj->tpi', ROT_POSITIVE_X, joint_locs)
joint_locs = np.einsum('ij, tpj->tpi', ROT_POSITIVE_Z, joint_locs)

frame = 2000


markers = o3d.geometry.PointCloud()
markers.points = o3d.utility.Vector3dVector(marker_locs[frame])
markers.paint_uniform_color([0.7, 0.3, 0.3])

joints_pcl = o3d.geometry.PointCloud()
joints_pcl.points = o3d.utility.Vector3dVector(joint_locs[frame])
joints_pcl.paint_uniform_color([0.1, 0.3, 0.7])




coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
o3d.visualization.draw_geometries([markers, joints_pcl, coord])
