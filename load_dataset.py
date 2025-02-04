import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


""" 
Load the dataset containing the point clouds and attempt to visualize it
Attempt to filter the data, i.e, removing points that are too far away or irrelevant.
Downsample the point cloud for higher efficiency.
"""

# --------------------------------------------------------------------------------------------------------------------
"""
Open3d Notes:
    PointCloud() is a class. It consists of point coordinates.
    Vector3dVector() converts Float64 numpy array of shape(n,3) to Open3D compatible format
    draw_geometries is a function that draws a list of geometry objects.
    voxel_down_sample is a function to downsample input pointcloud into output pointcloud with a voxel.
        A 'Voxel' is essentially a 3D pixel. Thats the best analogy.
"""
# --------------------------------------------------------------------------------------------------------------------


class Dataset():

    def load_point_cloud(file):
        """ Load the point cloud data from the KITTI dataset .bin files and their corresponding labels."""
        points = np.fromfile(file, dtype=np.float32).reshape(-1,4)
        return points
    
    def see_point_cloud(point_clouds):
        """ Visually see the point clouds that are loaded from the dataset """
        pc_data = o3d.geometry.PointCloud()
        pc_data.point_clouds = o3d.utility.Vector3dVector(point_clouds[:, :3])
        o3d.visualization.draw_geometries([pc_data])
    
    def filter_point_cloud(point_clouds):
        """Filter and downscale/downsample the point clouds from the given dataset."""
        pc_data = o3d.geometry.PointCloud()
        pc_data.point_clouds = o3d.utility.Vector3dVector(point_clouds[:, :3])
        downscale_pc_data = pc_data.voxel_down_sample(voxel_size=0.1)
        return np.asarray(downscale_pc_data.point_clouds)