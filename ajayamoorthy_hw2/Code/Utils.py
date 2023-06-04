import numpy as np
import open3d as o3d


def velodyne_to_pcd(file_location):
    # Load binary point cloud
    bin_pcd = np.fromfile(file_location, dtype=np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    return pcd
