import numpy as np
import open3d as o3d
from PointCloudtoImage import point_cloud_to_image_points

image_shape = (1242, 375)


def semantic_mapping(pcd, img, calibration):
    camera_points = point_cloud_to_image_points(pcd, calibration)

    # print(camera_points.shape)

    # Mask to remove any point that has x or y > image coord
    x_mask = camera_points[:, 0] > image_shape[0] - 1
    y_mask = camera_points[:, 1] > image_shape[1] - 1

    # mask to remove neg values
    neg_mask = camera_points < 0
    neg_mask = np.any(neg_mask, axis=1)

    # filter points that are outside image coord
    # color the noisy points with white color. To achieve this, change the color of first pixel to white
    img[0, 0] = 255
    camera_points[x_mask] = 0
    camera_points[y_mask] = 0
    camera_points[neg_mask] = 0

    point_labels = img[camera_points[:, 1], camera_points[:, 0]]
    point_labels = np.array(point_labels)

    # Open3d colors should be in the range 0, 1
    point_labels = point_labels/255
    # print(point_labels)

    # Create new point cloud with semantic color
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points))
    new_pcd.colors = o3d.utility.Vector3dVector(point_labels)
    # print(len(np.array(pcd.points)), len(point_labels))

    # o3d.visualization.draw_geometries([new_pcd])

    return new_pcd
