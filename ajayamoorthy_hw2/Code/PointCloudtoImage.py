import numpy as np


def point_cloud_to_image_points(pcd, calibration):
    pcd_points = np.array(pcd.points)

    number_of_points = pcd_points.shape[0]

    # homogeneous coordinate system
    pcd_points = np.hstack((pcd_points, np.ones((number_of_points, 1), dtype=np.float32)))

    # Extrinsic parameters - to homogeneous coordinate system
    velodyne_to_camera = np.vstack((calibration.Tr_velo_to_cam, np.array([0, 0, 0, 1])))

    R_rect = np.zeros((4, 4))
    R_rect[:3, :3] = calibration.R0_rect
    R_rect[3, 3] = 1

    P2 = np.vstack((calibration.P2, np.array([0, 0, 0, 1])))

    camera_points = P2 @ R_rect @ velodyne_to_camera @ pcd_points.T

    # Get x & y
    camera_points /= camera_points[2]

    camera_points = camera_points[:2].T
    camera_points = camera_points.astype(np.int32)

    return camera_points
