import open3d as o3d
import numpy as np
import os


def icp_point_to_point(source, target, pcd_index):
    # Initialize the transformation matrix
    current_transformation = np.array([
        [1, 0, 0, pcd_index],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    result = o3d.pipelines.registration.registration_icp(source,
                                                         target,
                                                         5,
                                                         current_transformation,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def perform_icp(pcd_list):
    # First file is the target pcd in ICP
    target_pcd = pcd_list[0]

    for i in range(1, len(pcd_list)):
        source_pcd = pcd_list[i]

        icp_result = icp_point_to_point(source_pcd, target_pcd, pcd_index=i)
        print(icp_result)

        # Transform the source pcd
        source_pcd = source_pcd.transform(icp_result.transformation)

        # Combine it with target pcd
        target_pcd += source_pcd

    return target_pcd
