import open3d as o3d
import numpy as np
import os
import cv2
import argparse

from Utils import velodyne_to_pcd
from ICP import perform_icp
from Calibration import Calibration
from SemanticMapping import semantic_mapping
from Segmentation import segmentation

# Add any Command Line arguments here
Parser = argparse.ArgumentParser()
Parser.add_argument('--flag', type=int, default=0, help='flag: 0-Segmented Image 1-RGB Image, Default:0')
Args = Parser.parse_args()
flag = Args.flag

# Define the input and output directories
data_directory = "Data/"
point_cloud_directory = "PointCloud/"
image_directory = "Images/"
calibration_directory = "calib/"
segmentedImg_directory = "SegImages/"
output_directory = "output/"

if flag == 0:
    # Semantic Segmentation of RGB images
    segmentation(input_location=data_directory + image_directory,
                 output_location=data_directory + segmentedImg_directory)

# File names of all the input data
point_cloud_filenames = [file for file in os.listdir(data_directory + point_cloud_directory)]
image_filenames = [file for file in os.listdir(data_directory + image_directory)]
segmentedImg_filenames = [file for file in os.listdir(data_directory + segmentedImg_directory)]
calibration_filenames = [file for file in os.listdir(data_directory + calibration_directory)]

# Step 1 - Read calibation
calib = Calibration(data_directory + calibration_directory + calibration_filenames[0])

icp_pcd_list = []

for i in range(len(point_cloud_filenames)):
    # Step 2 - Read the point cloud
    pcd = velodyne_to_pcd(data_directory + point_cloud_directory + point_cloud_filenames[i])
    # o3d.visualization.draw_geometries([pcd])
    print(pcd)

    # Step 3 - Read image
    if flag == 1:
        image = cv2.imread(data_directory + image_directory + image_filenames[i], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: 
        image = cv2.imread(data_directory + segmentedImg_directory + segmentedImg_filenames[i], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 4 - Call semantic mapping
    new_pcd = semantic_mapping(pcd, image, calib)

    # Write the new semantic mapped point cloud into a file
    o3d.io.write_point_cloud(data_directory + output_directory + point_cloud_filenames[i].rsplit('.', 1)[0] + ".pcd",
                             new_pcd)

    icp_pcd_list.append(new_pcd)

# Step 5 - ICP for point cloud registration
final_pcd = perform_icp(icp_pcd_list)
o3d.visualization.draw_geometries([final_pcd])

# Write the new semantic mapped point cloud into a file
o3d.io.write_point_cloud(data_directory + output_directory + "final_result.pcd", final_pcd)
