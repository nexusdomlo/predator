# import torch
# import os

# # Define the paths to the .pth files
# path_21 = 'assets/cloud_bin_21.pth'
# path_34 = 'assets/cloud_bin_34.pth'

# print(f"Checking point counts for original demo files...")

# # Check and load the first file
# if os.path.exists(path_21):
#     try:
#         pcd_21 = torch.load(path_21)
#         num_points_21 = pcd_21.shape[0]
#         print(f"'{path_21}' loaded successfully.")
#         print(f"Number of points in cloud_bin_21: {num_points_21}")
#     except Exception as e:
#         print(f"Error loading '{path_21}': {e}")
# else:
#     print(f"Error: '{path_21}' not found. Please make sure you have run 'sh scripts/download_data_weight.sh'.")

# print("-" * 30)

# # Check and load the second file
# if os.path.exists(path_34):
#     try:
#         pcd_34 = torch.load(path_34)
#         num_points_34 = pcd_34.shape[0]
#         print(f"'{path_34}' loaded successfully.")
#         print(f"Number of points in cloud_bin_34: {num_points_34}")
#     except Exception as e:
#         print(f"Error loading '{path_34}': {e}")
# else:
#     print(f"Error: '{path_34}' not found. Please make sure you have run 'sh scripts/download_data_weight.sh'.")
import open3d as o3d


src_pcd = o3d.io.read_point_cloud("/home/brian/DCP/Cut/data_2_cut.pcd")
src_pcd = src_pcd.voxel_down_sample(0.05)
# tgt_pcd = o3d.io.read_point_cloud("/home/brian/DCP/Cut/data_2.pcd")
# tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
print(len(src_pcd.points))