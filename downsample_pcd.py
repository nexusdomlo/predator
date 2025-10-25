import open3d as o3d
import argparse
import os

def downsample_point_cloud(input_path, output_path, num_points):
    """
    Loads a point cloud, downsamples it using Farthest Point Sampling,
    and saves the result.

    Args:
        input_path (str): Path to the input .pcd file.
        output_path (str): Path to save the downsampled .pcd file.
        num_points (int): The desired number of points in the output cloud.
    """
    print(f"Loading point cloud from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        pcd = o3d.io.read_point_cloud(input_path)
    except Exception as e:
        print(f"Error reading point cloud file: {e}")
        return

    original_point_count = len(pcd.points)
    print(f"Original number of points: {original_point_count}")

    if original_point_count <= num_points:
        print("Point cloud already has fewer or equal points than the target. Copying file.")
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved to: {output_path}")
        return

    print(f"Downsampling to {num_points} points using Farthest Point Sampling...")
    
    # Farthest Point Sampling
    downsampled_pcd = pcd.farthest_point_down_sample(num_points)

    print(f"New number of points: {len(downsampled_pcd.points)}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    o3d.io.write_point_cloud(output_path, downsampled_pcd)
    print(f"Successfully saved downsampled point cloud to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a .pcd point cloud to a target number of points.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .pcd file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the downsampled .pcd file.")
    parser.add_argument("--num_points", type=int, required=True, help="Target number of points for the output point cloud.")

    args = parser.parse_args()

    downsample_point_cloud(args.input, args.output, args.num_points)
