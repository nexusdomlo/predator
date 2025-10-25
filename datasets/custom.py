import os
import torch
import numpy as np
import open3d
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Custom dataset for testing Predator on user-provided point clouds.
    """
    def __init__(self, config, split, data_augmentation=False):
        super(CustomDataset, self).__init__()
        assert split == 'test', "CustomDataset is only for testing."
        self.config = config
        self.root = config.root
        self.voxel_size = config.first_subsampling_dl
        
        self.files = []
        self.prepare_files(split)

    def prepare_files(self, split):
        """
        Reads the test_pairs.txt file to get the list of point cloud pairs.
        """
        pair_file = os.path.join(self.root, 'test_pairs.txt')
        if not os.path.exists(pair_file):
            raise FileNotFoundError(f"'{pair_file}' not found. Please create it.")
            
        with open(pair_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                
                pcd0_path = os.path.join(self.root, 'pcds', parts[0])
                pcd1_path = os.path.join(self.root, 'pcds', parts[1])

                if not os.path.exists(pcd0_path):
                    print(f"Warning: {pcd0_path} not found, skipping pair.")
                    continue
                if not os.path.exists(pcd1_path):
                    print(f"Warning: {pcd1_path} not found, skipping pair.")
                    continue
                    
                self.files.append((pcd0_path, pcd1_path))
        
        print(f'Found {len(self.files)} pairs in the custom dataset.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads a pair of point clouds, voxelizes them, and returns them.
        """
        pcd0_path, pcd1_path = self.files[idx]

        # Load point clouds
        pcd0_o3d = open3d.io.read_point_cloud(pcd0_path)
        pcd1_o3d = open3d.io.read_point_cloud(pcd1_path)

        # Voxel downsample
        src_pcd_o3d = pcd0_o3d.voxel_down_sample(self.voxel_size)
        tgt_pcd_o3d = pcd1_o3d.voxel_down_sample(self.voxel_size)

        src_pcd = np.array(src_pcd_o3d.points, dtype=np.float32)
        tgt_pcd = np.array(tgt_pcd_o3d.points, dtype=np.float32)

        # The model expects features, but for custom data, we can use ones.
        src_feats = np.ones_like(src_pcd[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1], dtype=np.float32)

        # Dummy transformations and correspondences for compatibility
        rot = np.eye(3, dtype=np.float32)
        trans = np.zeros((3, 1), dtype=np.float32)
        matching_inds = torch.empty(0, 2, dtype=torch.int64)

        return (
            src_pcd, tgt_pcd, 
            src_feats, tgt_feats, 
            rot, trans, 
            matching_inds, 
            src_pcd, tgt_pcd, 
            torch.ones(1)
        )
