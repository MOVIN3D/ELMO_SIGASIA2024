import numpy as np
import h5py
from core.utils import animation_plot
import core.animation as anim
import os
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description='Visualize motion capture and point cloud data.')
parser.add_argument('--bvh', type=str, required=True, help='Path to the BVH file')
parser.add_argument('--h5', type=str, required=True, help='Path to the H5 file')
args = parser.parse_args()

# Use the provided file paths
bvh_file = args.bvh
h5_file_path = args.h5

# Load motion bvh data
# Load corresponding pcd data (hdf5)
mot = anim.Animation()
mot.load_bvh(bvh_file)   

pcd_np_dict = {}
num_loaded_frames = 0
with h5py.File(h5_file_path, 'r') as pcd_data:
    total_frames = len(pcd_data.keys())
    for frame_idx in range(len(pcd_data.keys())):
        group_name = f"frame-{frame_idx:06}"
        if group_name in pcd_data:
            group = pcd_data[group_name]
            try :
                pcd = group['pointcloud'][()]
                pcd_np_dict[frame_idx] = np.asarray(pcd, dtype=np.float32)
                num_loaded_frames += 1
            except:
                pcd_np_dict[frame_idx] = np.array([])
                print(f'No pcd data on frame number: {frame_idx:4}')
        else:
            raise ValueError('No pcd data on %s' % group_name)
print(f'Loaded {num_loaded_frames} among {total_frames} frames from {h5_file_path}')


animation_plot(mot, pcd_np_dict, fps=20)
