# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

from torch.utils.data import Dataset
from load_dataset import load_point_cloud

"""
This file will create the PyTorch dataloader, which helps efficiently
load and iterate over our data subset during training and inference. 
It can help with batching or shuffling.
"""


class KITTI_Dataset(Dataset):
    def __init__(self, _bin_files, _labels):
        self.bin_files = _bin_files
        self.labels = _labels

    def __len__(self):
        return len(self.bin_files)
    
    def __getitem__(self, index):
        points = load_point_cloud(self.bin_files[index])
        label = self.labels[index]
        return points, label
    
