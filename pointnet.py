import torch
import torch.nn as nn
import torch.nn.functional as nnF


"""
PointNet 'Deep Learning on Point Sets for 3D Classification and Segmentation'
https://arxiv.org/pdf/1612.00593
PointNet is a simple yet effective model for point cloud classification and segmentation.
We are going to try to implement it in PyTorch and use CUDA 
If we are unable to implement it, the model is available on HuggingFace.
https://huggingface.co/keras-io/pointnet_segmentation

It takes a Point Cloud (x,y,z) and Fully Connected (FC) layers extract high-dimensional features
ReLU activations introduce some non-linearity (ability to model complex relationships between inputs and outputs)
"""

class PointNet(nn.Module):
    def __init__(self, number_of_classes):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv3d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = nnF.relu(self.conv1(x))
        x = nnF.relu(self.conv2(x))
        x = nnF.relu(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = nnF.relu(self.fc1(x))
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#model = PointNet().cuda() so it can be run on a gpu.