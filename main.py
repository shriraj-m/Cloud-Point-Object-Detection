from torch.utils.data import DataLoader
from dataloader import KITTI_Dataset
from pointnet import PointNet

import torch.nn as nn
import torch

bin_files = 'path'
labels = [0, 1] # example
dataset = KITTI_Dataset(bin_files, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


model = PointNet(number_of_classes=10)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10): # Number of Epochs
    for i, (points, labels) in enumerate(dataloader):
        points = points.transpose(2,1) # This is because PointNet expects (batch_size, 3, num_points)
        outputs = model(points)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch[{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')


