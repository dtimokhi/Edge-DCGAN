#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:48:40 2019

@author: Dima
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

''' Load the dataset'''
def load_dataset(data_path):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def getCanny(image):
    nmp = image.numpy()
    nmp = np.transpose(nmp, (1, 2, 0))
    nmp = np.uint8(nmp)
    nmp = cv2.Canny(nmp, 218, 178)
    return nmp

''' Build the neural network'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Conv Blocks
        self.Conv2dBlock1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Conv2dBlock2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Conv2dBlock3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        ## Transposed block
        self.TransposedConv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
        )  
        
    def forward(self, x):
        x = self.Conv2dBlock1(x)
        x = self.Conv2dBlock2(x)
        x = self.Conv2dBlock3(x)
        x = self.TransposedConv(x)
        return x

