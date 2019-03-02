#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:33:46 2019

@author: Dima
"""

import torch
import torch.nn as nn
import torch.optim as optim

'''Model initialization'''
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
if __name__ == '__main__':
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    torch.save(net, 'model_enc_1.pt')