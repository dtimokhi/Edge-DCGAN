#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:39:33 2019

@author: Dima
"""
import torch
import torch.nn as nn
import sys
import AutoEncModel1 as am
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import cv2



''' Loading the model '''
def getEmptyModel():
    return am.Net()

def getModel(empty_net, model_path):
    empty_net.load_state_dict(torch.load(model_path))
    return empty_net

''' Loading the data '''
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
        shuffle=True
    )
    return train_loader

''' Canny edge function ''' 
def getCanny(image):
    nmp = image.numpy()
    nmp = np.transpose(nmp, (1, 2, 0))
    nmp = np.uint8(nmp)
    nmp = cv2.Canny(nmp, 218, 178)
    return nmp

''' Training function (1 epoch) '''
def train_model(model, loader, criterion, optimizer, epoch, device):
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        edges = torch.FloatTensor([getCanny(x).reshape([1,218,178]) for x in inputs])
        edges = edges.to(device)
        inputs = inputs.to(device)
        outputs = net(edges)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        # print statistics
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
    return loss

'''Train full model'''
def train_full(net, epochs, save_model_path, device):
    ## Load the Model
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
     
    ## Load the data
    train_loader = load_dataset('../train_data')
     
    ## Train the model
    for epoch in range(epochs):
        loss = train_model(net, train_loader, criterion, optimizer, epoch, device)
        print('Epoch %d loss: %.3f' %
              (epoch + 1, loss))
     
    ## Save the model
    torch.save(net, save_model_path)
    print("Training Done")

if __name__ == '__main__':
    ## First argument is the model path
    model_path = sys.argv[1]
    
    ## Second argument is the number of epochs
    epochs = np.int(sys.argv[2])
    
    ## Third argument is the save_model_path
    save_model = sys.argv[3]
    
    ## Use empty
    net = getEmptyModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    print(device)
    net.to(device)
    ## Get the model
    #n_net = getModel(net, model_path)
    
    ## Train the model
    train_full(net, epochs, save_model, device)
    
     

    