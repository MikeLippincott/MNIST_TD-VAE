__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>

train the model
"""


import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
import sys
from torchsummary import summary

with open("MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])
batch_size = 512
data_loader = DataLoader(data, batch_size = batch_size,
                         shuffle = True)

input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda()
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

checkpoint = torch.load("./output/model_epoch_2899.pt")
tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

log_file_handle = open("loginfo_test.txt", 'a')

for epoch in range(1):
    for idx, images in enumerate(data_loader):
        images = images.cuda()       
        tdvae.forward(images)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss = tdvae.calculate_loss(t_1, t_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)
        
        #print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "./output/model_epoch_{}.pt".format(epoch))



# info about the model 

params = list(tdvae.parameters())
print(len(params))
print(params[0].size())
print(params[0])
print(tdvae)

summary(tdvae)
summary(tdvae (1, 784))
summary(tdvae, (60000, 784))
