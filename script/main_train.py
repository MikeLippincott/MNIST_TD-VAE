__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
train the model
"""

import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from prep_data import *

#### preparing dataset
with open("MNIST.pkl", "rb") as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST["train_image"])
# second most important for overall performance
# mini batch gradient descent
# default method for implementing gradient descent in deep learning
# computationally effecient, stable convergance, faster learning
batch_size = 512
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

#### build a TD-VAE model
input_size = 784
# TODO: where does processed_x_size come from?
processed_x_size = 784
belief_state_size = 50
# from original paper: state size = 8
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)

# "CUDA is a parallel computing platform and application programming interface (API)
# that allows software to use certain types of graphics processing units (GPUs)
# for general purpose processing, an approach called general-purpose computing
# on GPUs (GPGPU). CUDA is a software layer that gives direct access to the
# GPU's virtual instruction set and parallel computational elements, for the
# execution of compute kernels"

# https://en.wikipedia.org/wiki/CUDA
tdvae = tdvae.cuda()

#### training
optimizer = optim.Adam(tdvae.parameters(), lr=0.0005)
num_epoch = 6000
log_file_handle = open("./log/loginfo.txt", "w")
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader):
        images = images.cuda()
        tdvae.forward(images)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1, 2, 3, 4])
        loss = tdvae.calculate_loss(t_1, t_2)
        # must clear out stored gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
            file=log_file_handle,
            flush=True,
        )

        print(
            "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item())
        )

    if (epoch + 1) % 50 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": tdvae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"./output/model_epoch_{epoch}.pt",
        )
log_file_handle.close()

# info about the model
params = list(tdvae.parameters())
for i in range(len(params)):
    print(params[i].size())

# example weight matrix
print(params[0])

# architecture
print(tdvae)

# matrix tranformations
# TODO: figure out how to extract matrix size after each transformation
summary(tdvae)
summary(tdvae(1, 784))
summary(tdvae, (60000, 784))

