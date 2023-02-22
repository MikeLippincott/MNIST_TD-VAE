__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
train the model
"""

import pathlib
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from prep_data import *

# Set paths
data_file = "MNIST.pkl"
log_file = pathlib.Path("loginfo.txt")

# Set constants
time_constant_max = 16  # There are 20 frames total
time_jump_options = [1, 2, 3, 4]  # Jump up to 4 frames away


# Set hyperparameters
batch_size = 512
num_epoch = 6000
learning_rate = 0.0005

# Prepare dataset
with open(data_file, "rb") as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST["train_image"])
# second most important for overall performance
# mini batch gradient descent
# default method for implementing gradient descent in deep learning
# computationally effecient, stable convergance, faster learning
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Build a TD-VAE model
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8  # from original paper
d_block_hidden_size = 50
decoder_hidden_size = 200

tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)

# "CUDA is a parallel computing platform and application programming interface (API)
# that allows software to use certain types of graphics processing units (GPUs)
# for general purpose processing, an approach called general-purpose computing
# on GPUs (GPGPU). CUDA is a software layer that gives direct access to the
# GPU's virtual instruction set and parallel computational elements, for the
# execution of compute kernels"

# https://en.wikipedia.org/wiki/CUDA
tdvae = tdvae.cuda()

# Train model
optimizer = optim.Adam(tdvae.parameters(), lr=learning_rate)

with open(log_file, "w") as log_file_handle:
    for epoch in range(num_epoch):
        for idx, images in enumerate(data_loader):
            images = images.cuda()

            # Make a forward step of preprocessing and LSTM
            tdvae.forward(images)

            # Randomly sample a time step and jumpy step
            t_1 = np.random.choice(time_constant_max)
            t_2 = t_1 + np.random.choice(time_jump_options)

            # Calculate loss function based on two time points
            loss = tdvae.calculate_loss(t_1, t_2)

            # must clear out stored gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(
                    epoch, idx, loss.item()
                ),
                file=log_file_handle,
                flush=True,
            )

            print(
                "epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(
                    epoch, idx, loss.item()
                )
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

'''
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
'''