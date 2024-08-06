#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas
import torch
import tqdm
from matplotlib import gridspec
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

utils_path = pathlib.Path("../../utils/").resolve(strict=True)
sys.path.append(str(utils_path))

from model import TD_VAE, DBlock, Decoder, PreProcess
from prep_data import MNIST_Dataset
from rollout import rollout_func

# In[2]:


# set up logging
logger = logging.getLogger(__name__)
# make the log directory
pathlib.Path("../log").mkdir(exist_ok=True)
logging.basicConfig(filename="../log/training_log.log", level=logging.INFO)


# In[3]:


# set path to the MNIST images
mnist_pickle_path = pathlib.Path("../../data/mnist/MNIST.pkl").resolve(strict=True)
# create the log directory if it does not exist
log_path = pathlib.Path("../log/").resolve()
log_path.mkdir(exist_ok=True)
log_file_path = pathlib.Path("../log/loginfo.txt").resolve()


# In[4]:


with open(mnist_pickle_path, "rb") as file_handle:
    MNIST = pickle.load(file_handle)

# get the MNIST data keys
print(MNIST.keys())
MNIST["train_image"].shape


# In[5]:


# set the batch size
batch_size = 512

# create the data class
# this class makes a rolling window of the data
data = MNIST_Dataset(
    MNIST["train_image"], MNIST["train_label"], binary=True, number_of_frames=20
)
# create the data loader
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


# ## Hyperparameter optimization with Optuna

# In[6]:


# Build a TD-VAE model
# dataset dependent constants

input_size = 784  # dataset dependent
processed_x_size = 784  # dataset dependent
# Set constants
time_constant_max = 16  # There are 20 frames total
time_jump_options = [1, 2, 3, 4]  # Jump up to 4 frames away

# hyper parameters
num_epochs = 1000
learning_rate = 0.0005
belief_state_size = 50  # hyperparameter
state_size = 8  # from original paper and hyperparameter
d_block_hidden_size = 50  # hyperparameter
decoder_hidden_size = 200  # hyperparameter

logger.info("Parameters set: ")
logger.info(f"num_epochs: {num_epochs}")
logger.info(f"learning_rate: {learning_rate}")
logger.info(f"belief_state_size: {belief_state_size}")
logger.info(f"state_size: {state_size}")
logger.info(f"d_block_hidden_size: {d_block_hidden_size}")
logger.info(f"decoder_hidden_size: {decoder_hidden_size}")
logger.info(f"batch_size: {batch_size}")
logger.info(f"input_size: {input_size}")
logger.info(f"processed_x_size: {processed_x_size}")
logger.info(f"time_constant_max: {time_constant_max}")
logger.info(f"time_jump_options: {time_jump_options}")


# In[7]:


tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)
tdvae = tdvae.cuda()
# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# In[8]:


optimizer = optim.Adam(tdvae.parameters(), lr=learning_rate)
# make model save directory
model_save_dir = pathlib.Path("../output").resolve()
model_save_dir.mkdir(parents=True, exist_ok=True)

# Train the model

for epoch in range(num_epochs):
    epoch_loss = 0

    for batch, (idx, images) in enumerate(data_loader):
        batch_counter = 0
        batch_loss = 0
        images = images["image"].cuda()
        # Make a forward step of preprocessing and LSTM
        tdvae.forward(images)

        # Randomly sample a time step and jumpy step
        t_1 = np.random.choice(time_constant_max)
        t_2 = t_1 + np.random.choice(time_jump_options)

        # Calculate loss function based on two time points
        loss = tdvae.calculate_loss(t_1, t_2)
        if loss.isnan():
            print("loss is nan")
            pass
        elif loss.isinf():
            print("loss is inf")
            pass
        elif loss.item() == 0:
            print("loss is zero")
            pass
        elif loss.item() < 0:
            print("loss is negative")
            pass
        elif loss.item() > 0:
            batch_counter += 1
            batch_loss += loss.item()
            # must clear out stored gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_loss += batch_loss / batch_counter
    logger.info(f"epoch: {epoch}, loss: {epoch_loss}")
    print("epoch: {:>4d}, loss: {:.4f}".format(epoch, epoch_loss))

    # save the model every 5 epochs and plot the jumpy reconstruction
    if (epoch + 1) % 50 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": tdvae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            pathlib.Path(model_save_dir / f"model_epoch_{epoch}.pt").resolve(),
        )
        plot = rollout_func(
            model_path=pathlib.Path(
                model_save_dir / f"model_epoch_{epoch}.pt"
            ).resolve(),
            input_size=input_size,
            processed_x_size=processed_x_size,
            belief_state_size=belief_state_size,
            state_size=state_size,
            d_block_hidden_size=d_block_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            mnist_pickle_path=mnist_pickle_path,
            epoch=epoch,
            batch_size=batch_size,
            num_frames=20,
            t1=16,
            t2=19,
        )
# save the final model
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": tdvae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    pathlib.Path(model_save_dir / f"model_epoch_final.pt").resolve(),
)


# In[9]:


epoch = "final"
rollout_func(
    model_path=pathlib.Path(model_save_dir / f"model_epoch_{epoch}.pt").resolve(),
    input_size=input_size,
    processed_x_size=processed_x_size,
    belief_state_size=belief_state_size,
    state_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
    mnist_pickle_path=mnist_pickle_path,
    epoch=epoch,
    batch_size=batch_size,
    num_frames=20,
    t1=16,
    t2=19,
)
