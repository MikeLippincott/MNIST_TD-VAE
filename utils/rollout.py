"""
original code by Xinqiang Ding <xqding@umich.edu>
After training the model, we can try to use the model to do jumpy predictions.
"""

import pathlib
import pickle

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import gridspec
from model import TD_VAE
from prep_data import *
from torch.utils.data import DataLoader


def rollout_func(
    model_path: pathlib.Path,
    input_size: int,
    processed_x_size: int,
    belief_state_size: int,
    d_block_hidden_size: int,
    decoder_hidden_size: int,
    state_size: int,
    mnist_pickle_path: pathlib.Path,
    epoch: str | int,
    batch_size: int,
    num_frames: int = 20,
    t1: int = 16,
    t2: int = 19,
) -> None:
    """
    This function is used to do jumpy predictions using the trained model.

    Parameters
    ----------
    model_path : pathlib.Path
        The path to the trained model.
    input_size : int
        The size of the input tensor.
    processed_x_size : int
        The size of the processed input tensor.
    belief_state_size : int
        The size of the belief state tensor.
    state_size : int
        The size of the state tensor.
    mnist_pickle_path : pathlib.Path
        The path to the mnist pickle file.
    num_frames : int
        The number of frames in the dataset.
    epoch : str | int
        The epoch number.
    batch_size : int
        The batch size.
    num_frames : int, optional
        The number of frames in the dataset, by default 20
    t1 : int, optional
        The t1 value to use for rollout predictions for prior frames, by default 16
    t2 : int, optional
        The t2 value to use for rollout predictions for future frames, by default 19

    """
    #### load trained model
    checkpoint = torch.load(model_path, weights_only=True)
    input_size = input_size
    processed_x_size = processed_x_size
    belief_state_size = belief_state_size
    state_size = state_size
    tdvae = TD_VAE(
        x_size=input_size,
        processed_x_size=processed_x_size,
        b_size=belief_state_size,
        z_size=state_size,
        d_block_hidden_size=d_block_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
    )
    num_frames = num_frames
    tdvae.load_state_dict(checkpoint["model_state_dict"])

    #### load dataset
    with open(mnist_pickle_path, "rb") as file_handle:
        MNIST = pickle.load(file_handle)
    tdvae.eval()
    tdvae = tdvae.cuda()

    data = MNIST_Dataset(
        MNIST["train_image"],
        MNIST["train_label"],
        binary=True,
        number_of_frames=num_frames,
    )
    batch_size = 6
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    idx, images = next(enumerate(data_loader))

    images = images[1]["image"].cuda()
    idx = images[0]

    ## calculate belief
    # forward step in the model to generate the belief state and latent state
    # we can grab the decoded jumpy prediction from the model after this step
    tdvae.forward(images)

    # assertions:
    assert t1 < t2
    assert t2 < num_frames
    assert t1 >= 0
    assert batch_size > 0

    rollout_images = tdvae.rollout(images, t1, t2)
    #### plot results
    fig = plt.figure(0, figsize=(12, 4))

    fig.clf()
    gs = gridspec.GridSpec(batch_size + 1, t2 + 2)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(batch_size):
        for j in range(t1):
            axes = plt.subplot(gs[i, j])
            axes.imshow(
                1 - images.cpu().data.numpy()[i, j].reshape(28, 28), cmap="binary"
            )
            axes.axis("off")

        for j in range(t1, t2 + 1):
            axes = plt.subplot(gs[i, j + 1])
            axes.imshow(
                1 - rollout_images.cpu().data.numpy()[i, j - t1].reshape(28, 28),
                cmap="binary",
            )
            axes.axis("off")

    for j in range(t1):
        axes = plt.subplot(gs[i + 1, j])
        # add the label below the image
        axes.text(0.5, 0.5, f"{1 + j}", fontsize=16, ha="center")
        axes.axis("off")
    for j in range(t1, t2 + 1):
        axes = plt.subplot(gs[i + 1, j + 1])
        # add the label below the image
        axes.text(0.5, 0.5, f"{1 + j}", fontsize=16, ha="center")
        axes.axis("off")
    fig.savefig(f"../output/rollout_result_{epoch}.png")
    plt.show(fig)
    plt.close(fig)
