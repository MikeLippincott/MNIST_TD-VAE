"""
original code by Xinqiang Ding <xqding@umich.edu>
After training the model, we can try to use the model to do jumpy predictions.
"""

import pathlib

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import gridspec
from model import TD_VAE
from prep_data import *
from torch.utils.data import DataLoader


def rollout_func(
    rollout_stack: torch.Tensor,
    images: torch.Tensor,
    epoch: str | int,
    batch_size: int,
    save_path: pathlib.Path,
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
                1 - rollout_stack.cpu().data.numpy()[i, j - t1].reshape(28, 28),
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
    fig.savefig(pathlib.Path(save_path / f"rollout_result_{epoch}.png"))
    plt.show(fig)
    plt.close(fig)
