""""
original code by Xinqiang Ding <xqding@umich.edu>: https://github.com/xqding/TD-VAE/blob/master/script/prep_data.py
"""

from typing import Dict, Tuple

import numpy as np
from torch.utils.data import Dataset


class MNIST_Dataset(Dataset):
    """
    MNIST dataset class for generating sequences of images that move in a

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        Dataset class from PyTorch
    """

    def __init__(
        self,
        image: np.array,
        label: np.array,
        binary: bool = True,
        number_of_frames: int = 20,
    ) -> None:
        """
        Initialize the dataset class

        Parameters
        ----------
        image : np.array
            Input image array
        label : np.array
            Input label array
        binary : bool, optional
            Binarize the input if true, by default True
        number_of_frames : int, optional
            number of frames to initialize , by default 20

        Methods
        -------
        __len__(self) -> int:
            Get the length of the dataset
        __getitem__(self, idx:int) -> Tuple[int, Dict[str, np.array]]:
            Get the item at the given index

        Returns
        -------
        None
        """

        super(MNIST_Dataset).__init__()
        self.image = image
        self.label = label
        self.binary = binary
        self.number_of_frames = number_of_frames

    def __len__(self) -> int:
        return self.label.shape[0]

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, np.array]]:
        image = np.copy(self.image[idx, :].reshape(28, 28))
        label = np.copy(self.label[idx,])

        if self.binary:
            # ## binarize MNIST images
            # this section of code was changed.
            # Otsu algorithm:
            # iteratively searches for the threshold that minimizes
            # the within-class variance, defined as a weighted sum of
            # variances of the two classes (background and foreground)

            nbins = 0.01
            all_colors = image.flatten()
            total_weight = len(all_colors)
            least_variance = -1
            least_variance_threshold = -1

            # create an array of all possible threshold values which we want to loop through
            color_thresholds = np.arange(
                np.min(image) + nbins, np.max(image) - nbins, nbins
            )

            # loop through the thresholds to find the one with the least within class variance
            if self.binary:
                # ## binarize MNIST images
                tmp = np.random.rand(28, 28)
                image = tmp <= image

        image = image.astype(np.float32)
        label = label.astype(np.int32)

        ## randomly choose a direction and generate a sequence
        ## of images that move in the chosen direction
        direction = np.random.choice(["left", "right"])
        image_list = []
        # image = np.roll(image, np.random.choice(np.arange(28)), 1)
        image_list.append(image.reshape(-1))
        for k in range(1, self.number_of_frames):
            if direction == "left":
                image = np.roll(image, -1, 1)
                image_list.append(image.reshape(-1))
            elif direction == "right":
                image = np.roll(image, 1, 1)
                image_list.append(image.reshape(-1))

        image_seq = np.array(image_list)
        label_seq = np.tile(label, (self.number_of_frames, 1))

        sample = {"image": image_seq, "label": label_seq}

        return idx, sample
