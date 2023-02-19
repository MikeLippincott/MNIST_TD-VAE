__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
"""


import numpy as np
from torch.utils.data import Dataset


class MNIST_Dataset(Dataset):
    def __init__(self, image, binary=True):
        super(MNIST_Dataset).__init__()
        self.image = image
        self.binary = binary

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        image = np.copy(self.image[idx, :].reshape(28, 28))

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
            for color_threshold in color_thresholds:
                bg_pixels = all_colors[all_colors < color_threshold]
                weight_bg = len(bg_pixels) / total_weight
                variance_bg = np.var(bg_pixels)

                fg_pixels = all_colors[all_colors >= color_threshold]
                weight_fg = len(fg_pixels) / total_weight
                variance_fg = np.var(fg_pixels)

                within_class_variance = (
                    weight_fg * variance_fg + weight_bg * variance_bg
                )

                if least_variance == -1 or least_variance > within_class_variance:
                    least_variance = within_class_variance
                    least_variance_threshold = color_threshold
            image[image >= color_threshold] = 1
            image[image < color_threshold] = 0
        image = image.astype(np.float32)

        ## randomly choose a direction and generate a sequence
        ## of images that move in the chosen direction
        direction = np.random.choice(["left", "right"])
        image_list = []
        # image = np.roll(image, np.random.choice(np.arange(28)), 1)
        image_list.append(image.reshape(-1))
        for k in range(1, 20):
            if direction == "left":
                image = np.roll(image, -1, 1)
                image_list.append(image.reshape(-1))
            elif direction == "right":
                image = np.roll(image, 1, 1)
                image_list.append(image.reshape(-1))

        image_seq = np.array(image_list)
        return image_seq
