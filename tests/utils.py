"""
These utils are implemented for testing.
"""

import torch

def load_images(num_samples=100, num_classes=10, num_channels=3, height=32, width=32):
    """
    A function to create dataset for testing.

    Parameters
    ----------
    num_samples
        Number of images
    num_classes
        Number of classes in dataset
    num_channels
        Number of channels
    height
        Height of image
    width
        Width of image

    Returns
    -------
    images
        All images in dataset
    labels
        Label corresponding to each image
    """
    images = torch.randn(num_samples, num_channels, height, width)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels
