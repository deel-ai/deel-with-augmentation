"""
This implements the plot functions.
"""

import os
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.utils as vutils

def plot_image(img_list, number=None):
    """
    Plot a number of generated images with optional label

    Parameters
    ----------
    img_list
        A list of generated images (Torch tensor)
    number
        Number of images we want to plot
    """
    if number is None:
        number = img_list.shape[0]
    else:
        number = min(number , 36)

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Fake images")
    plt.imshow(np.transpose(vutils.make_grid(img_list[:number],
                                                padding=5,
                                                normalize=True).cpu(),(1,2,0)))
    plt.show()

def plot_images(img_list, labels=None, number=None):
    """
    Plot a number of generated images with optional label

    Parameters
    ----------
    img_list
        A list of generated images (Torch tensor)
    labels
        Labels of generated images (Torch tensor - optional)
    number
        Number of images we want to plot
    """
    if number is None:
        number = img_list[0].shape[0]
    else:
        number = min(number , 36)
    nrows = int(np.sqrt(number))
    ncols = number // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
    axs = np.ravel(axs)
    for i, a_x in enumerate(axs):
        #ax.imshow(img_list[i])
        #ax.imshow(np.transpose(img_list[i],(1,2,0)))
        a_x.axis("off")
        if labels is not None:
            a_x.set_title(f"Label: {labels[i]}")

    fig.tight_layout()
    #plt.show()
    return fig, axs

def create_gif(source_path, target_path=None):
    """
    Create a GIF from images contained on the source path.

    Parameters
    ----------
    source_path
        Path pointing to the source directory with .png files. (string)
    target_path
        Name of the created GIF. (string, optional)
    """
    source_path = source_path+"/" if not source_path.endswith("/") else source_path
    images = []
    for file_name in sorted(os.listdir(source_path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(source_path, file_name)
            images.append(imageio.imread(file_path))

    if target_path is None:
        target_path = source_path+"movie.gif"
    imageio.mimsave(target_path, images)

def plot_to_tensorboard(writer, loss_dis, real, fake, tensorboard_step):
    """
    Function used to plot the results on the tensorboard tool if needed.
    """
    writer.add_scalar("Loss Discriminator", loss_dis, global_step=tensorboard_step)
    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)
