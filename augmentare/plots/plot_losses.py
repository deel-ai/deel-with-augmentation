"""
This is the function that plots the losses of the model.
"""

import matplotlib.pyplot as plt
def plot_losses_gan(gen_losses, dis_losses):
    """
    Plots losses for generator and discriminator on a common plot.

    Parameters
    ----------
    gen_losses
        A list of generator losses
    dis_losses
        A list of discriminator losses
    """
    plt.figure(figsize=(12, 9))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(dis_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_losses(losses):
    """
    Plots losses during training model.

    Parameters
    ----------
    losses
        A list of losses
    """
    plt.figure(figsize=(12, 9))
    plt.title("Loss During Training")
    plt.plot(losses, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
