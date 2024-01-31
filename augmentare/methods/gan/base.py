"""
This is the base file for all GAN's architecture.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Optimizer

from ..utils.utils import weights_init

class BaseGenerator(nn.Module):
    """
    Base class for GAN's Generator.
    """

    @abstractmethod
    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function for Generator.
        """
        raise NotImplementedError

class BaseDiscriminator(nn.Module):
    """
    Base class for GAN's Discriminator.
    """

    @abstractmethod
    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function for Discriminator.
        """
        raise NotImplementedError

class BaseGAN(ABC):
    """
    Base class for all the GAN methods.

    Parameters
    ----------
    generator
        A torch GAN's Generator architecture
    discriminator
        A torch GAN's Discriminator architecture
    optimizer_gen
        An optimizer for the generator
    optimizer_dis
        An optimizer for the discriminator
    loss_fn_gen
        A loss function for the generator
    loss_fn_dis
        A loss function for the discriminator
    device
        Cpu or CUDA
    latent_size
        Size of the latent space's vector (i.e size of generator input)
    """
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        optimizer_gen: Optimizer,
        optimizer_dis: Optimizer,
        loss_fn_gen: Callable,
        loss_fn_dis: Callable,
        device: str = 'cpu',
        latent_size: Optional[int] = None,
        init_weights: bool = True
        ):

        self.device = device

        # put on either CPU or GPU
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        # handle multi-GPU
        if device == 'cuda':
            ngpu = torch.cuda.device_count()
            if ngpu > 1:
                self.generator = nn.DataParallel(self.generator,list(range(ngpu)))
                self.discriminator = nn.DataParallel(self.discriminator,list(range(ngpu)))

        # initialize both networks
        if init_weights:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        # put them on training mode
        self.generator.train()
        self.discriminator.train()

        # other useful parameters
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.loss_gen = loss_fn_gen
        self.loss_dis = loss_fn_dis
        self.device = device
        self.latent_size = latent_size

    @abstractmethod
    def train(
        self,
        subset_a: Union[torch.tensor, Dataset],
        num_epochs: int,
        num_decay_epochs = Optional[int],
        num_classes = Optional[int],
        batch_size = Optional[int],
        subset_b = Optional[Union[torch.tensor, Dataset]]
    ):
        """
        Train both generator and discriminator and return the losses.

        Parameters
        ----------
        subset_a
            Torch.tensor or Dataset on which to train the GAN.
            It is considered the main dataset for all GAN models
        num_epochs
            The number of epochs you want to train your GAN
        num_decay_epochs
            The number of epochs to start linearly decaying the learning rate to 0 (optional)
        num_classes
            Number of classes in the dataset (optional)
        batch_size
            Training batch size (optional)
        subset_b
            The second Torch.tensor or Dataset (optional).
            It is used when the GAN model works with two datasets, for example the CycleGAN model

        Returns
        -------
        gen_losses, dis_losses
            Respectively, the losses of the generator and the discriminator
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_samples(
        self,
        nb_samples: int,
        num_classes = Optional[int],
        real_image_a = Optional[torch.tensor],
        real_image_b = Optional[torch.tensor]
    ):
        """
        Generate images from the GAN's generator.

        Parameters
        ----------
        nb_samples
            The number of samples to generate
        num_classes
            Number of classes in dataset (optional)
        real_image_a
            Real image in subset_a (optional)
        real_image_b
            Real image in subset_b (optional)

        Returns
        -------
        img_list
            A list of generated images
        """
        raise NotImplementedError()
      