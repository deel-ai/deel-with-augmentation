"""
This module implements DCGAN as described in the paper:

Ref. Radford & al., Unsupervised Representation Learning With Deep Convolutional
Generative Aversarial Networks (2015). https://arxiv.org/abs/1511.06434
"""

from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from .base import BaseGAN, BaseGenerator, BaseDiscriminator

class DCGANGenerator(BaseGenerator):
    """
    A generator for mapping a latent space to a sample space.

    Parameters
    ----------
    num_channels
        Number of channels in the training images
    latent_size
        Size of latent vector (i.e. size of generator input)
    feature_map_size
        Size of feature maps in generator
    """

    def __init__(self, num_channels, latent_size, feature_map_size):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going to a convolution, state size. nz x 1 x 1
            nn.ConvTranspose2d(latent_size, feature_map_size*8, kernel_size=4,
                                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size*8),
            nn.ReLU(True),
            # current state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_size*8, feature_map_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size*4),
            nn.ReLU(True),
            # current state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size*4, feature_map_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size*2),
            nn.ReLU(True),
            # current state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size*2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # current state size. ngf x 32 x 32
            nn.ConvTranspose2d(feature_map_size, num_channels, 4, 2, 1, bias=False),
            # current state size. nc x 64 x 64
            nn.Tanh() # Produce number between 0 and 1 for pixel values
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function DCGANGenerator.
        """
        return self.main(noise)

class DCGANDiscriminator(BaseDiscriminator):
    """
    A discriminator for discerning real from generated images.
    Output activation is Sigmoid.

    Parameters
    ----------
    num_channels
        Number of channels in the training images
    feature_map_size
        Size of feature maps in discriminator
    """

    def __init__(self, num_channels, feature_map_size):
        super().__init__()
        self.main = nn.Sequential(
            # input is nc x 64 x 64
            nn.Conv2d(num_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state size. ndf x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            # current state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_map_size*2, feature_map_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            # current state size. ndf x 8 x 8
            nn.Conv2d(feature_map_size*4, feature_map_size*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            # current state size. (ndf*4) x 4 x 4
            nn.Conv2d(feature_map_size*8, 1, 4, 1, 0, bias=False),
            # current state size. (ndf*4) x 1 x 1
            nn.Sigmoid() # Produce probability
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function DCGANDiscriminator
        """
        return self.main(noise)

class DCGAN(BaseGAN):
    """
    A basic DCGAN class for generating images.

    Ref. Radford & al., Unsupervised Representation Learning With Deep Convolutional
    Generative Aversarial Networks (2015). https://arxiv.org/abs/1511.06434

    Parameters
    ----------
    generator
        A torch DCGAN Generator architecture
    discriminator
        A torch DCGAN Discriminator architecture
    optimizer_gen
        An optimizer for generator
    optimizer_dis
        An optimizer for discriminator
    loss_fn_gen
        A loss function for generator
    loss_fn_dis
        A loss function for discriminator
    device
        Cpu or CUDA
    latent_size
        Size of the latent space's vector (i.e size of generator input)
    """

    def train_generator(self, real_samples):
        """
        Train the generator one step and return the loss.

        Parameters
        ----------
        real_samples
            Famples from your training dataset

        Returns
        -------
        gen_loss
            The loss of the generator
        """
        real_label = 1
        self.generator.zero_grad()
        real_samples = real_samples[0].to(self.device)
        b_size = real_samples.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
        noise = torch.randn(b_size, self.latent_size, 1, 1, device=self.device)
        generated = self.generator(noise)
        classifications = self.discriminator(generated).view(-1)
        gen_loss = self.loss_gen(classifications, label)
        gen_loss.backward()
        # Update G
        self.optimizer_gen.step()
        return gen_loss

    def train_discriminator(self, real_samples):
        """
        Train the discriminator one optimization step and return the loss.

        Parameters
        ----------
        real_samples
            True samples of your dataset

        Returns
        -------
        dis_loss
            The loss of the discriminator
        """
        # Train with all-real batch
        real_label = 1
        fake_label = 0
        self.discriminator.zero_grad()
        ## Real samples
        real_samples = real_samples[0].to(self.device)
        b_size = real_samples.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
        ## Forward pass real batch through D
        pred_real = self.discriminator(real_samples).view(-1)
        ## Calculate loss on all-real batch
        loss_real = self.loss_dis(pred_real, label)
        ## Calculate gradients for D in backward pass
        loss_real.backward()

        # Train with all-fake batch
        noise = torch.randn(b_size, self.latent_size, 1, 1, device=self.device)
        ## Generate fake image batch with G
        fake_samples = self.generator(noise)
        label.fill_(fake_label)
        ## Classify all fake batch with G
        pred_fake = self.discriminator(fake_samples.detach()).view(-1)
        ## Calculate D's loss on the all-fake batch
        loss_fake = self.loss_dis(pred_fake, label)
        ## Calculate gradients for this batch
        loss_fake.backward()
        ## Add the gradients from the all-real and all-fake batches
        dis_loss = loss_real + loss_fake
        # Update D
        self.optimizer_dis.step()
        return dis_loss

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
        Train both networks and return the losses.

        Parameters
        ----------
        subset_a
            Torch.tensor or Dataset
        num_epochs
            The number of epochs you want to train your DCGan

        Returns
        -------
        gen_losses, dis_losses
            The losses of both the discriminator and generator
        """
        gen_losses = []
        dis_losses = []
        for epoch in range(num_epochs):
            for i, data in enumerate(subset_a, 0):
                loss_gen = self.train_generator(data)
                loss_dis = self.train_discriminator(data)
                #Output training stats
                if i % 50 == 0:
                    print(f"[{epoch+1}/{num_epochs}][{i}/{len(subset_a)}] \
                            \tLoss_D: {loss_dis.item()} \tLoss_G: {loss_gen.item()}")
                # Save Losses for plotting later
                gen_losses.append(loss_gen.item())
                dis_losses.append(loss_dis.item())
        return gen_losses, dis_losses

    def generate_samples(
        self,
        nb_samples: int,
        num_classes = None,
        real_image_a = None,
        real_image_b = None
    ):
        """
        Sample images from the generator.

        Parameters
        ----------
        nb_samples
            The number of samples to generate

        Returns
        -------
        img_list
            A list of generated images
        """
        fixed_noise = torch.randn(nb_samples, self.latent_size, 1, 1, device=self.device)
        with torch.no_grad():
            img_list = self.generator(fixed_noise).detach().cpu()
        return  img_list
