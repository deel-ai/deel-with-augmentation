"""
This module implements CDCGAN as described in the paper:

Radford & al., Unsupervised Representation Learning With Deep Convolutional
Generative Aversarial Networks (2015). https://arxiv.org/abs/1511.06434
"""

from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .base import BaseGAN, BaseGenerator, BaseDiscriminator
from ..utils.utils import conv_block

class CDCGANGenerator(BaseGenerator):
    """
    A generator for mapping a latent space to a sample space.

    Parameters
    ----------
    num_classes
        Number of classes in the trining dataset
    latent_size
        Size of the latent space's vector (i.e size of generator input)
    label_embed_size
        Label embedding size
    channels
        Number of channels in the training images
    conv_dim
        Dimension of convolution
    """
    def __init__(self, num_classes, latent_size, label_embed_size, channels, conv_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_embed_size)
        self.conv1 = conv_block(
            latent_size + label_embed_size,
            conv_dim*4,
            padding=0,
            transpose=True
        )
        self.conv2 = conv_block(conv_dim*4, conv_dim*2, transpose=True)
        self.conv3 = conv_block(conv_dim*2, conv_dim, transpose=True)
        self.conv4 = conv_block(conv_dim, channels, use_bn=False, transpose=True)

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function for CDCGANGenerator.
        """
        # reshape noise vector
        noise = noise.reshape([noise.shape[0], -1, 1, 1])

        # embed labels and reshape
        label_emded = self.label_embedding(labels)
        label_emded = label_emded.reshape([label_emded.shape[0], -1, 1, 1])

        # concatenate noise and label embeddings
        noise = torch.cat((noise, label_emded), dim=1)

        # forward pass
        noise = F.relu(self.conv1(noise))
        noise = F.relu(self.conv2(noise))
        noise = F.relu(self.conv3(noise))
        noise = torch.tanh(self.conv4(noise))

        return noise

class CDCGANDiscriminator(BaseDiscriminator):
    """
    A discriminator for discerning real from generated images.
    Output activation is Sigmoid.

    Parameters
    ----------
    num_classes
        Number of classes in the training dataset
    channels
        Number of channels in the training images
    conv_dim
        Dimension of convolution layers
    image_size
        Size of images
    """
    def __init__(self, num_classes, channels, conv_dim, image_size):
        super().__init__()
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, image_size*image_size)
        self.conv1 = conv_block(channels + 1, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, kernel_size=4, stride=1, padding=0, use_bn=False)

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function CDCGANDiscriminator.
        """
        alpha = 0.2
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.reshape([label_embed.shape[0], 1,
                                            self.image_size, self.image_size])
        noise = torch.cat((noise, label_embed), dim=1)
        noise = F.leaky_relu(self.conv1(noise), alpha)
        noise = F.leaky_relu(self.conv2(noise), alpha)
        noise = F.leaky_relu(self.conv3(noise), alpha)
        noise = torch.sigmoid(self.conv4(noise))
        return noise.squeeze()

class CDCGAN(BaseGAN):
    """
    A basic CDCGAN class for generating images.

    Ref. Radford & al., Unsupervised Representation Learning With Deep Convolutional
    Generative Aversarial Networks (2015). https://arxiv.org/abs/1511.06434

    Parameters
    ----------
    generator
        A torch CDCGAN Generator architecture
    discriminator
        A torch CDCGAN Discriminator architecture
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
        Size of latent vector(i.e size of generator input)
    """

    def train_generator(self, real_samples, real_labels, batch_size):
        """
        Train the generator one optimization step and return the loss.

        Parameters
        ----------
        real_samples
            Samples from your training dataset
        real_labels
            True labels of real samples
        batch_size
            Batch size

        Returns
        -------
        gen_loss
            The loss of the generator
        """
        # Labels
        is_real = torch.ones(batch_size).to(self.device)

        real_samples = real_samples.to(self.device)
        real_labels = real_labels.to(self.device)
        z_fake = torch.randn(batch_size, self.latent_size, device=self.device)

        # Generate fake data
        fake_samples = self.generator(z_fake, real_labels)

        # Train Generator
        fake_out = self.discriminator(fake_samples, real_labels)
        gen_loss = self.loss_gen(fake_out, is_real)

        self.optimizer_gen.zero_grad()
        gen_loss.backward()
        self.optimizer_gen.step()
        return gen_loss

    def train_discriminator(self, real_samples, real_labels, batch_size):
        """
        Train the discriminator one optimization step and return the loss.

        Parameters
        ----------
        real_samples
            True samples of your dataset
        real_labels
            True labels of real samples
        batch_size
            Batch size

        Returns
        -------
        dis_loss
            The loss of the discriminator
        """
        # Labels
        is_real = torch.ones(batch_size).to(self.device)
        is_fake = torch.zeros(batch_size).to(self.device)

        real_samples = real_samples.to(self.device)
        real_labels = real_labels.to(self.device)
        z_fake = torch.randn(batch_size, self.latent_size, device=self.device)

        # Generate fake data
        fake_samples = self.generator(z_fake, real_labels)

        # Train Discriminator
        fake_out = self.discriminator(fake_samples.detach(), real_labels)
        real_out = self.discriminator(real_samples.detach(), real_labels)
        dis_loss = (self.loss_dis(fake_out,is_fake) + self.loss_dis(real_out,is_real)) / 2

        self.optimizer_dis.zero_grad()
        dis_loss.backward()
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
            The number of epochs you want to train your CDCGan
        batch_size
            Training batch size

        Returns
        -------
        gen_losses, dis_losses
            Respectively, the losses of the generator and the discriminator
        """
        gen_losses = []
        dis_losses = []
        for epoch in range(num_epochs):
            for i, (real_samples, real_labels) in enumerate(subset_a):
                loss_gen = self.train_generator(real_samples, real_labels, batch_size)
                loss_dis = self.train_discriminator(real_samples, real_labels, batch_size)
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
        nb_samples : int,
        num_classes = Optional[int],
        real_image_a = None,
        real_image_b = None
    ):
        """
        Sample images from the generator.

        Parameters
        ----------
        nb_samples
            The number of samples to generate for one class
        num_classes
            Number of classes in dataset

        Returns
        -------
        img_list
            A list of generated images
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(nb_samples*num_classes, self.latent_size).to(self.device)
            label = torch.arange(0, num_classes)
            label = torch.repeat_interleave(label, nb_samples).to(self.device)
            img_list = self.generator(noise, label)
        return img_list
