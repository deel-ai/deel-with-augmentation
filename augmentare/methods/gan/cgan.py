"""
This module implements CGAN as described in the paper:

Mehdi Mirza & Simon Osindero., Conditional Generative Adversarial Nets (2014).
https://arxiv.org/abs/1411.1784
"""

from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.autograd import Variable

from .base import BaseGAN, BaseGenerator, BaseDiscriminator

class CGANGenerator(BaseGenerator):
    """
    A generator is parameterized to learn and produce
    realistic samples for each label in the training dataset.

    Parameters
    ----------
    latent_size
        Size of the latent space's vector (i.e size of generator input)
    num_classes
        Number of classes in the training dataset, more than 0 for Conditional GANs
    image_shape
        Shape of image (C, H, W)
    """
    def __init__(self, latent_size, num_classes, image_shape):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes,num_classes)
        self.image_shape = image_shape
        self.layer = 128

        self.model = nn.Sequential(
            nn.Linear(latent_size + num_classes, self.layer),
            nn.BatchNorm1d(self.layer, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.layer, self.layer*2),
            nn.BatchNorm1d(self.layer*2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.layer*2, self.layer*4),
            nn.BatchNorm1d(self.layer*4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.layer*4, self.layer*8),
            nn.BatchNorm1d(self.layer*8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.layer*8, np.prod(self.image_shape)),
            nn.Tanh()
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function CGANGenerator.
        """
        emb = self.label_embedding(labels)
        gen_input = torch.cat([noise,emb], 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.image_shape)
        return img

class CGANDiscriminator(BaseDiscriminator):
    """
    A discriminator learns to distinguish fake and real samples, given the label information.
    Output activation is Sigmoid.

    Parameters
    ----------
    num_classes
        Number of classes in training dataset, more than 0 for Conditional GANs
    image_shape
        Shape of image (C, H, W)
    """
    def __init__(self, num_classes, image_shape):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes , num_classes)
        self.image_shape = image_shape
        self.layer = 512

        self.model = nn.Sequential(
            nn.Linear(num_classes + np.prod(image_shape), self.layer),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(self.layer , self.layer),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(self.layer , self.layer*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(self.layer*2,1),
            nn.Sigmoid()
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function CGANDiscriminator.
        """
        imgs = noise.view(noise.size(0), np.prod(self.image_shape))
        emb = self.label_embedding(labels)
        dis_input = torch.cat([imgs,emb], 1)
        imgs = self.model(dis_input)
        return imgs

class CGAN(BaseGAN):
    """
    A basic CGAN class for generating images with the condition.

    Ref. Mehdi Mirza & Simon Osindero., Conditional Generative Adversarial Nets (2014).
    https://arxiv.org/abs/1411.1784

    Parameters
    ----------
    generator
        A torch CGAN Generator architecture
    discriminator
        A torch CGAN Discriminator architecture
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

    def train_generator(self, num_classes, batch_size):
        """
        Train the generator one step and return the loss.

        Parameters
        ----------
        num_classes
            Number of classes in dataset
        batch_size
            Training batch size

        Returns
        -------
        gen_loss
            The loss of the generator
        """
        is_real = Variable(torch.ones(batch_size, 1, device=self.device))
        noise = Variable(torch.randn(batch_size, self.latent_size, device=self.device))
        fake_label = Variable(torch.LongTensor(np.random.randint(0,
                                num_classes, batch_size))).to(self.device)
        fake_data = self.generator(noise, fake_label)

        dis_result_from_fake = self.discriminator(fake_data, fake_label)
        gen_loss = self.loss_gen(dis_result_from_fake, is_real)

        self.generator.zero_grad()
        gen_loss.backward()
        self.optimizer_gen.step()
        return gen_loss

    def train_discriminator(self, real_samples, num_classes, batch_size):
        """
        Train the discriminator one step and return the loss.

        Parameters
        ----------
        real_samples
            Samples from the training dataset
        num_classes
            Number of classes in dataset
        batch_size
            Training batch size

        Returns
        -------
        dis_loss
            The loss of the discriminator
        """
        is_real = Variable(torch.ones(batch_size, 1, device=self.device))
        is_fake = Variable(torch.zeros(batch_size, 1, device=self.device))
        real_data = real_samples[0].to(self.device)
        real_label  = real_samples[1].to(self.device)
        dis_result_from_real = self.discriminator(real_data, real_label)
        dis_loss_real = self.loss_dis(dis_result_from_real, is_real)

        noise = Variable(torch.randn(batch_size, self.latent_size, device=self.device))
        fake_label = Variable(torch.LongTensor(np.random.randint(0,
                                num_classes, batch_size))).to(self.device)
        fake_data = self.generator(noise, fake_label)

        dis_result_from_fake = self.discriminator(fake_data, fake_label)
        dis_loss_fake = self.loss_dis(dis_result_from_fake, is_fake)

        dis_loss = dis_loss_real + dis_loss_fake

        self.discriminator.zero_grad()
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
            Torch.tensor or Dataset on which to train the CGAN
        num_epochs
            The number of epochs you want to train your CGAN
        num_classes
            Number of classes in dataset
        batch_size
            Training batch size

        Returns
        -------
        gen_losses, dis_losses
            The losses of both the generator and discriminator
        """
        gen_losses = []
        dis_losses = []
        for epoch in range(num_epochs):
            for i, data in enumerate(subset_a):
                loss_gen = self.train_generator(num_classes, batch_size)
                loss_dis = self.train_discriminator(data, num_classes, batch_size)
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
        num_classes = Optional[int],
        real_image_a = None,
        real_image_b = None
    ):
        """
        Sample images from the generator.

        Parameters
        ----------
        nb_samples
            The number of samples to generate
        num_classes
            Number of classes in dataset

        Returns
        -------
        img_list
            A list of generated images (one image of one class is successively a list)
        """
        fixed_noise = Variable(torch.randn(nb_samples, self.latent_size, device=self.device))
        fake_label = Variable(torch.LongTensor(np.zeros(nb_samples))).to(self.device)
        num_label = len(fake_label)
        for i in range(num_label):
            fake_label[i] = i % num_classes
        img_list = self.generator(fixed_noise, fake_label).cpu()
        return img_list
