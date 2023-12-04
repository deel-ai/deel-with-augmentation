"""
This module implements CycleGAN as described in the paper:

Ref. Jun-Yan Zhu & al., Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks (2017).
https://arxiv.org/abs/1703.10593
"""

from typing import Optional, Union, Callable
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer

from .base import BaseGAN, BaseGenerator, BaseDiscriminator
from ..utils.utils import ResidualBlock, DecayLR, ReplayBuffer, weights_init

class CYCLEGANGenerator(BaseGenerator):
    """
    A conditional generator for synthesizing an image given an input image.

    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            #Convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3,64,7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            #Downsampling
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            #Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            #Upsampling
            nn.ConvTranspose2d(256,128,3,stride=2,padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            #Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,3,7),
            nn.Tanh()
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function CYCLEGANGenerator.
        """
        return self.main(noise)

class CYCLEGANDiscriminator(BaseDiscriminator):
    """
    A discriminator for predicting how likely the generated image is to
    have come from the target image collection.

    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4, stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,4,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,4,stride=2,padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,4,padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,1,4,padding=1)
        )

    def forward(
        self,
        noise,
        labels = Optional[torch.tensor]
    ):
        """
        A forward function CYCLEGANDiscriminator.
        """
        noise = self.main(noise)
        noise = F.avg_pool2d(noise,noise.size()[2:]) # pylint: disable=E1102
        noise = torch.flatten(noise,1)
        return noise

class CYCLEGAN(BaseGAN):
    """
    A CycleGAN class for training of image-to-image translation model without paired examples.

    Ref. Jun-Yan Zhu & al., Unpaired Image-to-Image Translation using Cycle-Consistent
    Adversarial Networks (2017). https://arxiv.org/abs/1703.10593

    Parameters
    ----------
    generator
        A torch CycleGAN Generator architecture
    discriminator
        A torch CycleGAN Discriminator architecture
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
        generator: CYCLEGANGenerator,
        discriminator: CYCLEGANDiscriminator,
        optimizer_gen: Optimizer,
        optimizer_dis: Optimizer,
        loss_fn_gen: Callable,
        loss_fn_dis: Callable,
        device: Union[str, torch.device],
        latent_size = Optional[int]
        ):

        super().__init__(
            generator,
            discriminator,
            optimizer_gen,
            optimizer_dis,
            loss_fn_gen,
            loss_fn_dis,
            device,
            latent_size
        )

        self.generator_a2b = deepcopy(self.generator)
        self.generator_a2b.apply(weights_init)
        self.generator_b2a = deepcopy(self.generator)
        self.generator_b2a.apply(weights_init)

        self.discriminator_a = deepcopy(self.discriminator)
        self.discriminator_a.apply(weights_init)
        self.discriminator_b = deepcopy(self.discriminator)
        self.discriminator_b.apply(weights_init)

        self.optimizer_gen_a = deepcopy(self.optimizer_gen)
        self.optimizer_gen_a.add_param_group({'params': self.generator_a2b.parameters()})
        self.optimizer_gen_b = deepcopy(self.optimizer_gen)
        self.optimizer_gen_b.add_param_group({'params': self.generator_b2a.parameters()})

        self.optimizer_dis_a = deepcopy(self.optimizer_dis)
        self.optimizer_dis_a.add_param_group({'params': self.discriminator_a.parameters()})
        self.optimizer_dis_b = deepcopy(self.optimizer_dis)
        self.optimizer_dis_b.add_param_group({'params': self.discriminator_b.parameters()})

    def train_generator(self, real_samples_a, real_samples_b):
        """
        Train the generator for one optimization step and return the loss.

        Parameters
        ----------
        real_samples_a
            Samples from your dataset A
        real_samples_b
            Samples from your dataset B

        Returns
        -------
        gen_loss
            The loss of the generator
        """
        # Get batch_size data
        real_image_a = real_samples_a.to(self.device)
        real_image_b = real_samples_b.to(self.device)
        batch_size = real_image_a.size(0)

        # Real data label is 1, fake data is 0
        real_labels = torch.full((batch_size,1),1,device=self.device,dtype=torch.float32)

        # (1) Update G network: Generator A2B and B2A

        ## Set G_A's and G_B's gradients to zero
        self.optimizer_gen_a.zero_grad()
        self.optimizer_gen_b.zero_grad()

        # Identity loss
        # G_B2A(A)
        identity_image_a = self.generator_b2a(real_image_a)
        loss_identity_a = self.loss_gen(identity_image_a, real_image_a)*5.0

        # G_A2B(B)
        identity_image_b = self.generator_a2b(real_image_b)
        loss_identity_b = self.loss_gen(identity_image_b, real_image_b)*5.0

        # GAN losss
        # GAN loss D_A(G_A(A))
        fake_image_a = self.generator_b2a(real_image_b)
        fake_output_a = self.discriminator_a(fake_image_a)
        loss_gan_b2a = self.loss_gen(fake_output_a, real_labels)
        # GAN loss D_B(G_B(B))
        fake_image_b = self.generator_a2b(real_image_a)
        fake_output_b = self.discriminator_b(fake_image_b)
        loss_gan_a2b = self.loss_gen(fake_output_b, real_labels)

        # Cycle loss
        recovered_image_a = self.generator_b2a(fake_image_b)
        loss_cycle_aba = self.loss_gen(recovered_image_a, real_image_a)*10.0

        recovered_image_b = self.generator_a2b(fake_image_a)
        loss_cycle_bab = self.loss_gen(recovered_image_b, real_image_b)*10.0

        # Combined loss and calculate gradients
        gen_loss = loss_identity_a + loss_identity_b + loss_gan_a2b \
                    + loss_gan_b2a + loss_cycle_aba + loss_cycle_bab
        # Calculate gradients for G_A and G_B
        gen_loss.backward()
        # Update G_A and G_B's weights
        self.optimizer_gen_a.step()
        self.optimizer_gen_b.step()
        return gen_loss

    def train_discriminator(self, real_samples_a, real_samples_b):
        """
        Train the discriminator for one optimization step and return the loss.

        Parameters
        ----------
        real_samples_a
            Samples from your dataset A
        real_samples_b
            Samples from your dataset B

        Returns
        -------
        dis_loss
            The loss of the discriminator
        """
        # Get batch_size data
        real_image_a = real_samples_a.to(self.device)
        real_image_b = real_samples_b.to(self.device)
        batch_size = real_image_a.size(0)

        # Real data label is 1, fake data is 0
        real_labels = torch.full((batch_size,1),1,device=self.device,dtype=torch.float32)
        fake_labels = torch.full((batch_size,1),0,device=self.device,dtype=torch.float32)

        # (2) Update D network: Discriminator A
        # Set D_A gradients to zero
        self.optimizer_dis_a.zero_grad()

        # Real A image loss
        real_output_a = self.discriminator_a(real_image_a)
        err_dis_real_a = self.loss_dis(real_output_a, real_labels)

        # Fake A image loss
        fake_image_a = self.generator_b2a(real_image_b)
        fake_a_buffer = ReplayBuffer()
        fake_image_a = fake_a_buffer.push_and_pop(fake_image_a)
        fake_output_a = self.discriminator_a(fake_image_a.detach())
        err_dis_fake_a = self.loss_dis(fake_output_a, fake_labels)

        # Combined loss and calculate gradients
        loss_dis_a = (err_dis_real_a + err_dis_fake_a)/2

        # Calculate gradients for D_A
        loss_dis_a.backward()
        # Update D_A weights
        self.optimizer_dis_a.step()

        # (3) Update D network: Discrimination B
        # Set D_B gradients to zero
        self.optimizer_dis_b.zero_grad()

        # Real B image loss
        real_output_b = self.discriminator_b(real_image_b)
        err_dis_real_b = self.loss_dis(real_output_b, real_labels)

        # Fake B image loss
        fake_image_b = self.generator_a2b(real_image_a)
        fake_b_buffer = ReplayBuffer()
        fake_image_b = fake_b_buffer.push_and_pop(fake_image_b)
        fake_output_b = self.discriminator_b(fake_image_b.detach())
        err_dis_fake_b = self.loss_dis(fake_output_b, fake_labels)

        # Combined loss and calculate gradients
        loss_dis_b = (err_dis_real_b + err_dis_fake_b)/2
        # Calculate gradients for D_B
        loss_dis_b.backward()
        # Update D_B weights
        self.optimizer_dis_b.step()
        dis_loss = loss_dis_a + loss_dis_b
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
        The corresponding training function

        Parameters
        ----------
        subset_a
            Torch.tensor or Dataset
        num_epochs
            The number of epochs you want to train your CycleGAN
        num_decay_epochs
            The number of epochs to start linearly decaying the learning rate to 0
        subset_b
            The second Torch.tensor or Dataset

        Returns
        -------
        gen_losses, dis_losses
            The losses of both the discriminator and generator
        """
        lr_lambda = DecayLR(num_epochs, num_decay_epochs).step
        lr_scheduler_gen_a = torch.optim.lr_scheduler.LambdaLR(self.optimizer_gen_a,
                                                                lr_lambda=lr_lambda)
        lr_scheduler_gen_b = torch.optim.lr_scheduler.LambdaLR(self.optimizer_gen_b,
                                                                lr_lambda=lr_lambda)
        lr_scheduler_dis_a = torch.optim.lr_scheduler.LambdaLR(self.optimizer_dis_a,
                                                                lr_lambda=lr_lambda)
        lr_scheduler_dis_b = torch.optim.lr_scheduler.LambdaLR(self.optimizer_dis_b,
                                                                lr_lambda=lr_lambda)

        gen_losses = []
        dis_losses = []
        for epoch in range(num_epochs):
            for i, (data_a, data_b) in enumerate(zip(subset_a, subset_b)):
                data_a = data_a.unsqueeze(0)
                data_b = data_b.unsqueeze(0)
                loss_gen = self.train_generator(data_a, data_b)
                loss_dis = self.train_discriminator(data_a, data_b)
                #Output training stats
                if i % 50 == 0:
                    print(f"[{epoch+1}/{num_epochs}][{i}/{len(subset_a)}] \
                            \tLoss_D: {loss_dis.item()} \tLoss_G: {loss_gen.item()}")
                # Save Losses for plotting later
                gen_losses.append(loss_gen.item())
                dis_losses.append(loss_dis.item())
            # Update learning rates
            lr_scheduler_gen_a.step()
            lr_scheduler_gen_b.step()
            lr_scheduler_dis_a.step()
            lr_scheduler_dis_b.step()
        return gen_losses, dis_losses

    def generate_samples(
        self,
        nb_samples = None,
        num_classes = None,
        real_image_a = Optional[torch.tensor],
        real_image_b = Optional[torch.tensor]
    ):
        """
        Sample images from the generator.

        Parameters
        ----------
        real_image_a
            Real image in subset_a
        real_image_b
            Real image in subset_b

        Returns
        -------
        fake_image_a
            A list of generated images A
        fake_image_b
            A list of generated images B
        """
        real_image_a = real_image_a.to(self.device)
        real_image_b = real_image_b.to(self.device)

        fake_image_a = 0.5*(self.generator_b2a(real_image_b).data + 1.0)
        fake_image_b = 0.5*(self.generator_a2b(real_image_a).data + 1.0)
        return fake_image_a, fake_image_b
