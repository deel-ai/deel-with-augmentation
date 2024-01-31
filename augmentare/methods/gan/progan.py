"""
This module implements ProGAN as described in the paper:

Tero Karras & al., Progressive Growing of GANs for Improved Quality,
Stability, and Variation (2017). https://arxiv.org/abs/1710.10196
"""

from math import log2
from typing import Optional, Union, Callable

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer

from .base import BaseGAN, BaseGenerator, BaseDiscriminator
from ..utils.utils import WSConv2d, PixelNorm, CNNBlock, gradient_penalty, \
                    fade_in_gen, fade_in_dis, minibatch_std

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
torch.backends.cudnn.benchmarks = True

class PROGANGenerator(BaseGenerator):
    """
    A generator to incrementally size the output by starting with a very small image,
    then the blocks of layers added incrementally and increasing the input
    size of the discriminant model until the desired image size is obtained.

    Parameters
    ----------
    latent_size
        Size of latent vector(i.e size of generator input)
    in_channels
        In channels
    img_channels
        Number of channels in the training images
    alpha
        Alpha should be scalar within [0,1]. Adding a new block of layers using
        skip connection to connect the new block to the input of the discriminator
        or output of the generator and finally add to the existing input or output
        layer with weighting. The weighting parameter alpha controls the influence
        of the new block and that starts at zero or a very small number and linearly
        increases to 1.0 over training iterations
    step
        Number of steps to increase image size from starting size to desired size
    """
    def __init__(self, latent_size, in_channels, img_channels=3, alpha=1e-5, steps=4):
        super().__init__()

        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(latent_size, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]) , nn.ModuleList([self.initial_rgb]))

        for i in range (len(factors) - 1):
            conv_in_c = int(in_channels*factors[i])
            conv_out_c = int(in_channels*factors[i+1])
            self.prog_blocks.append(CNNBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels,
                                            kernel_size=1, stride=1, padding=0))

        self.alpha = alpha
        self.steps = steps

    def forward(self, noise, labels=None):
        """
        A forward function PROGANGenerator.
        """
        out = self.initial(noise)
        steps = self.steps

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return fade_in_gen(self.alpha, final_upscaled, final_out)

class PROGANDiscriminator(BaseDiscriminator):
    """
    A discriminator for discerning real from generated images.

    Parameters
    ----------
    in_channels
        In channels
    img_channels
        Number of channels in the training images
    alpha
        Alpha should be scalar within [0,1]. Adding a new block of layers using
        skip connection to connect the new block to the input of the discriminator
        or output of the generator and finally add to the existing input or output
        layer with weighting. The weighting parameter alpha controls the influence
        of the new block and that starts at zero or a very small number and linearly
        increases to 1.0 over training iterations
    step
        Number of steps to increase image size from starting size to desired size
    """
    def __init__(self, in_channels, img_channels=3, alpha=1e-5, steps=4):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # input size 1024x1024 -> 512 -> 256 -> ...
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels*factors[i])
            conv_out = int(in_channels*factors[i-1])
            self.prog_blocks.append(CNNBlock(conv_in,conv_out,
                                            pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in,
                                            kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)

        # Down sampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # input size 4x4
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1)
        )

        self.alpha = alpha
        self.steps = steps

    def forward(self, noise, labels=None):
        """
        A forward function PROGANDiscriminator.
        """
        steps = self.steps

        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](noise))

        if steps == 0: # image size is 4x4
            out = minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step+1](self.avg_pool(noise)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = fade_in_dis(self.alpha, downscaled, out)

        for step in range(cur_step+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

class PROGAN(BaseGAN):
    """
    A basic ProGAN class for synthesizing high resolution and high quality images via the
    incremental growing of the discriminator and the generator networks during the training process.

    Ref. Tero Karras & al., Progressive Growing of GANs for Improved Quality,
    Stability, and Variation (2017). https://arxiv.org/abs/1710.10196

    Parameters
    ----------
    generator
        A torch ProGAN Generator architecture
    discriminator
        A torch ProGAN Discriminator architecture
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
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        optimizer_gen: Optimizer,
        optimizer_dis: Optimizer,
        loss_fn_gen: Callable,
        loss_fn_dis: Callable,
        device,
        latent_size: Optional[int] = None,
    ):
        super().__init__(
            generator,
            discriminator,
            optimizer_gen,
            optimizer_dis,
            loss_fn_gen,
            loss_fn_dis,
            device,
            latent_size,
            init_weights=False
        )

    def _set_alpha(self, new_alpha):
        self.generator.alpha = new_alpha
        self.discriminator.alpha = new_alpha

    def _set_steps(self, new_steps):
        self.generator.steps = new_steps
        self.discriminator.steps = new_steps

    def train_generator(self, noise):
        """
        Train the generator one step and return the loss.

        Parameters
        ----------
        noise
            Noise for train generator

        Returns
        -------
        gen_loss
            The loss of the generator
        """
        with torch.cuda.amp.autocast():
            fake = self.generator(noise)
            net_gen_fake = self.discriminator(fake)
            gen_loss = -torch.mean(net_gen_fake)

        if self.device == 'cuda':
            self.optimizer_gen.zero_grad()
            self.loss_gen.scale(gen_loss).backward()
            self.loss_gen.step(self.optimizer_gen)
            self.loss_gen.update()
        elif self.device == 'cpu':
            self.optimizer_gen.zero_grad()
            gen_loss.backward()
            self.optimizer_gen.step()
        return gen_loss

    def train_discriminator(self, real_samples, noise):
        """
        Train the discriminator one step and return the loss.

        Parameters
        ----------
        real_samples
            True samples of your dataset
        noise
            Noise for train discriminator

        Returns
        -------
        dis_loss
            The loss of the discriminator
        """
        real_samples = real_samples.to(self.device)
        lamda_gb = 10
        with torch.cuda.amp.autocast():
            fake = self.generator(noise)
            dis_real = self.discriminator(real_samples)
            dis_fake = self.discriminator(fake.detach())
            g_p = gradient_penalty(
                self.discriminator,
                real_samples,
                fake,
                self.device
            )
            dis_loss = (-(torch.mean(dis_real) - torch.mean(dis_fake)) \
                        + lamda_gb*g_p + (0.001 * torch.mean(dis_real**2)))

        if self.device == 'cuda':
            self.optimizer_dis.zero_grad()
            self.loss_dis.scale(dis_loss).backward()
            self.loss_dis.step(self.optimizer_dis)
            self.loss_dis.update()
        elif self.device == 'cpu':
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
            Torch.tensor or Dataset on which to train the ProGAN
        num_epochs
            The number of epochs you want to train your ProGAN
        batch_size
            Training batch size

        Returns
        -------
        gen_losses, dis_losses
            The losses of both the discriminator and generator
        """
        gen_losses = []
        dis_losses = []
        start_image_size = 256
        step = int(log2(start_image_size/4))
        progressive_epochs = [num_epochs]*len(batch_size)
        self._set_steps(step)

        for epochs in progressive_epochs[step:]:
            alpha = 1e-5
            for epoch in range(epochs):
                loop = tqdm(subset_a, leave=True)
                for i, (real,_) in enumerate(loop):
                    real = real.to(self.device)
                    cur_batch_size = real.shape[0]
                    noise = torch.randn(cur_batch_size, self.latent_size, 1, 1).to(self.device)

                    loss_dis = self.train_discriminator(real, noise)
                    loss_gen = self.train_generator(noise)

                    # Update alpha and ensure less than 1
                    alpha += cur_batch_size / ((progressive_epochs[step] * 0.5) * len(subset_a))
                    alpha = min(alpha, 1)
                    self._set_alpha(alpha)

                    #Output training stats
                    if i % 500 == 0:
                        print(f"[{epoch+1}/{num_epochs}][{i}/{len(subset_a)}] \
                            \tLoss_D: {loss_dis.item()} \tLoss_G: {loss_gen.item()}")
                    # Save Losses for plotting later
                    gen_losses.append(loss_gen.item())
                    dis_losses.append(loss_dis.item())
        return gen_losses, dis_losses

    def generate_samples(
        self,
        nb_samples : int,
        num_classes = None,
        real_image_a = None,
        real_image_b = None,
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
        self.generator.eval()
        self.generator.alpha = 1.0
        with torch.no_grad():
            noise = torch.randn(nb_samples, self.latent_size, 1, 1).to(self.device)
            img_list = self.generator(noise) * 0.5 + 0.5
        return img_list
