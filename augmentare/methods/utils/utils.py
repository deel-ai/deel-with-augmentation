"""
This utils implements for methods.
"""

import random
import torch
from torch import nn

def weights_init(m_in):
    """
    Custom weights initialization called on net_gen and net_dis.
    """
    classname = m_in.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m_in.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m_in.weight, 1.0, 0.02)
        nn.init.constant_(m_in.bias, 0)

def fade_in_gen(alpha, upscaled, generated):
    """
    Alpha should be scalar within [0, 1], and upscale.shape == generated.shape.
    This function is used in ProGAN.
    """
    return torch.tanh(alpha*generated + (1-alpha)*upscaled)

def fade_in_dis(alpha, downscaled, out):
    """
    Alpha should be scalar within [0, 1]
    This function is used in ProGAN.
    """
    return alpha*out + (1-alpha)*downscaled

def minibatch_std(noise):
    """
    The standard deviation of activation function across the images in the mini-batch is added
    as a new channel which is prior to the last block of convolutional layers
    in the discriminator model.
    """
    batch_statistics = (torch.std(noise, dim=0).mean().repeat(noise.shape[0],1,
                        noise.shape[2],noise.shape[3]))
    return torch.cat([noise, batch_statistics], dim=1)

class DecayLR:
    """
    The class create Decay learning rate. Unlike some other models,
    CycleGAN is trained from scratch, with a learning rate for some
    initial epochs and then linearly decaying this scale to zero overthe
    the next epochs.

    Parameters
    ----------
    epochs
        Number of epochs
    decay_epochs
        Number of decay epochs
    """
    def __init__(self, epochs, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay!"
        self.epochs = epochs
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        """
        Perform one step of learning rate decay.

        Parameters
        ----------
        epoch
            The current epoch.

        Returns
        -------
        lr
            The updated Learning rate.
        """
        return 1 - max(0, epoch - self.decay_epochs)/(self.epochs - self.decay_epochs)

class ReplayBuffer:
    """
    ReplayBuffer for CycleGAN.
    """
    def __init__(self, size=50):
        assert (size > 0), "Empty buffer"
        self.size = size
        self.data = []

    def push_and_pop(self, data):
        """
        Push and pop.
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.size -1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class ResidualBlock(nn.Module):
    """
    Residual Blocks are skip-connection blocks that learn residual
    functions with reference to the layer inputs, instead of learning unreferenced functions.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,in_channels,3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,in_channels,3),
            nn.InstanceNorm2d(in_channels))
    def forward(self,noise):
        """
        The forward for Residual Block.
        """
        return self.res(noise) + noise

class WSConv2d(nn.Module):
    """
    Weighted Scale Convolution.
    """
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.scale = (gain/(input_channel*(kernel_size**2)))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,noise):
        """
        The forward for WSConv2d.
        """
        return self.conv(noise*self.scale) + self.bias.view(1, self.bias.shape[0],1,1)

class PixelNorm(nn.Module):
    """
    A pixel normalization is used to normalize activation maps of
    the feature vector in each pixel to unit length and is
    applied after the convolutional layers in the generator.
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self,x_in):
        """
        The forward for PixelNorm.
        """
        return x_in / torch.sqrt(torch.mean(x_in**2, dim=1, keepdim=True) + self.epsilon)

class CNNBlock(nn.Module):
    """
     A class create CNN Block.
    """
    def __init__(self, input_channel, output_channel, pixel_norm=True):
        super().__init__()
        self.conv1 = WSConv2d(input_channel, output_channel)
        self.conv2 = WSConv2d(output_channel, output_channel)
        self.leaky = nn.LeakyReLU(0.2)
        self.pi_norm = PixelNorm()
        self.use_pn = pixel_norm

    def forward(self,noise):
        """
        The forward for CNNBlock.
        """
        noise = self.leaky(self.conv1(noise))
        noise = self.pi_norm(noise) if self.use_pn else noise
        noise = self.leaky(self.conv2(noise))
        noise = self.pi_norm(noise) if self.use_pn else noise
        return noise

def gradient_penalty(net_dis, real, fake, device="cuda"):
    """
    Penalty function for gradient. The gradient penalty (GP)
    is a tool that aids in the stabilization of training.
    The term in the gradient penalty refers to a tensor of
    uniformly generated random values between 0 and 1.
    The foregoing losses are normally averaged across
    the minibatch because we usually train in batches.
    """
    batch_size, channel, height, weight = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, channel, height, weight).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = net_dis(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    grad_penalty = torch.mean((gradient_norm - 1) ** 2)
    return grad_penalty

def conv_block(c_in, c_out, kernel_size=4, stride=2, padding=1, use_bn=True, transpose=False):
    """
    Function to create a Conv Block
    """
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, kernel_size,
                                            stride, padding, bias= not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)
