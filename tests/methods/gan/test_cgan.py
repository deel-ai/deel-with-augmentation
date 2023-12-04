"""
This tests implements CGAN
"""

import numpy as np
import torch
from torch import nn
from torch import optim
from augmentare.methods.gan.cgan import CGAN, CGANGenerator, CGANDiscriminator
from ...utils import load_images

def test_generator_output_shape():
    """
    A test function for output shape of generator.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32, 32)
    latent_vec = [50 , 100]
    num_classes = [5 , 10]
    for i in range(2):
        image_shape = (channel[i], height, width)
        net_gen = CGANGenerator(latent_vec[i], num_classes[i], image_shape)
        noise = torch.ones(batch_size[i], latent_vec[i])
        labels = torch.LongTensor(np.random.randint(0, num_classes[i], batch_size[i]))
        output = net_gen(noise, labels)
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_discriminator_output_shape():
    """
    A test function for output shape of discriminator.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32, 32)
    num_classes = [5 , 10]
    for i in range(2):
        image_shape = (channel[i], height, width)
        net_dis = CGANDiscriminator(num_classes[i], image_shape)
        images = torch.ones(batch_size[i], channel[i], height, width)
        labels = torch.LongTensor(np.random.randint(0, num_classes[i], batch_size[i]))
        output = net_dis(images, labels)
        assert output.shape == (batch_size[i], 1)

def test_train():
    """
    A test function for type of output of CGAN.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32, 32)
    latent_vec = [50 , 100]
    num_classes = [5 , 10]
    learning_rate = 0.0002
    beta = 0.5

    for i in range(2):
        real_batch = load_images(batch_size[i], num_classes[i], channel[i], height, width)

        image_shape = (channel[i], height, width)
        net_gen = CGANGenerator(latent_vec[i], num_classes[i], image_shape)
        net_dis = CGANDiscriminator(num_classes[i], image_shape)
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
        optimizer_dis = optim.Adam(net_dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        loss_fn_gen =  nn.BCELoss()
        loss_fn_dis =  nn.BCELoss()

        # Test train
        gan = CGAN(net_gen, net_dis, optimizer_gen, optimizer_dis,
                    loss_fn_gen, loss_fn_dis, 'cpu', latent_vec[i])

        gen_loss = gan.train_generator(num_classes[i], batch_size[i])
        dis_loss = gan.train_discriminator(real_batch, num_classes[i], batch_size[i])
        assert isinstance(gen_loss.item(), float)
        assert isinstance(dis_loss.item(), float)

        img_list = gan.generate_samples(64, num_classes[i])
        assert torch.is_tensor(img_list) is True
        assert img_list.shape == (64, channel[i], height, width)
