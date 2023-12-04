"""
This tests implements CDCGAN
"""

import numpy as np
import torch
from torch import nn
from torch import optim
from augmentare.methods.gan.cdcgan import CDCGAN, CDCGANGenerator, CDCGANDiscriminator
from ...utils import load_images

def test_generator_output_shape():
    """
    A test function for output shape of generator.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32 , 32)
    latent_vec = [50 , 100]
    num_classes = [5 , 10]
    label_embed_size = [5 , 10]
    conv_dim = [64 , 128]
    for i in range(2):
        net_gen = CDCGANGenerator(num_classes[i], latent_vec[i],
                                    label_embed_size[i], channel[i], conv_dim[i])
        noise = torch.ones(batch_size[i], latent_vec[i])
        labels = torch.LongTensor(np.random.randint(0, num_classes[i], batch_size[i]))
        output = net_gen(noise, labels)
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_discriminator_output_shape():
    """
    A test function for output shape of discriminator.

    image_size = height = width
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32 , 32)
    num_classes = [5 , 10]
    conv_dim = [64 , 128]
    image_size = 32
    for i in range(2):
        net_dis = CDCGANDiscriminator(num_classes[i], channel[i], conv_dim[i], image_size)
        images = torch.ones(batch_size[i], channel[i], height, width)
        labels = torch.LongTensor(np.random.randint(0, num_classes[i], batch_size[i]))
        output = net_dis(images, labels)
        assert output.shape == (batch_size[i],)

def test_train():
    """
    A test function for type of output of CGAN.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (32 , 32)
    latent_vec = [50 , 100]
    num_classes = [5 , 10]
    label_embed_size = [5 , 10]
    conv_dim = [64 , 128]
    image_size = 32
    learning_rate = 0.0002
    beta = 0.5
    for i in range(2):
        real_batch, real_labels = load_images(batch_size[i], num_classes[i],
                                                channel[i], height, width)

        net_gen = CDCGANGenerator(num_classes[i], latent_vec[i],
                                    label_embed_size[i], channel[i], conv_dim[i])
        net_dis = CDCGANDiscriminator(num_classes[i], channel[i], conv_dim[i], image_size)
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
        optimizer_dis = optim.Adam(net_dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        loss_fn_gen =  nn.BCELoss()
        loss_fn_dis =  nn.BCELoss()

        # Test train
        gan = CDCGAN(net_gen, net_dis, optimizer_gen, optimizer_dis,
                    loss_fn_gen, loss_fn_dis, 'cpu', latent_vec[i])

        gen_loss = gan.train_generator(real_batch, real_labels, batch_size[i])
        dis_loss = gan.train_discriminator(real_batch, real_labels, batch_size[i])
        assert isinstance(gen_loss.item(), float)
        assert isinstance(dis_loss.item(), float)

        img_list = gan.generate_samples(5, num_classes[i])
        assert torch.is_tensor(img_list) is True
        assert img_list.shape == (5*num_classes[i], channel[i], height, width)
