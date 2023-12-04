"""
This tests implements CYCLEGAN
"""

import torch
from torch import nn
from torch import optim
from augmentare.methods.gan.cyclegan import CYCLEGAN, CYCLEGANGenerator, CYCLEGANDiscriminator
from ...utils import load_images

def test_generator_output_shape():
    """
    A test function for output shape of generator.
    """
    batch_size = [4 , 8]
    channel = [3 , 3]
    height, width = (8, 8)
    for i in range(2):
        net_gen = CYCLEGANGenerator()
        images = torch.ones(batch_size[i], channel[i], height, width)
        output = net_gen(images)
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_discriminator_output_shape():
    """
    A test function for output shape of discriminator.
    """
    batch_size = [4 , 8]
    channel = [3 , 3]
    height, width = (32, 32)
    for i in range(2):
        net_dis = CYCLEGANDiscriminator()
        images = torch.ones(batch_size[i], channel[i], height, width)
        output = net_dis(images)
        assert output.shape == (batch_size[i], 1)

def test_train():
    """
    A test function for type of output of CGAN.
    """
    batch_size = [4 , 8]
    channel = [3 , 3]
    height, width = (32, 32)
    num_classes = [5 , 10]
    learning_rate = 0.0002
    beta = 0.5
    for i in range(2):
        real_batch_a = load_images(batch_size[i], num_classes[i], channel[i], height, width)
        real_batch_b = load_images(batch_size[i], num_classes[i], channel[i], height, width)

        net_gen = CYCLEGANGenerator()
        net_dis = CYCLEGANDiscriminator()
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
        optimizer_dis = optim.Adam(net_dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        loss_fn_gen =  nn.L1Loss()
        loss_fn_dis =  nn.L1Loss()

        # Test train
        gan = CYCLEGAN(net_gen, net_dis, optimizer_gen, optimizer_dis,
                    loss_fn_gen, loss_fn_dis, 'cpu')

        gen_loss = gan.train_generator(real_batch_a[0], real_batch_b[0])
        dis_loss = gan.train_discriminator(real_batch_a[0], real_batch_b[0])
        assert isinstance(gen_loss.item(), float)
        assert isinstance(dis_loss.item(), float)

        fake_image_a, fake_image_b = gan.generate_samples(None, None,
                                                          real_batch_a[0], real_batch_b[0])
        assert torch.is_tensor(fake_image_a) is True
        assert fake_image_a.shape == real_batch_b[0].shape
        assert torch.is_tensor(fake_image_b) is True
        assert fake_image_b.shape == real_batch_a[0].shape
