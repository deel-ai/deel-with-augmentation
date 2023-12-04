"""
This tests implements DCGAN
"""

import torch
from torch import nn
from torch import optim
from augmentare.methods.gan.dcgan import DCGANGenerator, DCGANDiscriminator, DCGAN
from ...utils import load_images

def test_generator_output_shape():
    """
    A test function for output shape of generator.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (64, 64)
    latent_vec = [50 , 100]
    fea_map_gen = [32 , 64]
    for i in range(2):
        net_gen = DCGANGenerator(channel[i], latent_vec[i], fea_map_gen[i])
        noise = torch.ones(batch_size[i], latent_vec[i], 1, 1)
        output = net_gen(noise)
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_discriminator_output_shape():
    """
    A test function for output shape of discriminator.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (64, 64)
    fea_map_dis = [32 , 64]
    for i in range(2):
        net_dis = DCGANDiscriminator(channel[i], fea_map_dis[i])
        images = torch.ones(batch_size[i], channel[i], height, width)
        output = net_dis(images)
        assert output.shape == (batch_size[i], 1, 1, 1)

def test_train():
    """
    A test function for type of output of DCGAN.
    """
    batch_size = [32 , 64]
    channel = [1 , 3]
    height, width = (64 , 64)
    num_classes = [5 , 10]
    latent_vec = [50 , 100]
    fea_map_gen = [32 , 64]
    fea_map_dis = [32 , 64]
    learning_rate = 0.0002
    beta = 0.5

    for i in range(2):
        real_batch = load_images(batch_size[i], num_classes[i], channel[i], height, width)

        net_gen = DCGANGenerator(channel[i], latent_vec[i], fea_map_gen[i])
        net_dis = DCGANDiscriminator(channel[i], fea_map_dis[i])
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
        optimizer_dis = optim.Adam(net_dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        loss_fn_gen =  nn.BCELoss()
        loss_fn_dis =  nn.BCELoss()

        # Test train
        gan = DCGAN(net_gen, net_dis, optimizer_gen, optimizer_dis,
                    loss_fn_gen, loss_fn_dis, 'cpu', latent_vec[i])

        gen_loss = gan.train_generator(real_batch)
        dis_loss = gan.train_discriminator(real_batch)
        assert isinstance(gen_loss.item(), float)
        assert isinstance(dis_loss.item(), float)

        img_list = gan.generate_samples(64)
        assert torch.is_tensor(img_list) is True
        assert img_list.shape == (64, channel[i], height, width)
