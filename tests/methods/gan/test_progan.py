"""
This tests implements PROGAN
"""

from math import log2
import torch
from torch import optim
from augmentare.methods.gan.progan import PROGAN, PROGANGenerator, PROGANDiscriminator
from ...utils import load_images

def test_generator_output_shape():
    """
    A test function for output shape of generator.
    """
    alpha = 1e-5
    start_image_size = 64
    step = int(log2(start_image_size/4))
    latent_vec = [16, 32]
    batch_size = [4 , 8]
    channel = [1 , 3]
    height, width = (64, 64)
    for i in range(2):
        net_gen = PROGANGenerator(latent_vec[i], height, channel[i], alpha, step)
        noise = torch.ones(batch_size[i], latent_vec[i], 1, 1)
        output = net_gen(noise)
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_discriminator_output_shape():
    """
    A test function for output shape of discriminator.
    """
    alpha = 1e-5
    start_image_size = 64
    step = int(log2(start_image_size/4))

    batch_size = [4 , 8]
    channel = [1 , 3]
    height, width = (64, 64)
    for i in range(2):
        net_dis = PROGANDiscriminator(height, channel[i], alpha, step)
        images = torch.ones(batch_size[i], channel[i], height, width)
        output = net_dis(images)
        assert output.shape == (batch_size[i], 1)

def test_train():
    """
    A test function for type of output of PROGAN.
    """
    alpha = 1e-5
    start_image_size = 64
    step = int(log2(start_image_size/4))
    latent_vec = [16, 32]
    batch_size = [4 , 8]
    channel = [1 , 3]
    height, width = (64, 64)
    num_classes = [5 , 10]
    learning_rate = 0.0002
    beta = 0.5
    for i in range(2):
        real_batch = load_images(batch_size[i], num_classes[i], channel[i], height, width)
        #noise = torch.ones(batch_size[i], latent_vec[i], 1, 1).to('cuda')  (if device is cuda)
        noise = torch.ones(batch_size[i], latent_vec[i], 1, 1)

        net_gen = PROGANGenerator(latent_vec[i], height, channel[i], alpha, step)
        net_dis = PROGANDiscriminator(height, channel[i], alpha, step)
        optimizer_gen = optim.Adam(net_gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
        optimizer_dis = optim.Adam(net_dis.parameters(), lr=learning_rate, betas=(beta, 0.999))
        loss_fn_gen =  torch.cuda.amp.GradScaler()
        loss_fn_dis =  torch.cuda.amp.GradScaler()

        # Test train
        gan = PROGAN(net_gen, net_dis, optimizer_gen, optimizer_dis,
                    loss_fn_gen, loss_fn_dis, 'cpu', latent_vec[i])

        gen_loss = gan.train_generator(noise)
        dis_loss = gan.train_discriminator(real_batch[0], noise)
        assert isinstance(gen_loss.item(), float)
        assert isinstance(dis_loss.item(), float)

        img_list = gan.generate_samples(2)
        assert torch.is_tensor(img_list) is True
        assert img_list.shape == (2, channel[i], height, width)
