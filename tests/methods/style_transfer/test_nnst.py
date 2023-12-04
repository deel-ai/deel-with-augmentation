"""
Test for NNST.
"""

import torch
from augmentare.methods.style_transfer.nnst import NNST, Vgg16Pretrained
from ...utils import load_images

def test_stylization_output():
    """
    A test function for type and shape of output when styling.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [1 , 1]
    channel = [3 , 3]
    height, width = (256 , 512)

    cnn = Vgg16Pretrained()
    cnn.to(device)
    def phi(x_in, y_in, z_in):
        return cnn.forward(x_in, inds=y_in, concat=z_in)

    for i in range(2):

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)
        model = NNST(content_images[0], style_images[0], device=device)

        stylized_im = model.stylization(
            phi,
            max_iter= 200,
            l_rate=2e-3,
            style_weight=1.,
            max_scls = 4,
            flip_aug = False,
            content_loss = False,
            zero_init = False,
            dont_colorize = False
        )

        assert torch.is_tensor(stylized_im) is True
        assert stylized_im.shape == (batch_size[i], channel[i], height, width)

def test_nnst_generate_output():
    """
    A test function for type of output of NNST generate.
    It also checks the shape of the generated image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [1 , 1]
    channel = [3 , 3]
    height, width = (256 , 512)

    for i in range(2):

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)
        model = NNST(content_images[0], style_images[0], device=device)

        gen_image = model.nnst_generate(
            max_scales=5, alpha=0.75,
            content_loss=False, flip_aug=False,
            zero_init = False, dont_colorize=False
        )

        assert torch.is_tensor(gen_image) is True
        assert gen_image.shape == (batch_size[i], channel[i], height, width)
