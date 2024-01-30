"""
Tests for CCPL.
"""

import torch
from torch.utils.data import DataLoader
from augmentare.methods.style_transfer.ccpl import CCPL
from ...utils import load_images

def test_train_output():
    """
    A test function for type an shape of output when training CCPL.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [200 , 400]
    channel = [3 , 3]
    height, width = (256 , 256)
    vgg_path = None
    for i in range(2):
        model = CCPL(training_mode= "art", vgg_path=vgg_path, device=device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        content_set = []
        style_set = []
        for j in range(len(content_images[0])):
            content_set.append(content_images[0][j])
            style_set.append(style_images[0][j])

        content_loader = DataLoader(content_set, batch_size=8)

        style_loader = DataLoader(style_set, batch_size=8)

        losses = model.train_network(content_loader, style_loader, num_s=8, num_l=3,
                        max_iter=10, content_weight=1.0, style_weight=10.0, ccp_weight=5.0
        )

        assert isinstance(losses, list)
        assert isinstance(losses[0], float)

def test_ccpl_generate_output():
    """
    A test function for type an shape of generated image by CCPL.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [10 , 20]
    channel = [3 , 3]
    height, width = (256 , 256)
    vgg_path = ('/home/vuong.nguyen/vuong/augmentare/augmentare/'
                'methods/style_transfer/model/vgg_normalised_ccpl.pth')

    for i in range(2):
        model = CCPL(training_mode= "art", vgg_path=vgg_path, device=device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        content_set = torch.stack([content_images[0][i] for i in range(3)])
        style_set = torch.stack([style_images[0][i] for i in range(3)])

        torch.cuda.empty_cache()

        interpolation = True
        if interpolation:
            gen_image = model.ccpl_generate(content_set, style_set, alpha=1.0,
                            interpolation= interpolation)

            assert torch.is_tensor(gen_image) is True
            assert gen_image.shape == (len(style_set), channel[i], height, width)

        else:
            gen_image = model.ccpl_generate(content_set, style_set, alpha=1.0,
                        interpolation= interpolation, preserve_color=False)

            assert torch.is_tensor(gen_image) is True
            assert gen_image.shape == (1, channel[i], height, width)
