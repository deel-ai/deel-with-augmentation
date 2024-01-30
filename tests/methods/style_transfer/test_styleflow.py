"""
Tests for Style Flow.
"""

import torch
from torch.utils.data import DataLoader
from augmentare.methods.style_transfer.style_flow import STYLEFLOW
from ...utils import load_images

def test_train_output():
    """
    A test function for type an shape of output when training Style Flow.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [3 , 6]
    channel = [3 , 3]
    height, width = (256 , 256)
    vgg_path = None
    for i in range(2):
        model = STYLEFLOW(in_channel=3, n_flow=15, n_block=2, vgg_path = vgg_path,
                            affine=False, conv_lu=False, keep_ratio=0.8, device=device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        dataset = []
        for j in range(len(content_images[0])):
            pair = (content_images[0][j], style_images[0][j])
            dataset.append(pair)

        train_loader = DataLoader(dataset, batch_size=1, num_workers=8)

        losses = model.train_network(train_loader=train_loader,
            content_weight = 0.1, style_weight=1, type_loss="TVLoss"
        )

        assert isinstance(losses, list)
        assert isinstance(losses[0], float)

def test_style_flow_generate_output():
    """
    A test function for type an shape of generated image by Style Flow.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [1 , 1]
    channel = [3 , 3]
    height, width = (128 , 256)
    vgg_path = None
    for i in range(2):
        model = STYLEFLOW(in_channel=3, n_flow=15, n_block=2, vgg_path = vgg_path,
                            affine=False, conv_lu=False, keep_ratio=0.8, device=device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)
        torch.cuda.empty_cache()
        gen_image = model.style_flow_generate(content_images[0], style_images[0])

        assert torch.is_tensor(gen_image) is True
        assert gen_image.shape == (batch_size[i], channel[i], height, width)
