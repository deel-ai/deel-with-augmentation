"""
Tests for AdaIN.
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from augmentare.methods.style_transfer.adain import ADAIN
from ...utils import load_images

def test_train_output():
    """
    A test function for type and shape of output when training AdaIn.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [3 , 6]
    channel = [3 , 3]
    height, width = (256 , 256)
    learning_rate = 0.0002

    for i in range(2):
        model = ADAIN(device=device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        optimizer = Adam(model.parameters(), lr=learning_rate)

        dataset = []
        for j in range(len(content_images[0])):
            pair = (content_images[0][j], style_images[0][j])
            dataset.append(pair)

        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        loss_train = model.train_network(
            num_epochs=2,
            train_loader= train_loader,
            optimizer= optimizer,
            alpha=1.0,
            lamb = 10
        )

        assert isinstance(loss_train, list)
        assert isinstance(loss_train[0], float)

def test_adain_generate_output():
    """
    A test function for type and shape of output of AdaIn generate.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = [3 , 6]
    channel = [3 , 3]
    height, width = (256 , 256)

    for i in range(2):
        model = ADAIN(device)

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        gen_image = model.adain_generate(content_images[0],
                                            style_images[0], alpha=1.0)
        assert torch.is_tensor(gen_image) is True
        assert gen_image.shape == (batch_size[i], channel[i], height, width)
        