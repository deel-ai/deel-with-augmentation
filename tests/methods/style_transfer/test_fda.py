"""
Tests for FDA.
"""

import torch
from augmentare.methods.style_transfer.fda import FDA
from ...utils import load_images

def test_type_and_shape_output():
    """
    A test function for type and shape of output when using FDA.
    """
    batch_size = [1 , 1]
    channel = [3 , 3]
    height, width = (512 , 1024)

    for i in range(2):

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        model = FDA(content_images[0], style_images[0])

        output = model.fda_source_to_target(beta=0.00001)

        assert torch.is_tensor(output) is True
        assert output.shape == (batch_size[i], channel[i], height, width)

def test_type_and_shape_output_another_way():
    """
    A test function for type and shape of output when using FDA.
    """
    batch_size = [1 , 1]
    channel = [3 , 3]
    height, width = (512 , 1024)

    for i in range(2):

        content_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        style_images = load_images(num_samples=batch_size[i], num_classes=1,
                                        num_channels=channel[i], height=height, width=width)

        model = FDA(content_images[0], style_images[0])

        output = model.fda_source_to_target_2(beta=0.00001)

        assert torch.is_tensor(output) is True
        assert output.shape == (batch_size[i], channel[i], height, width)
