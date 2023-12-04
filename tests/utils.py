import torch

def load_images(num_samples=100, num_classes=10, num_channels=3, height=32, width=32):
    images = torch.randn(num_samples, num_channels, height, width)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels
