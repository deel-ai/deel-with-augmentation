"""
Pretrained VGG19.
"""

from torch import nn
from torchvision import models

class Vgg19Pretrained(nn.Module):
    """
    A Pretrained VGG19 class for extracting features.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for para in self.parameters():
            para.requires_grad = False

    def forward(self, images, output_last_feature=False):
        """
        A forward function VggEncoder.
        """
        out1 = self.slice1(images)
        out2 = self.slice2(out1)
        out3 = self.slice3(out2)
        out4 = self.slice4(out3)
        if output_last_feature:
            return out4
        return out1, out2, out3, out4
    