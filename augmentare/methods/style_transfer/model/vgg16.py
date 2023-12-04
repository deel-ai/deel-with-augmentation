"""
Pretrained VGG16.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class Vgg16Pretrained(nn.Module):
    """
    A Pretrained VGG16 class for extracting features.
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features

        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for img in range(1):
            self.slice1.add_module(str(img), vgg_pretrained_features[img])
        for img in range(1, 9):
            self.slice2.add_module(str(img), vgg_pretrained_features[img])
        for img in range(9, 16):
            self.slice3.add_module(str(img), vgg_pretrained_features[img])
        for img in range(16, 23):
            self.slice4.add_module(str(img), vgg_pretrained_features[img])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x_in, inds=None, concat=True):
        """
        A forward function for Vgg16Pretrained.
        """
        if inds is None:
            inds = [1, 3, 6, 8, 11, 13, 15, 22, 29]
        x_clone = x_in.clone()  # prevent accidentally modifying input in place
        # Preprocess input according to original ImageNet training
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            x_clone[:, i:(i + 1), :, :] = (x_clone[:, i:(i + 1), :, :] - mean[i]) / std[i]

        # Get hidden state at layers specified by 'inds'
        l_2 = []
        if -1 in inds:
            l_2.append(x_in)

        # Only need to run network until we get to the max depth we want outputs from
        for i in range(max(inds) + 1):
            x_clone = self.vgg_layers[i].forward(x_clone)
            if i in inds:
                l_2.append(x_clone)

        # Concatenate hidden states if desired (after upsampling to spatial size of largest output)
        if concat:
            if len(l_2) > 1:
                zi_list = []
                max_h = l_2[0].size(2)
                max_w = l_2[0].size(3)
                for z_i in l_2:
                    if len(zi_list) == 0:
                        zi_list.append(z_i)
                    else:
                        zi_list.append(F.interpolate(z_i, (max_h, max_w), mode='bilinear'))

                output = torch.cat(zi_list, 1)
            else:  # don't bother doing anything if only returning one hidden state
                output = l_2[0]
        else:  # Otherwise return list of hidden states
            output = l_2
        return output
