"""
These utils are implemented for Style Flow.
"""

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_mean_std(features, eps=1e-5):
    """
    A function that calculates mean and std.

    Parameters
    ----------
    features
        Shape of features = [batch_size, channels, height, width]
    eps
        Eps is a small value added to the variance to avoid divide-by-zero
    """
    size = features.size()
    assert len(size)==4
    batch_size, channels = size[:2]

    features_mean = features.view(batch_size,
                    channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)

    features_std = features.view(batch_size,
                    channels, -1).var(dim=2).sqrt().view(batch_size,
                                                        channels, 1, 1) + eps
    return features_mean, features_std

def weighted_mse_loss(input_mean, target_mean, input_std, target_std, keep_ratio):
    """
    A function to get weighted MSE loss.
    """
    loss_mean = (input_mean - target_mean) ** 2
    sort_loss_mean,idx = torch.sort(loss_mean,dim=1)
    sort_loss_mean[:,int(sort_loss_mean.shape[1]*keep_ratio):] = 0

    loss_std = (input_std - target_std) ** 2
    loss_std[:,idx[:,int(idx.shape[1]*keep_ratio):]] = 0
    return sort_loss_mean.mean(),loss_std.mean()

def gram_matrix(x_input):
    """
    A fucntion to calculate gram matrix.
    """
    # a = batch size(=1)
    # b = number of feature maps
    # (c,d) = dimensions of a f. map (N = c*d)
    a_dim, b_dim, c_dim, d_dim = x_input.size()

    features = x_input.view(a_dim * b_dim, c_dim * d_dim)  # Resise F_XL into \hat F_XL

    gram = torch.mm(features, features.t())  # Compute the gram product

    # We normalize the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gram.div(a_dim * b_dim * c_dim * d_dim)

def get_smooth(x_input, direction):
    """
    A function to get smooth for calculating the gradients of loss.
    """
    weights = torch.tensor([[0., 0.],
                            [-1., 1.]]
                            ).to(DEVICE)
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 0, 1)
    if direction == 'x':
        weights = weights_x
    elif direction == 'y':
        weights = weights_y

    output = torch.abs(F.conv2d(x_input, weights, stride=1, padding=1))  # Stride, Padding
    return output

def avg(x_input, direction):
    """
    A function to apply average pooling.
    """
    return nn.AvgPool2d(kernel_size=3, stride=1,
                        padding=1)(get_smooth(x_input, direction))

def tv_loss(x_in, loss_weight=1):
    """
    A function to calculate the TV loss.
    """
    batch_size = x_in.size()[0]
    h_x = x_in.size()[2]
    w_x = x_in.size()[3]

    h_var = x_in[:,:,1:,:]
    count_h = h_var.size()[1] * h_var.size()[2] * h_var.size()[3]

    w_var = x_in[:,:,:,1:]
    count_w = w_var.size()[1] * w_var.size()[2] * w_var.size()[3]

    h_tv = torch.pow((x_in[:,:,1:,:]-x_in[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x_in[:,:,:,1:]-x_in[:,:,:,:w_x-1]),2).sum()
    return loss_weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size

def gradients_loss(stylized, target):
    """
    A fucntion to calculate the gradients of loss.
    """
    target_gray = torch.mean(target, dim=1, keepdim=True)
    stylized_gray = torch.mean(stylized, dim=1, keepdim=True)
    gradients_stylized_x = get_smooth(stylized_gray,'x')
    gradients_stylized_y = get_smooth(stylized_gray,'y')

    return torch.mean(gradients_stylized_x * torch.exp(-10 * avg(target_gray, 'x'))\
                        + gradients_stylized_y * torch.exp(-10 * avg(target_gray, 'y')))

def set_random_seed(seed):
    """
    A function to set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def adain(content, style_mean, style_std):
    """
    A forward function for ADAIN.

    Parameters
    ----------
    content
        Content input to calulate AdaIn
    style_mean
        Mean of style
    style_std
        Std of style
    """
    assert style_mean is not None
    assert style_std is not None

    size = content.size()
    content_mean, content_std = calc_mean_std(content)

    style_mean = style_mean.reshape(size[0],content_mean.shape[1],1,1)
    style_std = style_std.reshape(size[0],content_mean.shape[1],1,1)

    normalized_features = (content - content_mean.expand(size)) / content_std.expand(size)
    sum_mean = style_mean.expand(size)
    sum_std = style_std.expand(size)
    return normalized_features*sum_std + sum_mean
