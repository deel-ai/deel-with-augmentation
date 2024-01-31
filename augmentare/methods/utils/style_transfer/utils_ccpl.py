"""
These utils are implemented for CCPL.
"""

import torch

def adjust_learning_rate(optimizer, iteration_count, learning_rate, lr_decay):
    """
    Imitating the original implementation.
    """
    new_lr = learning_rate / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def calc_mean_std(feat, eps=1e-5):
    """
    A function that calculates mean and std.

    Parameters
    ----------
    feat
        Shape of features = [batch_size, channels, height, width]
    eps
        Eps is a small value added to the variance to avoid divide-by-zero
    """
    size = feat.size()
    assert len(size) == 4
    n_feat, c_feat = size[:2]
    feat_var = feat.view(n_feat, c_feat, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n_feat, c_feat, 1, 1)
    feat_mean = feat.view(n_feat, c_feat, -1).mean(dim=2).view(n_feat, c_feat, 1, 1)
    return feat_mean, feat_std

def calc_mean(feat):
    """
    A function to calculate the mean of the feature.
    """
    size = feat.size()
    assert len(size) == 4
    n_feat, c_feat = size[:2]
    feat_mean = feat.view(n_feat, c_feat, -1).mean(dim=2).view(n_feat, c_feat, 1, 1)
    return feat_mean

def nor_mean_std(feat):
    """
    A function to normalize the mean and standard deviation of the feature.
    """
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat

def nor_mean(feat):
    """
    A function to normalize the mean of the feature.
    """
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean

def calc_cov(feat):
    """
    A function to calculate the covariance of the feature.
    """
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0,2,1)).div(feat.size(2))
    return f_cov

def _calc_feat_flatten_mean_std(feat):
    """
    A function to calculate the mean and standard deviation of the feature after flattening.
    It takes 3D feat (C, H, W), return mean and std of array within channels.
    """
    assert feat.size()[0] == 3
    assert isinstance(feat, torch.FloatTensor)
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def _mat_sqrt(x_in):
    """
    A function to get matrix sqrt.
    """
    u_component, d_component, v_component = torch.svd(x_in)
    return torch.mm(torch.mm(u_component, d_component.pow(0.5).diag()), v_component.t())

def coral(source, target):
    """
    Assume both source and target are 3D array (C, H, W)
    Note: flatten -> f
    """
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)
    return source_f_transfer.view(source.size())
    