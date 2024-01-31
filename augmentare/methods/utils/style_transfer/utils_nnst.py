"""
These utils are implemented for NNST.
"""

import random
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread
from torch.autograd import Variable

def dec_lap_pyr(x_in, levs):
    """
    A function to construct batch of 'levs' level laplacian pyramids from x_in.

    Parameters
    ----------
    x_in
        The input is the pytorch tensor in the form of BxCxHxW
    levs
        Integer number of pyramid levels to construct

    Returns
    -------
    pyr
        A list of pytorch tensors, each representing a pyramid level,
        pyr[0] contains the finest level, pyr[-1] the coarsest
    """
    pyr = []
    cur = x_in  # Initialize approx. coefficients with original image
    for _ in range(levs):
        # Construct and store detail coefficients from current approx. coefficients
        height = cur.size(2)
        width = cur.size(3)
        x_small = F.interpolate(cur, (height // 2, width // 2), mode='bilinear')
        x_back = F.interpolate(x_small, (height, width), mode='bilinear')
        lap = cur - x_back
        pyr.append(lap)

        # Store new approx. coefficients
        cur = x_small
    pyr.append(cur)
    return pyr

def syn_lap_pyr(pyr):
    """
    A function to collapse batch of laplacian pyramids stored in list of
    pytorch tensors 'pyr' into a single tensor.

    Parameters
    ----------
    pyr
        List of pytorch tensors, where pyr[i] has size BxCx(H/(2**i)x(W/(2**i))

    Returns
    -------
    output
        A BxCxHxW pytorch tensor
    """
    cur = pyr[-1]
    levs = len(pyr)

    for i in range(0, levs - 1)[::-1]:
        # Create new approximation coefficients from current approx. and detail coefficients
        # at next finest pyramid level
        up_x = pyr[i].size(2)
        up_y = pyr[i].size(3)
        cur = pyr[i] + F.interpolate(cur, (up_x, up_y), mode='bilinear')
    output = cur
    return output

# Global variable
USE_GPU = True

def to_device(tensor):
    """
    A function to ensure torch tensor 'tensor' is moved to gpu
    if global variable USE_GPU is True.
    """
    if USE_GPU:
        return tensor.cuda()
    return tensor

def match_device(ref, mut):
    """
    A function to put torch tensor 'mut' on the same device as torch tensor 'ref'.
    """
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

def get_gpu_memory_map():
    """
    A function to get the current gpu usage.
    Taken from: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage
        Keys are device ids as integers,
        and values are memory usage as integers in MB
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x_in) for x_in in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)

def flatten_grid(x_in):
    """
    A function to collapse spatial dimensions of pytorch tensor 'x_in' and transpose.

    Parameters
    ----------
    x_in
        1xCxHxW pytorch tensor

    Returns
    -------
    y_out
        (H*W)xC pytorch tensor
    """
    assert x_in.size(0) == 1, "undefined behavior for batched input"
    y_out = x_in.contiguous().view(x_in.size(1), -1).clone().transpose(1, 0)
    return y_out

def scl_spatial(x_in, height, width):
    """
    Shorter alias for default way I call F.interpolate (i.e. as bilinear interpolation).
    """
    return F.interpolate(x_in, (height, width), mode='bilinear', align_corners=True)

def load_path_for_pytorch(im_path, target_size=1000, side_comp=max, verbose=False):
    """
    Loads image at 'path', selects height or width with function 'side_comp'
    then scales the image, setting selected dimension to 'target_size' and
    maintaining aspect ratio. Will also convert RGBA or greyscale images to
    RGB.

    Returns
    -------
    output
        A HxWxC pytorch tensor of rgb values scaled between 0 and 1
    """
    # Load Image
    x_in = imread(im_path).astype(np.float32)

    # Converts image to rgb if greyscale
    if len(x_in.shape) < 3:
        x_in = np.stack([x_in, x_in, x_in], 2)

    # Removes alpha channel if present
    if x_in.shape[2] > 3:
        x_in = x_in[:, :, :3]

    # Rescale rgb values
    x_in = x_in / 255.

    # Convert from numpy
    x_dims = x_in.shape
    x_in = torch.from_numpy(x_in).contiguous().permute(2, 0, 1).contiguous()

    # Rescale to desired size
    # by default maintains aspect ratio relative to long side
    # change side_comp to be min for short side
    fac = float(target_size) / side_comp(x_dims[:2])
    height = int(x_dims[0] * fac)
    width = int(x_dims[1] * fac)
    output = scl_spatial(x_in.unsqueeze(0), height, width)[0]

    if verbose:
        print(f'DEBUG: image from path {im_path} loaded with size {x_dims}')

    return output

def get_feat_norms(x_in):
    """
    A function to make L2 norm of x[i,:,j,k] = 1 for all i,j,k. Clamps before sqrt for stability.
    """
    return torch.clamp(x_in.pow(2).sum(1, keepdim=True), 1e-8, 1e8).sqrt()


def phi_cat(x_in, phi, layer_l):
    """
    A function to extract conv features from 'x_in' at list of VGG16 layers 'layer_l'.
    Then normalize features from each conv block based on # of channels, resize,
    and concatenate into hypercolumns.

    Parameters
    ----------
    x_in
        Bx3xHxW pytorch tensor, presumed to contain rgb images
    phi
        Lambda function calling a pretrained Vgg16Pretrained model
    layer_l
        Layer indexes to form hypercolumns out of

    Returns
    -------
    feats
        BxCxHxW pytorch tensor of hypercolumns extracted from 'x_in' C depends on 'layer_l'
    """
    height = x_in.size(2)
    width = x_in.size(3)

    feats = phi(x_in, layer_l, False)
    # Normalize each layer by # channels so # of channels doesn't dominate
    # cosine distance
    feats = [fea / fea.size(1) for fea in feats]

    # Scale layers' features to target size and concatenate
    feats = torch.cat([scl_spatial(fea, height // 4, width // 4) for fea in feats], 1)
    return feats

def extract_feats(img, phi, flip_aug=False):
    """
    A function to extract hypercolumns from 'img' using pretrained VGG16 (passed as phi),
    if speficied, extract hypercolumns from rotations of 'im' as well.

    Parameters
    ----------
    img
        A Bx3xHxW pytorch tensor, presumed to contain rgb images
    phi
        A lambda function calling a pretrained Vgg16Pretrained model
    flip_aug
        Whether to extract hypercolumns from rotations of 'im' as well

    Returns
    -------
    feats
        A tensor of hypercolumns extracted from 'im', spatial index is presumed to no longer matter
    """
    # In the original paper used all layers, but dropping conv5 block increases
    # speed without harming quality
    layer_l = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]
    feats = phi_cat(img, phi, layer_l)

    # If specified, extract features from 90, 180, 270 degree rotations of 'im'
    if flip_aug:
        aug_list = [torch.flip(img, [2]).transpose(2, 3),
                    torch.flip(img, [2, 3]),
                    torch.flip(img, [3]).transpose(2, 3)]

        for i, im_aug in enumerate(aug_list):
            feats_new = phi_cat(im_aug, phi, layer_l)

            # Code never looks at patches of features, so fine to just stick
            # features from rotated images in adjacent spatial indexes, since
            # they will only be accessed in isolation
            if i == 1:
                feats = torch.cat([feats, feats_new], 2)
            else:
                feats = torch.cat([feats, feats_new.transpose(2, 3)], 2)
    return feats

def replace_features(src, ref):
    """
    A function to replace each feature vector in 'src' with the nearest (under centered
    cosine distance) feature vector in 'ref'.

    Parameters
    ----------
    src
        1xCxAxB tensor of content features
    ref
        1xCxHxW tensor of style features

    Returns
    -------
    rplc
        1xCxHxW tensor of features, where rplc[0,:,i,j] is the nearest
        neighbor feature vector of src[0,:,i,j] in ref
    """
    # Move style features to gpu (necessary to mostly store on cpu for gpus w/
    # < 12GB of memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_flat = flatten_grid(ref).to(device)
    rplc = []
    for j in range(src.size(0)):
        # How many rows of the distance matrix to compute at once, can be
        # reduced if less memory is available, but this slows method down
        stride = 128**2 // max(1, (ref.size(2) * ref.size(3)) // (128 ** 2))
        b_i = 0
        e_i = 0

        # Loop until all content features are replaced by style feature / all
        # rows of distance matrix are computed
        out = []
        src_flat_all = flatten_grid(src[j:j + 1, :, :, :])
        while b_i < src_flat_all.size(0):
            e_i = min(b_i + stride, src_flat_all.size(0))

            # Get chunck of content features, compute corresponding portion
            # distance matrix, and store nearest style feature to each content
            # feature
            src_flat = to_device(src_flat_all[b_i:e_i, :])
            d_mat = pairwise_distances_cos_center(ref_flat, src_flat)
            _, nn_inds = torch.min(d_mat, 0)
            del d_mat  # distance matrix uses lots of memory, free asap

            # Get style feature closest to each content feature and save
            # in 'out'
            nn_inds = nn_inds.unsqueeze(1).expand(nn_inds.size(0), ref_flat.size(1))
            ref_sel = torch.gather(ref_flat, 0, nn_inds).transpose(1,0).contiguous()
            out.append(ref_sel)#.view(1, ref.size(1), src.size(2), ei - bi))

            b_i = e_i
        out = torch.cat(out, 1)
        out = out.view(1, ref.size(1), src.size(2), src.size(3))
        rplc.append(out)

    rplc = torch.cat(rplc, 0)
    return rplc

def center(x_in):
    """
    Subtract the mean of 'x_in' over leading dimension.
    """
    return x_in - torch.mean(x_in, 0, keepdim=True)

def pairwise_distances_cos(x_in, y_in):
    """
    A function to compute all pairwise cosine distances
    between rows of matrix 'x_in' and matrix 'y_in'.

    Parameters
    ----------
    x_in
        NxD pytorch tensor
    y_in
        MxD pytorch tensor

    Returns
    -------
    dis
        NxM pytorch tensor where d[i,j] is the cosine distance between
        the vector at row i of matrix 'x_in' and the vector at row j of
        matrix 'y_in'
    """
    assert x_in.size(1) == y_in.size(1), "can only compute distance between vectors of same length"
    assert (len(x_in.size()) == 2) and (len(y_in.size()) == 2), "pairwise distance computation"\
                                                          " assumes input tensors are matrices"

    x_norm = torch.sqrt((x_in**2).sum(1).view(-1, 1))
    y_norm = torch.sqrt((y_in**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y_in / y_norm, 0, 1)

    dis = 1. - torch.mm(x_in / x_norm, y_t)
    return dis

def pairwise_distances_sq_l2(x_in, y_in):
    """
    A function to compute all pairwise squared l2 distances
    between rows of matrix 'x_in' and matrix 'y_in'.

    Parameters
    ----------
    x_in
        NxD pytorch tensor
    y_in
        MxD pytorch tensor

    Returns
    -------
    dis
        NxM pytorch tensor where d[i,j] is the squared l2 distance between
        the vector at row i of matrix 'x_in' and the vector at row j of
        matrix 'y_in'
    """
    assert x_in.size(1) == y_in.size(1), "can only compute distance between vectors of same length"
    assert (len(x_in.size()) == 2) and (len(y_in.size()) == 2), "pairwise distance computation"\
                                                          " assumes input tensors are matrices"

    x_norm = (x_in ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y_in, 0, 1)
    y_norm = (y_in ** 2).sum(1).view(1, -1)

    dis = -2.0 * torch.mm(x_in, y_t)
    dis += x_norm
    dis += y_norm
    return dis

def pairwise_distances_l2(x_in, y_in):
    """
    Compute all pairwise l2 distances between rows of 'x_in' and 'y_in',
    thresholds minimum of squared l2 distance for stability of sqrt.
    """
    dis = torch.clamp(pairwise_distances_sq_l2(x_in, y_in), min=1e-8)
    return torch.sqrt(dis)

def pairwise_distances_l2_center(x_in, y_in):
    """
    Subtracts mean row from 'x_in' and 'y_in' before
    computing pairwise l2 distance between all rows.
    """
    return pairwise_distances_l2(center(x_in), center(y_in))

def pairwise_distances_cos_center(x_in, y_in):
    """
    Subtracts mean row from 'x_in' and 'y_in' before
    computing pairwise cosine distance between all rows.
    """
    return pairwise_distances_cos(center(x_in), center(y_in))

def linear_2_oklab(x_in):
    """
    Converts pytorch tensor 'x_in' from Linear to OkLAB colorspace,
    described here: https://bottosson.github.io/posts/oklab/

    Parameters
    ----------
    x_in
        Pytorch tensor of size B x 3 x H x W, assumed to be in linear
        srgb colorspace, scaled between 0 and 1

    Returns
    -------
    y_out
        Pytorch tensor of size B x 3 x H x W in OkLAB colorspace
    """
    assert x_in.size(1) == 3, "attempted to convert colorspace of tensor w/ > 3 channels"

    x_in = torch.clamp(x_in, 0., 1.)

    r_color = x_in[:, 0:1, :, :]
    g_color = x_in[:, 1:2, :, :]
    b_color = x_in[:, 2:3, :, :]

    li_out = 0.4121656120 * r_color + 0.5362752080 * g_color + 0.0514575653 * b_color
    m_out = 0.2118591070 * r_color + 0.6807189584 * g_color + 0.1074065790 * b_color
    s_out = 0.0883097947 * r_color + 0.2818474174 * g_color + 0.6302613616 * b_color

    li_out = torch.pow(li_out, 1. / 3.)
    m_out = torch.pow(m_out, 1. / 3.)
    s_out = torch.pow(s_out, 1. / 3.)

    l_out = 0.2104542553 * li_out + 0.7936177850 * m_out - 0.0040720468 * s_out
    a_out = 1.9779984951 * li_out - 2.4285922050 * m_out + 0.4505937099 * s_out
    b_out = 0.0259040371 * li_out + 0.7827717662 * m_out - 0.8086757660 * s_out

    y_out = torch.cat([l_out, a_out, b_out], 1)
    return y_out

def oklab_2_linear(x_in):
    """
    Converts pytorch tensor 'x' from OkLAB to Linear colorspace,
    described here: https://bottosson.github.io/posts/oklab/

    Parameters
    ----------
    x_in
        Pytorch tensor of size B x 3 x H x W, assumed to be in OkLAB colorspace

    Returns
    -------
    y_out
        Pytorch tensor of size B x 3 x H x W in Linear sRGB colorspace
    """
    assert x_in.size(1) == 3, "attempted to convert colorspace of tensor w/ > 3 channels"

    l_out = x_in[:, 0:1, :, :]
    a_out = x_in[:, 1:2, :, :]
    b_out = x_in[:, 2:3, :, :]

    li_out = l_out + 0.3963377774 * a_out + 0.2158037573 * b_out
    m_out = l_out - 0.1055613458 * a_out - 0.0638541728 * b_out
    s_out = l_out - 0.0894841775 * a_out - 1.2914855480 * b_out

    li_out = torch.pow(li_out, 3)
    m_out = torch.pow(m_out, 3)
    s_out = torch.pow(s_out, 3)

    r_color = 4.0767245293 * li_out - 3.3072168827 * m_out + 0.2307590544 * s_out
    g_color = -1.2681437731 * li_out + 2.6093323231 * m_out - 0.3411344290 * s_out
    b_color = -0.0041119885 * li_out - 0.7034763098 * m_out + 1.7068625689 * s_out

    y_out = torch.cat([r_color, g_color, b_color], 1)
    return torch.clamp(y_out, 0., 1.)

def get_pad(x_in):
    """
    Applies 1 pixel of replication padding to x_in.

    Parameters
    ----------
    x_in
        B x D x H x W pytorch tensor
    """
    return F.pad(x_in, (1,1,1,1), mode='replicate')

def filter_func(x_in):
    """
    Applies modified bilateral filter to AB channels of x_in, guided by L channel.

    Parameters
    ----------
    x_in
        B x 3 x H x W pytorch tensor containing an image in LAB colorspace
    """
    height = x_in.size(2)
    width = x_in.size(3)

    # Seperate out luminance channel, don't use AB channels to measure similarity
    x_l = x_in[:,:1,:,:]
    xab = x_in[:,1:,:,:]
    xl_pad = get_pad(x_l)

    xl_w = {}
    for i in range(3):
        for j in range(3):
            xl_w[str(i) + str(j)] =  xl_pad[:, :, i:(i+height), j:(j+width)]

    # Iteratively apply in 3x3 window rather than use spatial kernel
    max_iters = 5
    cur = torch.zeros_like(xab)

    # comparison function for pixel intensity
    def comp(x_in, y_in):
        dis = torch.abs(x_in - y_in) * 5.
        return torch.pow(torch.exp(-1. * dis),2)

    # apply bilateral filtering to AB channels, guideded by L channel
    cur = xab.clone()
    for _ in range(max_iters):
        cur_pad = get_pad(cur)
        xl_v = {}
        for i in range(3):
            for j in range(3):
                xl_v[str(i) + str(j)] = cur_pad[:, :, i:(i+height), j:(j+width)]

        denom = torch.zeros_like(x_l)
        cur = cur * 0.

        for i in range(3):
            for j in range(3):
                scl = comp(x_l, xl_w[str(i) + str(j)])
                cur = cur + xl_v[str(i) + str(j)] * scl
                denom = denom + scl

        cur = cur / denom
    # store result and return
    x_in[:, 1:, :, :] = cur
    return x_in

def clamp_range(x_in, y_in):
    """
    Clamp the range of x_in to [min(y_in), max(y_in)]

    Parameters
    ----------
    x
        Pytorch tensor
    y
        Pytorch tensor
    """
    return torch.clamp(x_in, y_in.min(), y_in.max())

def color_match(content_img, style_img, output_img, moment_only=False):
    """
    Constrain the low frequences of the AB channels of output image (containing hue and saturation)
    to be an affine transformation of 'c' matching the mean and covariance of the style image 's'.
    Compared to the raw output of optimization this is highly constrained, but in practice
    we find the benefit to robustness to be worth the reduced stylization.

    Parameters
    ----------
    content_img
        B x 3 x H x W pytorch tensor containing content image
    style_img
        B x 3 x H x W pytorch tensor containing style image
    output_img
        B x 3 x H x W pytorch tensor containing initial output image
    moment_only
        Boolean, prevents applying bilateral filter to AB channels
        of final output to match luminance's edges
    """
    con = torch.clamp(content_img, 0., 1.)
    sty = torch.clamp(style_img, 0., 1.)
    out = torch.clamp(output_img, 0., 1.)

    x_in = linear_2_oklab(con)
    # x_flat = x_in.view(x_in.size(0), x_in.size(1), -1, 1)
    y_in = linear_2_oklab(sty)
    out = linear_2_oklab(out)

    x_new = out.clone()
    for i in range(3):
        x_new[:, i:i + 1,:,:] = clamp_range(x_new[:, i:i + 1,:,:], y_in[:, i:i + 1, :, :])

    _, cov_s = zca_tensor(x_new, y_in)

    if moment_only or cov_s[1:,1:].abs().max() < 6e-5:
        x_new[:,1:,:,:] = out[:,1:,:,:]
        x_new, _ = zca_tensor(x_new, y_in)
    else:
        x_new[:,1:,:,:] = x_in[:,1:,:,:]
        x_new[:,1:,:,:] = zca_tensor(x_new[:,1:,:,:], y_in[:,1:,:,:])[0]
        x_new = filter_func(x_new)

    for i in range(3):
        x_new[:,i:i+1,:,:] = clamp_range(x_new[:,i:i+1,:,:], y_in[:,i:i+1,:,:])

    # x_pyr = dec_lap_pyr(x_in,4)
    # y_pyr = dec_lap_pyr(y_in,4)
    x_new_pyr = dec_lap_pyr(x_new,4)
    o_pyr = dec_lap_pyr(out,4)
    x_new_pyr[:-1] = o_pyr[:-1]

    return oklab_2_linear(x_new)

def optimize_output_im(s_pyr, c_pyr, style_im, target_feats,
                       l_r, max_iter, scl, phi, final_pass=False,
                       content_loss=False, flip_aug=True):
    """
    Optimize laplacian pyramid coefficients of stylized image at a given
    resolution, and return stylized pyramid coefficients.

    Parameters
    ----------
    s_pyr
        Laplacian pyramid of style image
    c_pyr
        Laplacian pyramid of content image
    style_im
        Style image
    target_feats
        Precomputed target features of stylized output
    lr
        Learning rate for optimization
    max_iter
        Maximum number of optimization iterations
    scl
        Integer controls which resolution to optimize (corresponds
        to pyramid level of target resolution)
    phi
        Lambda function to compute features using pretrained VGG16
    final_pass
        If true, ignore 'target_feats' and recompute target features before every
        step of gradient descent (and compute feature matches seperately for each
        layer instead of using hypercolumns)
    content_loss
        If true, also minimize content loss that maintains self-similarity in color space
        between 32pixel downsampled output image and content image
    flip_aug
        If true, extract style features from rotations of style image. This increases
        content preservation by making more options available when matching style
        features to content features

    Returns
    -------
    s_pyr
        Pyramid coefficients of stylized output image at target resolution
    """
    # Initialize optimizer variables and optimizer
    output_im = syn_lap_pyr(s_pyr[scl:])
    opt_vars = [Variable(li.data, requires_grad=True) for li in s_pyr[scl:]]
    optimizer = torch.optim.Adam(opt_vars, lr=l_r)

    # Original features uses all layers, but dropping conv5 block  speeds up
    # method without hurting quality
    feature_list_final = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]

    # Precompute features that remain constant
    if not final_pass:
        # Precompute normalized features targets during hypercolumn-matching
        # regime for cosine distance
        target_feats_n = target_feats / get_feat_norms(target_feats)

    else:
        # For feature-splitting regime extract style features for each conv
        # layer without downsampling (including from rotations if applicable)
        s_feat = phi(style_im, feature_list_final, False)

        if flip_aug:
            aug_list = [torch.flip(style_im, [2]).transpose(2, 3),
                        torch.flip(style_im, [2, 3]),
                        torch.flip(style_im, [3]).transpose(2, 3)]

            for i_a, im_aug in enumerate(aug_list):
                s_feat_tmp = phi(im_aug, feature_list_final, False)

                if i_a != 1:
                    s_feat_tmp = [s_feat_tmp[iii].transpose(2, 3)
                                  for iii in range(len(s_feat_tmp))]

                s_feat = [torch.cat([s_feat[iii], s_feat_tmp[iii]], 2)
                          for iii in range(len(s_feat_tmp))]

    # Precompute content self-similarity matrix if needed for 'content_loss'
    if content_loss:
        c_full = syn_lap_pyr(c_pyr)
        c_scl = max(c_full.size(2), c_full.size(3))
        c_fac = c_scl // 32
        height = int(c_full.size(2) / c_fac)
        width = int(c_full.size(3) / c_fac)

        c_low_flat = flatten_grid(scl_spatial(c_full, height, width))
        self_sim_target = pairwise_distances_l2(c_low_flat, c_low_flat).clone().detach()


    # Optimize pyramid coefficients to find image that produces stylized activations
    for i in range(max_iter):

        # Zero out gradient and loss before current iteration
        optimizer.zero_grad()
        ell = 0.

        # Synthesize current output from pyramid coefficients
        output_im = syn_lap_pyr(opt_vars)


        # Compare current features with stylized activations
        if not final_pass:  # hypercolumn matching / 'hm' regime

            # Extract features from current output, normalize for cos distance
            cur_feats = extract_feats(output_im, phi)
            cur_feats_n = cur_feats / get_feat_norms(cur_feats)

            # Update overall loss w/ cosine loss w.r.t target features
            ell = ell + (1. - (target_feats_n * cur_feats_n).sum(1)).mean()


        else:  # feature splitting / 'fs' regime
            # Extract features from current output (keep each layer seperate
            # and don't downsample)
            cur_feats = phi(output_im, feature_list_final, False)

            # Compute matches for each layer. For efficiency don't explicitly
            # gather matches, only access through distance matrix.
            ell_fs = 0.
            for h_i, s_feat_i in enumerate(s_feat):
                # Get features from a particular layer
                s_tmp = s_feat_i
                cur_tmp = cur_feats[h_i]
                chans = s_tmp.size(1)

                # Sparsely sample feature tensors if too big, otherwise just
                # reshape
                if max(cur_tmp.size(2), cur_tmp.size(3)) > 64:
                    stride = max(cur_tmp.size(2), cur_tmp.size(3)) // 64
                    offset_a = random.randint(0, stride - 1)
                    offset_b = random.randint(0, stride - 1)
                    s_tmp = s_tmp[:, :, offset_a::stride, offset_b::stride]
                    cs_tmp = cur_tmp[:, :, offset_a::stride, offset_b::stride]

                r_col_samp = s_tmp.contiguous().view(1, chans, -1)
                s_col_samp = cs_tmp.contiguous().view(1, chans, -1)

                # Compute distance matrix and find minimum along each row to
                # implicitly get matches (and minimize distance between them)
                d_mat = pairwise_distances_cos_center(r_col_samp[0].transpose(1, 0),
                                                      s_col_samp[0].transpose(1, 0))
                d_min, _ = torch.min(d_mat, 0)

                # Aggregate loss over layers
                ell_fs = ell_fs + d_min.mean()

            # Update overall loss
            ell = ell + ell_fs

        # Optional self similarity content loss between downsampled output
        # and content image. Always turn off at end for best results.
        if content_loss and not (final_pass and i > 100):
            o_scl = max(output_im.size(2), output_im.size(3))
            o_fac = o_scl / 32.
            height = int(output_im.size(2) / o_fac)
            width = int(output_im.size(3) / o_fac)

            o_flat = flatten_grid(scl_spatial(output_im, height, width))
            self_sim_out = pairwise_distances_l2(o_flat, o_flat)

            ell = ell + torch.mean(torch.abs((self_sim_out - self_sim_target)))

        # Update output's pyramid coefficients
        ell.backward()
        optimizer.step()

    # Update output's pyramid coefficients for current resolution
    # (and all coarser resolutions)
    s_pyr[scl:] = dec_lap_pyr(output_im, len(c_pyr) - 1 - scl)
    return s_pyr

def whiten(x_in, ui_in, u_in, s_in):
    """
    Applies whitening as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf

    Parameters
    ----------
    x_in
        N x D pytorch tensor
    ui_in
        D x D transposed eigenvectors of whitening covariance
    u_in
        D x D eigenvectors of whitening covariance
    s_in
        D x 1 eigenvalues of whitening covariance
    """
    def tps(x_in):
        return x_in.transpose(1, 0)
    return tps(torch.matmul(u_in, torch.matmul(ui_in, tps(x_in)) / s_in))

def colorize(x_in, ui_in, u_in, s_in):
    """
    Applies "coloring transform" as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf

    Parameters
    ----------
    x_in
        N x D pytorch tensor
    ui_in
        D x D transposed eigenvectors of coloring covariance
    u_in
        D x D eigenvectors of coloring covariance
    s_in
        D x 1 eigenvalues of coloring covariance
    """
    def tps(x_in):
        return x_in.transpose(1, 0)
    return tps(torch.matmul(u_in, torch.matmul(ui_in, tps(x_in)) * s_in))

def zca(content, style):
    """
    Matches the mean and covariance of 'content' to those of 'style'.

    Parameters
    ----------
    content
        N x D pytorch tensor of content feature vectors
    style
        N x D pytorch tensor of style feature vectors
    """
    mu_c = content.mean(0, keepdim=True)
    mu_s = style.mean(0, keepdim=True)

    content = content - mu_c
    style = style - mu_s

    cov_c = torch.matmul(content.transpose(1,0), content) / float(content.size(0))
    cov_s = torch.matmul(style.transpose(1,0), style) / float(style.size(0))

    u_c, sig_c, _ = torch.svd(cov_c + torch.eye(cov_c.size(0)).cuda()*1e-4)
    u_s, sig_s, _ = torch.svd(cov_s + torch.eye(cov_s.size(0)).cuda()*1e-4)

    sig_c = sig_c.unsqueeze(1)
    sig_s = sig_s.unsqueeze(1)


    u_c_i = u_c.transpose(1,0)
    u_s_i = u_s.transpose(1,0)

    scl_c = torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8))
    scl_s = torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8))


    whiten_c = whiten(content, u_c_i, u_c, scl_c)
    color_c = colorize(whiten_c, u_s_i, u_s, scl_s) + mu_s

    return color_c, cov_s

def zca_tensor(content, style):
    """
    Matches the mean and covariance of 'content' to those of 'style'.

    Parameters
    ----------
    content
        B x D x H x W pytorch tensor of content feature vectors
    style
        B x D x H x W pytorch tensor of style feature vectors
    """
    content_rs = content.permute(0,2,3,1).contiguous().view(-1,content.size(1))
    style_rs = style.permute(0,2,3,1).contiguous().view(-1,style.size(1))

    c_s, cov_s = zca(content_rs, style_rs)

    c_s = c_s.view(content.size(0),content.size(2),content.size(3),content.size(1)).permute(0,3,1,2)
    return c_s.contiguous(), cov_s
