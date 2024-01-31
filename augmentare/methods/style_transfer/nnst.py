"""
This module implements Neural Neighbor Style Transfer.
This code is implmented based on the source
code at: https://github.com/nkolkin13/NeuralNeighborStyleTransfer
"""

from abc import abstractmethod
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model.vgg16 import Vgg16Pretrained
from ..utils.style_transfer.utils_nnst import dec_lap_pyr, syn_lap_pyr, extract_feats,\
                    replace_features, optimize_output_im, color_match

class NNST(nn.Module):
    """
    Neural Neighbor Style Transfer is a pipeline that offers state-of-the-art quality,
    generalization, and competitive efficiency for artistic style transfer. It's approach
    is similar to prior work, but it dramatically improve the final visual quality.

    Ref. Nicholas Kolkin & al., Neural Neighbor
    Style Transfer (2022). https://arxiv.org/abs/2203.13215

    Parameters
    ----------
    content_im
        content image that we choose to stylizer
    style_img
        style image that we choose for style transfer
    device
        CPU or CUDA
    """
    def __init__(self, content_im, style_im, device):
        super().__init__()
        self.content_im = content_im.to(device)
        self.style_im = style_im.to(device)
        self.device = device

    @abstractmethod
    def stylization(self, phi, max_iter = 350,
                        l_rate=1e-3, style_weight=1.,
                        max_scls=0, flip_aug=False,
                        content_loss=False, zero_init=False,
                        dont_colorize=False):
        """
        Produce stylization of content_im in the style of style_im.

        Parameters
        ----------
        phi
            Lambda function to extract features using VGG16Pretrained model
        max_iter
            Maximum number of optimization iterations for "optimize_output_im" function
            to update image pyramid per scale. Optimize laplacian pyramid coefficients of
            stylized image at a given resolution, and return stylized pyramid coefficients
        l_rate
            Learning rate of optimizer updating pyramid coefficients
        style_weight
            Controls stylization level, between 0 and 1
        max_scl
            Number of scales to stylize (performed coarse to fine)
        flip_aug
            Extract features from rotations of style image too or not.
            If true, extract style features from rotations of style
            image. This increases content preservation by making
            more options available when matching style features
            to content features
        content_loss
            If true, also minimize content loss that maintains
            self-similarity in color space between 32pixel
            downsampled output image and content image
        zero_init
            If true initialize w/ grey image, o.w. initialize w/ downsampled content image
        dont_colorize
            Colorize or not

        Returns
        -------
        stylized_im
            Output stylized image
        """
        # Get max side length of final output and set number of pyramid levels to
        # optimize over
        max_size = max(self.content_im.size(2), self.content_im.size(3))
        pyr_levs = 8

        # Decompose style image, content image, and output image into laplacian
        # pyramid
        style_pyr = dec_lap_pyr(self.style_im, pyr_levs)
        c_pyr = dec_lap_pyr(self.content_im, pyr_levs)
        s_pyr = dec_lap_pyr(self.content_im.clone(), pyr_levs)

        # Initialize output image pyramid
        if zero_init:
            # Initialize with flat grey image (works, but less vivid)
            num_pyr = len(s_pyr)
            for i in range(num_pyr):
                s_pyr[i] = s_pyr[i] * 0.
            s_pyr[-1] = s_pyr[-1] * 0. + 0.5

        else:
            # Initialize with low-res version of content image (generally better
            # results, improves contrast of final output)
            z_max = 2
            if max_size < 1024:
                z_max = 3

            for i in range(z_max):
                s_pyr[i] = s_pyr[i] * 0.

        # Stylize using hypercolumn matching from coarse to fine scale
        l_i = -1
        scl = None
        for scl in range(max_scls)[::-1]:

            # Get content image and style image from pyramid at current resolution
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            style_im_tmp = syn_lap_pyr(style_pyr[scl:])
            content_im_tmp = syn_lap_pyr(c_pyr[scl:])
            #output_im_tmp = syn_lap_pyr(s_pyr[scl:])
            l_i += 1

            # Construct stylized activations
            with torch.no_grad():

                # Control tradeoff between searching for features that match
                # current iterate, and features that match content image (at
                # coarsest scale, only use content image)
                alpha = style_weight
                if l_i == 0:
                    alpha = 0.

                # Search for features using high frequencies from content
                # (but do not initialize actual output with them)
                output_extract = syn_lap_pyr([c_pyr[scl]] + s_pyr[(scl + 1):])

                # Extract style features from rotated copies of style image
                feats_s = extract_feats(style_im_tmp, phi, flip_aug=flip_aug).cpu()

                # Extract features from convex combination of content image and
                # current iterate:
                c_tmp = (output_extract * alpha) + (content_im_tmp * (1. - alpha))
                feats_c = extract_feats(c_tmp, phi).cpu()

                # Replace content features with style features
                target_feats = replace_features(feats_c, feats_s)

            # Synthesize output at current resolution using hypercolumn matching
            s_pyr = optimize_output_im(s_pyr, c_pyr, style_im_tmp,
                                    target_feats, l_rate, max_iter, scl, phi,
                                    content_loss=content_loss)

            # Get output at current resolution from pyramid
            with torch.no_grad():
                output_im = syn_lap_pyr(s_pyr)

        # Perform final pass using feature splitting (pass in flip_aug argument
        # because style features are extracted internally in this regime)
        s_pyr = optimize_output_im(s_pyr, c_pyr, style_im_tmp,
                                target_feats, l_rate, max_iter, scl, phi,
                                final_pass=True, content_loss=content_loss,
                                flip_aug=flip_aug)
        # Get final output from pyramid
        with torch.no_grad():
            output_im = syn_lap_pyr(s_pyr)

        if dont_colorize:
            stylized_im = output_im
        else:
            stylized_im = color_match(self.content_im, self.style_im, output_im)

        return stylized_im

    def nnst_generate(self, max_scales, alpha,
                content_loss=False, flip_aug=False,
                zero_init = False, dont_colorize=False):
        """
        A function that generates the image by NNST method.

        Parameters
        ----------
        max_scales
            Number of scales to stylize (performed coarse to fine)
        alpha
            Alpha is between 0 and 1
        content_loss
            Use self-sim content loss or not
        flig_aug
            Extract features from rotations of style image too or not
        zero_init
            If true initialize w/ grey image, o.w. initialize w/ downsampled content image
        dont_colorize
            Colorize or not

        Returns
        -------
        gen_image
            Generated image
        """
        # Fix random seed
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        style_weight = 1. - alpha

        # Define feature extractor
        cnn = Vgg16Pretrained()
        cnn.to(self.device)
        def phi(x_in, y_in, z_in):
            return cnn.forward(x_in, inds=y_in, concat=z_in)

        # Stylization
        if self.device == "cuda":
            torch.cuda.synchronize()

        gen_image = self.stylization(
            phi,
            max_iter= 200,
            l_rate=2e-3,
            style_weight = style_weight,
            max_scls = max_scales,
            flip_aug = flip_aug,
            content_loss = content_loss,
            zero_init = zero_init,
            dont_colorize =  dont_colorize
        )
        torch.cuda.synchronize()
        return gen_image

class NnstGenerate(nn.Module):
    """
    A class to batch create new styled images using the NNST method.

    Parameters
    ----------
    device
        Cpu or CUDA
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def tensor_processing(self, img_tensor, target_size=1000, side_comp=max):
        """
        A image processing function.

        Parameters
        ----------
        img_tensor
            Tensor of image need processing
        target_size
            Size of image that you want to aim for
        side_comp
            Function for rescaling to desired size

        Returns
        -------
        x_in
            Result image after processing
        """
        x_in = img_tensor.permute(1,2,0)
        x_dims = x_in.shape
        x_in = x_in.contiguous().permute(2, 0, 1).contiguous()

        # Rescale to desired size
        # by default maintains aspect ratio relative to long side
        # change side_comp to be min for short side
        fac = float(target_size) / side_comp(x_dims[:2])
        height = int(x_dims[0] * fac)
        width = int(x_dims[1] * fac)
        # Scale spatial (bilinear interpolation)
        x_in = F.interpolate(x_in.unsqueeze(0), (height, width),
                            mode='bilinear', align_corners=True)[0]
        return x_in

    def forward(self, dataset, style_img, number_img, alpha):
        """
        A forward function for NnstGenerate class.

        Parameters
        ----------
        dataset
            An image tensor that you want to restyle
        style_img
            Style image that we choose for style transfer
        number_img
            Number of images that you want to style in dataset
        alpha
            Alpha is between 0 and 1

        Returns
        -------
        generated_list
            List of generated images
        """
        if number_img > len(dataset):
            raise ValueError("We must have number_img < len(dataset)")

        generated_list = []
        list_content_img = dataset[:number_img]
        im_trg = self.tensor_processing(style_img, target_size=512).to(self.device).unsqueeze(0)

        for _, con_img in enumerate(list_content_img):
            im_src = self.tensor_processing(con_img,
                                target_size=512).to(self.device).unsqueeze(0)

            model = NNST(im_src, im_trg, self.device)

            gen_image = model.nnst_generate(
                max_scales=5, alpha=alpha,
                content_loss=False, flip_aug=False,
                zero_init = False, dont_colorize=False
            )
            torch.cuda.empty_cache()
            img = gen_image.squeeze(0)
            new_im_out = np.clip(img.detach().cpu().numpy(), 0., 1.)
            new_im_out = (new_im_out*255).real.astype(np.uint8)
            new_im_out = torch.from_numpy(new_im_out/255)
            generated_list.append(new_im_out)
        return generated_list
