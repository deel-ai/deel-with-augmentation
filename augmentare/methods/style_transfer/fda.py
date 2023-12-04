"""
This module implements Fourier Domain Adaptation.
"""

from abc import abstractmethod
import torch
from torch import nn
from torch.fft import fftn, ifftn
import numpy as np

class FDA(nn.Module):
    """
    Fourier Domain Adaptation is a simple method for unsupervised domain adaptation,
    whereby the discrepancy between the source and target distributions is reduced
    by swapping the low-frequency spectrum of one with the other.

    Ref. Yanchao Yang & Stefano Soatto., FDA: Fourier Domain Adaptation
    for Semantic Segmentation (2020). https://arxiv.org/abs/2004.05498

    Parameters
    ----------
    source_img
        Source image
    target_img
        Target image
    """
    def __init__(self, source_img, target_img):
        super().__init__()
        self.src_img = source_img
        self.trg_img = target_img

    @staticmethod
    def _low_freq_mutate(amp_src, amp_trg, beta=0.1):
        """
        Replace the low frequency part of the source amplitude with that from the target.

        Parameters
        ----------
        amp_src
            Source amplitude
        amp_trg
            Target amplitude
        beta
            Controls the size of the low frequency window to be replaced

        Returns
        -------
        amp_src
            Output of source image with low frequency amplitude was replaced from target
        """
        _, _, height, width = amp_src.size()
        batch = (np.floor(np.amin((height, width))*beta)).astype(int)     # get b
        amp_src[:,:,0:batch, 0:batch]     = amp_trg[:,:, 0:batch, 0:batch]      # top left
        amp_src[:,:,0:batch,width-batch:width] = amp_trg[:,
                                                    :,0:batch,width-batch:width]    # top right
        amp_src[:,:,height-batch:height,0:batch] = amp_trg[:,
                                                    :,height-batch:height,0:batch]    # bottom left
        amp_src[:,:,height-batch:height,width-batch:width] = \
                                amp_trg[:,:,height-batch:height,width-batch:width]  # bottom right
        return amp_src

    @abstractmethod
    def fda_source_to_target(self, beta=0.1):
        """
        FDA for exchanging magnitude.

        Parameters
        ----------
        beta
            Controls the size of the low frequency window to be replaced

        Returns
        -------
        src_in_trg
            Output image was exchanged magnitude
        """
        # get fft of both source and target
        fft_src = fftn(self.src_img.clone(), dim=(2, 3))
        fft_trg = fftn(self.trg_img.clone(), dim=(2, 3))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_trg, _ = torch.abs(fft_trg), torch.angle(fft_trg)

        # replace the low frequency amplitude part of source with that from target
        amp_src_ = FDA._low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta = beta)

        # recompose fft of source
        fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
        fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()
        fft_src_ = torch.complex(fft_src_real, fft_src_imag)

        # get the recomposed image: source content, target style
        src_in_trg = ifftn(fft_src_, dim=(2, 3))

        return src_in_trg

    def fda_source_to_target_2(self, beta=0.1):
        """
        FDA for exchanging magnitude but only take the real fft of source.

        Parameters
        ----------
        beta
            Controls the size of the low frequency window to be replaced

        Returns
        -------
        src_in_trg
            Output image was exchanged magnitude
        """
        # get fft of both source and target
        fft_src = fftn(self.src_img.clone())
        fft_trg = fftn(self.trg_img.clone())

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_trg, _ = torch.abs(fft_trg), torch.angle(fft_trg)

        # replace the low frequency amplitude part of source with that from target
        amp_src_ = FDA._low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta = beta)

        # recompose fft of source
        fft_src_ = amp_src_ * torch.exp(1j * pha_src)

        # get the recomposed image: source content, target style
        src_in_trg = ifftn(fft_src_, dim = (-2, -1))
        src_in_trg = torch.real(src_in_trg)

        return src_in_trg

class FdaGenerate(nn.Module):
    """
    A class to batch create new styled images using the FDA method.
    """
    def forward(self, dataset, style_img, number_img, beta):
        """
        A forward function for FdaGenerate class.

        Parameters
        ----------
        dataset
            An image tensor that you want to restyle
        style_img
            Style image that we choose for style transfer
        number_img
            Number of images that you want to style in dataset
        beta
            Controls the size of the low frequency window to be replaced

        Returns
        -------
        generated_list
            List of generated images
        """
        if number_img > len(dataset):
            raise ValueError("We must have number_img < len(dataset)")
        generated_list = []
        list_content_img = dataset[:number_img]
        im_trg = np.expand_dims(style_img, axis=0)
        im_trg = torch.from_numpy(im_trg)

        for _, con_img in enumerate(list_content_img):
            im_src = np.expand_dims(con_img, axis=0)
            im_src = torch.from_numpy(im_src)
            model = FDA(im_src, im_trg)
            src_in_trg = model.fda_source_to_target(beta=beta)
            img = src_in_trg.squeeze(0)
            new_im_out = np.clip(img.detach().cpu().numpy(), 0., 1.)
            new_im_out = (new_im_out*255).real.astype(np.uint8)
            new_im_out = torch.from_numpy(new_im_out/255)
            generated_list.append(new_im_out)
        return generated_list
