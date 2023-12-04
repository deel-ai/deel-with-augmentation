"""
This module implements AdaIn Style Transfer.
"""

from abc import abstractmethod
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .model.vgg19 import Vgg19Pretrained
from .model.decoder import Decoder

class ADAIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) that aligns the mean and variance of the content
    features with those of the style features. It achieves speed comparable to the fastest
    existing approach, without the restriction to a pre-defined set of styles. In addition,
    this approach allows flexible user controls such as content-style trade-off, style
    interpolation, color & spatial controls, all using a single feed-forward neural network.

    Ref. Xun Huang & Serge Belongie., Arbitrary Style Transfer in Real-time
    with Adaptive Instance Normalization (2017). https://arxiv.org/abs/1703.06868
    """
    def __init__(self, device):
        super().__init__()
        self.vgg_encoder = Vgg19Pretrained()
        self.vgg_encoder.to(device)
        self.decoder = Decoder()
        self.decoder.to(device)
        self.device = device

    @staticmethod
    def _calc_mean_std(features, eps=1e-6):
        """
        A function that calculates mean and std of features on the channel axis.

        Parameters
        ----------
        features
            Shape of features = [batch_size, channels, height, width]
        eps
            Eps is a small value added to the variance to avoid divide-by-zero

        Returns
        -------
        features_mean
            Mean of features
        feature_std
            Std of features
        """
        size = features.size()
        batch_size, channels = size[:2]
        features_mean = features.reshape(batch_size,
                        channels, -1).mean(dim=2).reshape(batch_size, channels, 1, 1)

        features_std = features.reshape(batch_size,
                        channels, -1).std(dim=2).reshape(batch_size, channels, 1, 1)\
                                        + eps
        return features_mean, features_std

    @staticmethod
    def _adain(content_features, style_features):
        """
        Adaptive Instance Normalization is a normalization method that aligns the mean and
        variance of the content features with those of the style features.

        Instance Normalization normalizes the input to a single style specified by the
        affine parameters. Adaptive Instance Normaliation is an extension. In AdaIN,
        we receive a content input x and a style input y, and we simply align the channel-wise
        mean and variance of x to match those of y.

        Parameters
        ----------
        content_features
            Shape of content_features = [batch_size, channels, height, width]
        style_features
            Shape of style_features = [batch_size, channels, height, width]

        Returns
        -------
        normalized_features
            Normalized features by AdaIn
        """
        content_mean, content_std = ADAIN._calc_mean_std(content_features)
        style_mean, style_std = ADAIN._calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std\
                                                                             + style_mean
        return normalized_features

    @staticmethod
    def _calc_content_loss(features, target):
        """
        A function that calcules the content loss.

        Parameters
        ----------
        features
            Features for calculating Content loss
        target
            Target for calculating Content loss

        Returns
        -------
        content_loss
            Content loss
        """
        content_loss = F.mse_loss(features, target)
        return content_loss

    @staticmethod
    def _calc_style_loss(content_middle_features, style_middle_features):
        """
        A function that calcules the style loss.

        Parameters
        ----------
        content_middle_features
            Middle features of content images
        style_middle_features
            Middle features of style images

        Returns
        -------
        style_loss
            Style loss
        """
        style_loss = 0
        for content, style in zip(content_middle_features, style_middle_features):
            c_mean, c_std = ADAIN._calc_mean_std(content)
            s_mean, s_std = ADAIN._calc_mean_std(style)
            style_loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return style_loss

    @abstractmethod
    def train_network(self, num_epochs, train_loader, optimizer, alpha=1.0, lamb = 10):
        """
        Train the AdaIn network and return the losses.

        Parameters
        ----------
        num_epochs
            The number of epochs you want to train your AdaIn
        train_loader
            Torch.tensor or Dataloader (Pairs of content images and style images)
        optimizer
            An optimizer for AdaIn
        alpha
            Alpha controls the fusion degree in AdaIn
        lamb
            Lamb control the weight of loss

        Returns
        -------
        loss_train
            The losses of training process AdaIn
        """
        loss_train = []
        for i in range(1, num_epochs + 1):
            for j, (content_image, style_image) in tqdm(enumerate(train_loader, 1)):
                content_image = content_image.to(self.device)
                style_image = style_image.to(self.device)

                content_features = self.vgg_encoder(content_image, output_last_feature=True)
                style_features = self.vgg_encoder(style_image, output_last_feature=True)
                ada = ADAIN._adain(content_features, style_features)
                ada = alpha * ada + (1 - alpha) * content_features
                out = self.decoder(ada)

                output_features = self.vgg_encoder(out, output_last_feature=True)
                output_middle_features = self.vgg_encoder(out, output_last_feature=False)
                style_middle_features = self.vgg_encoder(style_image, output_last_feature=False)

                loss_c = ADAIN._calc_content_loss(output_features, ada)
                loss_s = ADAIN._calc_style_loss(output_middle_features, style_middle_features)
                loss = loss_c + lamb * loss_s

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())

                if i % 10 == 0:
                    print(f"[{i}/{num_epochs+1} epochs], [{j}/{len(train_loader)+1} train_loader],\
                                        \tLoss: {loss.item()}")
        return loss_train

    def adain_generate(self, content_image, style_image, alpha=1.0):
        """
        A function that generates one image after training by AdaIn method.

        Parameters
        ----------
        content_image
            Content image that we choose to stylizer
        style_image
            Style image that we choose for style transfer
        alpha
            Alpha controls the fusion degree in AdaIn

        Returns
        -------
        gen_image
            Generated image
        """
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)

        content_feature = self.vgg_encoder(content_image, output_last_feature = True)
        style_feature = self.vgg_encoder(style_image, output_last_feature = True)
        ada = ADAIN._adain(content_feature, style_feature)
        ada = alpha * ada + (1 - alpha) * content_feature
        gen_image = self.decoder(ada)
        return gen_image

class AdainGenerate(nn.Module):
    """
    A class to train and batch create new styled images using the AdaIN method.

    Parameters
    ----------
    train_loader
        Dataloader (Pairs of content images and style images)
    device
        Cpu or CUDA
    """
    def __init__(self, train_loader, device):
        super().__init__()
        self.train_loader = train_loader
        self.device = device

    def denorm(self, tensor):
        """
        A decoding function that normalizes the image to the original.

        Parameters
        ----------
        tensor
            Image need denorm

        Returns
        -------
        res
            Original image
        """
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(self.device)
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(self.device)
        res = torch.clamp(tensor * std + mean, 0, 1)
        return res

    def forward(self, dataset, style_img, number_img, num_epochs, alpha):
        """
        A forward function for AdainGenerate class.

        Parameters
        ----------
        dataset
            An image tensor that you want to restyle
        style_img
            Style image that we choose for style transfer
        number_img
            Number of images that you want to style in dataset
        num_epochs
            The number of epochs you want to train your AdaIn
        alpha
            Alpha controls the fusion degree in AdaIn

        Returns
        -------
        generated_list
            List of generated images
        """
        if number_img > len(dataset):
            raise ValueError("We must have number_img < len(dataset)")

        model = ADAIN(self.device)
        optimizer = Adam(model.parameters(), lr=1e-4)
        torch.cuda.empty_cache()
        _ = model.train_network(
                    num_epochs=num_epochs,
                    train_loader= self.train_loader,
                    optimizer= optimizer
                )

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([normalize])

        generated_list = []
        list_content_img = dataset[:number_img]
        im_trg = trans(style_img).unsqueeze(0).to(self.device)
        list_content_img = trans(list_content_img)

        for _, con_img in enumerate(list_content_img):
            im_src = con_img.unsqueeze(0).to(self.device)

            gen_image = model.adain_generate(im_src, im_trg, alpha=alpha)
            gen_image = self.denorm(gen_image)

            torch.cuda.empty_cache()
            img = gen_image.squeeze(0)
            new_im_out = np.clip(img.detach().cpu().numpy(), 0., 1.)
            new_im_out = (new_im_out * 255).astype(np.uint8)
            new_im_out = torch.from_numpy(new_im_out/255)
            generated_list.append(new_im_out)
        return generated_list
