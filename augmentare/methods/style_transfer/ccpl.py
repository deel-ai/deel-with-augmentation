"""
This module implements CCPL.
This code is implmented based on the source
code at: https://github.com/JarrentWu1031/CCPL
"""

from abc import abstractmethod
import itertools
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
from .model.vgg import vgg
from .model.decoder import decoder
from ..utils.style_transfer.utils_ccpl import adjust_learning_rate, calc_mean_std,\
                                            nor_mean_std, nor_mean, calc_cov, coral

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)])

class SCT(nn.Module):
    """
    Simple Covariance Transformation module to fuse style features and content features.
    It computes the covariance of the style feature and directly multiplies the feature
    covariance with the normalized content features.

    Parameters
    ----------
    training_mode
        Mode of training: Artistic or Photo-realistic
    """
    def __init__(self, training_mode='art'):
        super().__init__()
        if training_mode == 'art':
            self.cnet = nn.Sequential(nn.Conv2d(512,256,1,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,128,1,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,32,1,1,0))
            self.snet = nn.Sequential(nn.Conv2d(512,256,3,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,128,3,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,32,1,1,0))
            self.uncompress = nn.Conv2d(32,512,1,1,0)
        else: #pho
            self.cnet = nn.Sequential(nn.Conv2d(256,128,1,1,0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128,32,1,1,0))
            self.snet = nn.Sequential(nn.Conv2d(256,128,3,1,0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128,32,1,1,0))
            self.uncompress = nn.Conv2d(32,256,1,1,0)

    def forward(self, content, style):
        """
        A forward function for SCT.
        """
        cf_nor = nor_mean_std(content)
        sf_nor, smean = nor_mean(style)
        c_f = self.cnet(cf_nor)
        s_f = self.snet(sf_nor)
        bacth, channel, width, height = c_f.size()
        s_cov = calc_cov(s_f)
        g_f = torch.bmm(s_cov, c_f.flatten(2, 3)).view(bacth, channel, width, height)
        g_f = self.uncompress(g_f)
        g_f = g_f + smean.expand(cf_nor.size())
        return g_f

class Normalize(nn.Module):
    """
    Normalize class.
    """
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x_in):
        """
        A forward function for Normalize.
        """
        norm = x_in.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        output = x_in.div(norm + 1e-7)
        return output

class ModelCCPL(nn.Module):
    """
    ModelCCPL class.
    """
    def __init__(self, mlp_in, device):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp_in
        self.device = device

    def _neighbor_sample(self, feature, layer, num_s, sample_ids=None):
        """
        A function that calculate neighbor sample.
        """
        _, _, height, width = feature.size()
        feat_r = feature.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids is None:
            sample_ids = []
            dic = {0: -(width+1), 1: -width, 2: -(width-1),
                    3: -1, 4: 1, 5: width-1, 6: width, 7: width+1}

            s_ids = torch.randperm((height - 2) * (width - 2),
                                    device=self.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = torch.div(s_ids, (width - 2), rounding_mode='trunc') + 1 # centors
            cw_ids = s_ids % (width - 2) + 1
            c_ids = (ch_ids * width + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(self.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d =self.mlp[3*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    def _patch_nceloss(self, f_q, f_k, tau=0.07):
        """
        PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation
        """
        # batch size, channel size, and number of sample locations
        batch_size, _, num_sample = f_q.shape
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(num_sample, dtype=torch.bool)[None, :, :].to(self.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(batch_size * num_sample, dtype=torch.long).to(self.device)
        output = self.cross_entropy_loss(predictions, targets)
        return output

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        """
        A forward function for ModelCCPL.
        """
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self._neighbor_sample(feats_q[i], i, num_s, sample_ids=None)
            f_k, _ = self._neighbor_sample(feats_k[i], i, num_s, sample_ids)
            loss_ccp += self._patch_nceloss(f_q, f_k, tau)
        return loss_ccp

class Net(nn.Module):
    """
    Net is used for CCPL.
    """
    def __init__(self, encoder_in, decoder_in, device, training_mode='art'):
        super().__init__()
        enc_layers = list(encoder_in.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder_in
        self.sct = SCT(training_mode)
        self.mlp = mlp if training_mode == 'art' else mlp[:9]

        self.ccpl = ModelCCPL(self.mlp, device=device)
        self.mse_loss = nn.MSELoss()
        self.end_layer = 4 if training_mode == 'art' else 3
        self.mode = training_mode

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input_image):
        """
        Encoder function who extract relu1_1, relu2_1, relu3_1, relu4_1 from input image.
        """
        results = [input_image]
        for i in range(self.end_layer):
            func = getattr(self, f'enc_{i+1}')
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input_image):
        """
        Encoder function who extract relu4_1 from input image.
        """
        for i in range(self.end_layer):
            input_image = getattr(self, f'enc_{i+1}')(input_image)
        return input_image

    def feature_compress(self, feat):
        """
        A feature compress function.
        """
        feat = feat.flatten(2,3)
        feat = self.mlp(feat)
        feat = feat.flatten(1,2)
        feat = Normalize(2)(feat)
        return feat

    def calc_content_loss(self, input_content, target):
        """
        A function calculate content loss.
        """
        assert input_content.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input_content, target)

    def calc_style_loss(self, input_style, target):
        """
        A function calculate style loss.
        """
        assert input_style.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input_style)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, num_s, num_layer):
        """
        A forward function for Net.
        """
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        g_f = self.sct(content_feats[-1], style_feats[-1])
        gimage = self.decoder(g_f)
        g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1])
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        start_layer = end_layer - num_layer
        loss_ccp = self.ccpl(g_t_feats, content_feats, num_s, start_layer, end_layer)

        return loss_c, loss_s, loss_ccp

class CCPL(nn.Module):
    """
    CCPL class.

    Parameters
    ----------
    training_mode
        Mode of training: Artistic or Photo-realistic
    vgg_path
        Path of vgg_normalised
    device
        Cpu or CUDA
    """
    def __init__(self, training_mode, vgg_path, device):
        super().__init__()

        new_vgg = vgg
        new_vgg.load_state_dict(torch.load(vgg_path))
        if training_mode == "art":
            new_decoder = decoder
            new_vgg =  nn.Sequential(*list(new_vgg.children())[:31])
        else:
            new_decoder = nn.Sequential(*list(decoder.children())[10:])
            new_vgg = nn.Sequential(*list(new_vgg.children())[:18])

        self.vgg = new_vgg
        self.vgg.to(device)
        self.decoder = new_decoder
        self.decoder.to(device)
        self.model = Net(self.vgg, self.decoder, device, training_mode)
        self.model.train()
        self.model.to(device)
        self.device = device

    @abstractmethod
    def train_network(self, content_set, style_set, num_s, num_l, max_iter,
                        content_weight, style_weight, ccp_weight):
        """
        Train the CCPL network and return the losses.

        Parameters
        ----------
        content_set
            Torch.tensor or Dataloader of content images
        style_set
            Torch.tensor or Dataloader of style images
        num_s
            Number of starting layer
        num_l
            Number of layer of CCPL
        max_iter
            Number of iteration maximun
        content_weight
            Weight of content
        style_weight
            Weight of style
        ccp_weight
            Weight of CCPL

        Returns
        -------
        loss_train
            The losses of training process CCPL
        """
        optimizer = Adam(itertools.chain(self.model.decoder.parameters(),
                    self.model.sct.parameters(), self.model.mlp.parameters()), lr=1e-4)
        content_iter = iter(content_set)
        style_iter = iter(style_set)
        loss_train = []
        for i in tqdm(range(max_iter)):
            adjust_learning_rate(optimizer, iteration_count=i, learning_rate=1e-4, lr_decay=5e-5)
            content_images = next(content_iter).to(self.device)
            style_images = next(style_iter).to(self.device)
            loss_c, loss_s, loss_ccp = self.model(content_images, style_images, num_s, num_l)
            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss_ccp = ccp_weight * loss_ccp
            loss = loss_c + loss_s + loss_ccp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())
        return loss_train

    @staticmethod
    def style_transfer(vgg_in, decoder_in, sct_in, content, style,
                        device, alpha=1.0, interpolation_weights=None):
        """
        Style transfer function for styling the image input.
        """
        assert 0.0 <= alpha <= 1.0
        content_f = vgg_in(content)
        style_f = vgg_in(style)
        if interpolation_weights:
            _, channels, height, width = content_f.size()
            feat = torch.FloatTensor(1, channels, height, width).zero_().to(device)
            base_feat = sct_in(content_f, style_f)
            for i, wei in enumerate(interpolation_weights):
                feat = feat + wei * base_feat[i:i + 1]
            content_f = content_f[0:1]
        else:
            feat = sct_in(content_f, style_f)
        return decoder_in(feat)

    def ccpl_generate(self, content_images, style_images, alpha=1.0,
                        interpolation= False, preserve_color= True):
        """
        A function that generates one image after training by CCPL method.

        Parameters
        ----------
        content_image
            Content image that we choose to stylizer
        style_image
            Style image that we choose for style transfer
        alpha
            Alpha controls the styled degree in CCPL
        interpolation
            If true then is the case one content image, N style images
        preserve_color
            True or False

        Returns
        -------
        gen_image
            Generated image
        """
        for con_image in content_images:
            # one content image, N style images
            if interpolation:
                con_image = con_image.unsqueeze(0).expand_as(style_images)
                con_image = con_image.to(self.device)
                style_images = style_images.to(self.device)
                with torch.no_grad():
                    gen_image = CCPL.style_transfer(self.vgg, self.decoder, self.model.sct,
                                        con_image, style_images, device = self.device, alpha=alpha)
            # one content image and one style image
            else:
                for sty_image in style_images:
                    if preserve_color:
                        sty_image = coral(sty_image, con_image)
                    sty_image = sty_image.to(self.device).unsqueeze(0)
                    new_con_image = con_image.to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        gen_image = CCPL.style_transfer(self.vgg, self.decoder, self.model.sct,
                                        new_con_image, sty_image, device = self.device, alpha=alpha)
        return gen_image

class CcplGenerate(nn.Module):
    """
    A class to train and batch create new styled images using the CCPL method.

    Parameters
    ----------
    con_img_train
        Torch.tensor or Dataloader of content images
    sty_img_train
        Torch.tensor or Dataloader of style images
    device
        Cpu or CUDA
    """
    def __init__(self, con_img_train, sty_img_train, device):
        super().__init__()
        self.con_img_train = con_img_train
        self.sty_img_train = sty_img_train
        self.device = device

    def forward(self, dataset, style_img, number_img, max_iter, vgg_path):
        """
        A forward function for CcplGenerate class.

        Parameters
        ----------
        dataset
            An image tensor that you want to restyle
        style_img
            Style image that we choose for style transfer
        number_img
            Number of images that you want to style in dataset
        max_iter
            Number of iterations that you want to train CCPL network
        vgg_path
            Path of vgg_normalised

        Returns
        -------
        generated_list
            List of generated images
        """
        if number_img > len(dataset):
            raise ValueError("We must have number_img < len(dataset)")

        model = CCPL(training_mode= "pho", vgg_path=vgg_path, device=self.device)
        _ = model.train_network(self.con_img_train, self.sty_img_train, num_s=8, num_l=3,
                        max_iter=max_iter, content_weight=1.0, style_weight=10.0, ccp_weight=5.0)

        generated_list = []
        list_content_img = dataset[:number_img]
        im_trg = np.expand_dims(style_img, axis=0)
        im_trg = torch.from_numpy(im_trg)

        for _, con_img in enumerate(list_content_img):
            im_src = np.expand_dims(con_img, axis=0)
            im_src = torch.from_numpy(im_src)

            gen_image = model.ccpl_generate(
                im_src, im_trg,
                alpha=1.0, interpolation= False, preserve_color= False
            )
            torch.cuda.empty_cache()
            img = gen_image.squeeze(0)
            new_im_out = np.clip(img.detach().cpu().numpy(), 0., 1.)
            new_im_out = (new_im_out * 255).astype(np.uint8)
            new_im_out = torch.from_numpy(new_im_out/255)
            generated_list.append(new_im_out)
        return generated_list
