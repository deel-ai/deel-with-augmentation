"""
This module implements Style Flow.
This code is implmented based on the source
code at: https://github.com/weepiess/StyleFlow-Content-Fixed-I2I
"""

from abc import abstractmethod
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from .model.vgg import vgg
from ..utils.style_transfer.utils_style_flow import calc_mean_std, set_random_seed,\
                                    weighted_mse_loss, tv_loss, gradients_loss, adain

class SAN(nn.Module):
    """
    Style-Aware Normalization.

    Parameters
    ----------
    input_dim
        Dimension of input image
    output_dim
        Dimension of output image
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(input_dim, affine=True),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=input_dim, out_channels=768, kernel_size=1)
        )

        self.code_encode_mean = nn.Sequential(
            nn.Linear(1920,1024),
            nn.PReLU(),
            nn.Linear(1024,512),
            nn.PReLU(),
            nn.Linear(512,2*output_dim)
        )

        self.fc_mean_enc = nn.Sequential(
            nn.Linear(768,2*output_dim),
            nn.PReLU()
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(output_dim*4,output_dim),
            nn.PReLU(),
            nn.Linear(output_dim,output_dim,bias=False)
        )

        self.fc_std = nn.Sequential(
            nn.Linear(output_dim*4,output_dim),
            nn.PReLU(),
            nn.Linear(output_dim,output_dim,bias=False)
        )

    def forward(self, x_in, code):
        """
        A forward function for Style-Aware Normalization.
        """
        x_in = self.encoder(x_in)
        x_in = torch.flatten(x_in, 1)

        code_c_mean = torch.tanh(self.code_encode_mean(code))
        x_mean = self.fc_mean_enc(x_in)
        merge = torch.cat([x_mean, code_c_mean], dim=1)

        mean = self.fc_mean(merge)
        std = self.fc_std(merge)
        return mean, std

class ActNorm(nn.Module):
    """
    Activation Normalization.
    It was introduced in - Glow: Generative flow with invertible
    1x1 convolutions - https://arxiv.org/abs/1807.03039
    """
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input_image):
        """
        A initialize function for Activation Normalization.
        """
        with torch.no_grad():
            flatten = input_image.permute(1, 0, 2, 3).contiguous().view(input_image.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input_image):
        """
        A forward function for Activation Normalization.
        """
        if self.initialized.item() == 0:
            self.initialize(input_image)
            self.initialized.fill_(1)
        return self.scale * (input_image + self.loc)

    def reverse(self, output):
        """
        A reverse function for Activation Normalization.
        """
        input_image = output / self.scale - self.loc
        return input_image

class InvConv2d(nn.Module):
    """
    A InvConv2d class.
    """
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q_w, _ = torch.linalg.qr(weight)
        weight = q_w.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input_image):
        """
        A forward function for InvConv2d.
        """
        output = F.conv2d(input_image, self.weight)
        return output

    def reverse(self, output):
        """
        A reverse function for InvConv2d.
        """
        input_image = F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )
        return input_image

class InvConv2dLU(nn.Module):
    """
    A InvConv2dLU class.
    """
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q_w, _ = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.linalg.lu(q_w)
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', u_mask)
        self.register_buffer('l_mask', l_mask)
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input_image):
        """
        A forward function for InvConv2dLU.
        """
        weight = self.calc_weight()
        output = F.conv2d(input_image, weight)
        return output

    def calc_weight(self):
        """
        A calc_weight function for InvConv2dLU.
        """
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        """
        A reverse function for InvConv2dLU.
        """
        weight = self.calc_weight()
        input_image = F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
        return input_image

class ZeroConv2d(nn.Module):
    """
    A ZeroConv2d class.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input_image):
        """
        A forward function for ZeroConv2d.
        """
        output = F.pad(input_image, [1, 1, 1, 1], value=1)
        output = self.conv(output)
        output = output * torch.exp(self.scale * 3)
        return output

class AffineCoupling(nn.Module):
    """
    A AffineCoupling class.
    """
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input_image):
        """
        A forward function for AffineCoupling.
        """
        in_a, in_b = input_image.chunk(2, 1)

        if self.affine:
            log_s, out_t = self.net(in_a).chunk(2, 1)
            sig_s = F.sigmoid(log_s + 2)
            out_b = (in_b + out_t) * sig_s
        else:
            net_out = self.net(in_a)
            out_b = in_b - net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        """
        A reserve function for AffineCoupling.
        """
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, out_t = self.net(out_a).chunk(2, 1)
            sig_s = F.sigmoid(log_s + 2)
            in_b = out_b / sig_s - out_t
        else:
            net_out = self.net(out_a)
            in_b = out_b + net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    """
    A Flow class consists of three reversible transformations: Actnorm layer,
    1x1 Convolution layer and Coupling layer.
    """
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, image):
        """
        A forward function for Flow.
        """
        image = self.actnorm(image)
        image = self.invconv(image)

        if self.use_coupling:
            image = self.coupling(image)
        return image

    def reverse(self, output):
        """
        A reserve function for Flow.
        """
        if self.use_coupling:
            output = self.coupling.reverse(output)
        output = self.invconv.reverse(output)
        output = self.actnorm.reverse(output)
        return output

class Block(nn.Module):
    """
    A Block class.
    """
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.san = SAN(squeeze_dim, squeeze_dim)
        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

    def forward(self, input_image):
        """
        A forward function for Block.
        """
        b_size, n_channel, height, width = input_image.shape
        squeezed = input_image.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        output = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        for flow in self.flows:
            output = flow(output)
        return output

    def reverse(self, output, style):
        """
        A reserve function for Block.
        """
        input_image = output
        mean, std = self.san(input_image,style)
        input_image = adain(input_image, mean, std)

        for flow in self.flows[::-1]:
            input_image = flow.reverse(input_image)

        b_size, n_channel, height, width = input_image.shape

        unsqueezed = input_image.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        return unsqueezed

class Glow(nn.Module):
    """
    A Glow class.
    """
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for _ in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4

        self.blocks.append(Block(n_channel, n_flow, affine=affine))

    def forward(self, content_image, forward=True, style=None):
        """
        A forward function for Glow.
        """
        if forward:
            return self._forward_set(content_image)
        return self._reverse_set(content_image, style=style)

    def _forward_set(self, content_image):
        """
        A forward_set function for Glow.
        """
        output = content_image
        for block in self.blocks:
            output = block(output)
        return output

    def _reverse_set(self, content_image, style):
        """
        A reserve_set function for Glow.
        """
        output = content_image
        for _, block in enumerate(self.blocks[::-1]):
            output = block.reverse(output, style)
        return output

class Net(nn.Module):
    """
    Net is used in StyleFlow.
    """
    def __init__(self, encoder, keep_ratio=0.6):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 31

        self.mse_loss = nn.MSELoss()
        self.keep_ratio = keep_ratio

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input_image):
        """
        Encoder function who extract relu1_1, relu2_1, relu3_1, relu4_1 from input image.
        """
        results = [input_image]
        for i in range(4):
            func = getattr(self, f'enc_{i+1}')
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input_image):
        """
        Encoder function who extract relu4_1 from input image.
        """
        for i in range(4):
            input_image = getattr(self, f'enc_{i+1}')(input_image)
        return input_image

    def calc_content_loss(self, input_content, target):
        """
        A function calculate content loss.
        """
        assert input_content.size() == target.size()
        assert target.requires_grad is False
        size1 = input_content.size()
        size2 = target.size()
        input_mean, input_std = calc_mean_std(input_content)
        target_mean, target_std = calc_mean_std(target)
        normalized_feat1 = (input_content - input_mean.expand(size1)) / input_std.expand(size1)
        normalized_feat2 = (target - target_mean.expand(size2)) / target_std.expand(size2)
        return self.mse_loss(normalized_feat1, normalized_feat2)

    def calc_style_loss(self, input_style, target):
        """
        A function calculate style loss.
        """
        assert input_style.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input_style)
        target_mean, target_std = calc_mean_std(target)
        loss_mean,loss_std = weighted_mse_loss(input_mean, target_mean,
                                                input_std, target_std, self.keep_ratio)
        return loss_mean + loss_std

    def cat_tensor(self, img):
        """
        A function cat tensor.
        """
        feat = self.encode_with_intermediate(img)
        mean,std = calc_mean_std(feat[0])
        mean = mean.squeeze(2)
        mean = mean.squeeze(2)
        std = std.squeeze(2)
        std = std.squeeze(2)
        output = torch.cat([mean,std],dim=1)
        for i in range(1,len(feat)):
            mean, std = calc_mean_std(feat[i])
            mean = mean.squeeze(2)
            mean = mean.squeeze(2)
            std = std.squeeze(2)
            std = std.squeeze(2)
            output = torch.cat([output, mean, std],dim=1)
        return output

    def forward(self, content_images, style_images, stylized_images):
        """
        A forward function for Net.
        """
        style_feats = self.encode_with_intermediate(style_images)
        content_feat = self.encode(content_images)
        stylized_feats = self.encode_with_intermediate(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feat)
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
        return loss_c, loss_s

class STYLEFLOW(nn.Module):
    """
    StyleFlow class.

    Parameters
    ----------
    in_channel
        Number of channel
    n_flow
        Number of Flow
    n_block
        Number of Block
    vgg_path
        Path of vgg_normalised
    affine
        If true use AffineCoupling
    conv_lu
        If true use InvConv2dLU otherwise use InvConv2d
    keep_ratio
        Keep ratio for Net
    device
        Cpu or CUDA
    """
    def __init__(self, in_channel, n_flow, n_block, vgg_path,
                    affine=True, conv_lu=True, keep_ratio=0.8, device="cpu"):
        super().__init__()

        self.init = True
        set_random_seed(0)
        self.model = Glow(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu)
        self.model.to(device)
        encoder = vgg
        if vgg_path is not None:
            encoder.load_state_dict(torch.load(vgg_path))
        self.encoder = Net(encoder, keep_ratio=keep_ratio)
        self.encoder.to(device)
        self.device = device

    @abstractmethod
    def train_network(self, train_loader, content_weight, style_weight, type_loss=None):
        """
        Train the StyleFlow network and return the losses.

        Parameters
        ----------
        train_loader
            Torch.tensor or Dataloader (Pairs of content images and style images)
        content_weight
            Weight of content
        style_weight
            Weight of style
        type_loss
            Type of loss function that you want to use

        Returns
        -------
        loss_train
            The losses of training process StyleFlow
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00002)

        loss_train = []
        for i, (content_iter, style_iter) in enumerate(train_loader):
            content_image = content_iter.to(self.device)
            style_image = style_iter.to(self.device)
            target_style = style_iter.to(self.device)

            if self.init:
                base_code = self.encoder.cat_tensor(style_image)
                #self.model(content_image, domain_class=base_code.to(self.device))
                z_c = self.model(content_image, forward=True)
                stylized = self.model(z_c, forward=False, style=base_code.to(self.device))
                self.init = False

            base_code = self.encoder.cat_tensor(target_style)
            #stylized = self.model(content_image, domain_class=base_code.to(self.device))
            z_c = self.model(content_image, forward=True)
            stylized = self.model(z_c, forward=False, style=base_code.to(self.device))
            stylized = torch.clamp(stylized, 0, 1)

            if type_loss == "TVLoss":
                smooth_loss = tv_loss(stylized)
            elif type_loss is None:
                smooth_loss = gradients_loss(stylized, target_style)

            loss_c, loss_s = self.encoder(content_image, style_image, stylized)
            loss_c = loss_c.mean().to(self.device)
            loss_s = loss_s.mean().to(self.device)
            loss = content_weight*loss_c + style_weight*loss_s + smooth_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train.append(loss.item())

            if i % 500 == 0:
                print(f"[{i}/{len(train_loader)} Stylizations], \tLoss: {loss.item()}")

        return loss_train

    def style_flow_generate(self, content_image, style_image):
        """
        A function that generates one image after training by StyleFlow method.

        Parameters
        ----------
        content_image
            Content image that we choose to stylizer
        style_image
            Style image that we choose for style transfer

        Returns
        -------
        gen_image
            Generated image
        """
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)

        base_code = self.encoder.cat_tensor(style_image)
        #stylized = self.model(content_image, domain_class=base_code.to(self.device))
        z_c = self.model(content_image, forward=True)
        stylized = self.model(z_c, forward=False, style=base_code.to(self.device))
        gen_image = torch.clamp(stylized, 0, 1)
        return gen_image

class StyleflowGenerate(nn.Module):
    """
    A class to train and batch create new styled images using the StyleFlow method.

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

    def forward(self, dataset, style_img, number_img, vgg_path):
        """
        A forward function for StyleflowGenerate class.

        Parameters
        ----------
        dataset
            An image tensor that you want to restyle
        style_img
            Style image that we choose for style transfer
        number_img
            Number of images that you want to style in dataset
        vgg_path
            Path of vgg_normalised

        Returns
        -------
        generated_list
            List of generated images
        """
        if number_img > len(dataset):
            raise ValueError("We must have number_img < len(dataset)")

        model = STYLEFLOW(in_channel=3, n_flow=15, n_block=2, vgg_path=vgg_path,
                                    affine=False, conv_lu=False, keep_ratio=0.8, device=self.device)
        _ = model.train_network(train_loader=self.train_loader,
                    content_weight = 0.1, style_weight=1, type_loss="TVLoss"
                )
        torch.cuda.empty_cache()
        trans = transforms.Compose([transforms.Resize((256,256))])
        generated_list = []
        list_content_img = dataset[:number_img]
        im_trg = style_img.unsqueeze(0)
        im_trg = trans(im_trg)

        for _, con_img in enumerate(list_content_img):
            im_src = con_img.unsqueeze(0)
            im_src = trans(im_src)

            gen_image = model.style_flow_generate(
                content_image = im_src,
                style_image = im_trg
            )

            torch.cuda.empty_cache()
            img = gen_image.squeeze(0)
            new_im_out = np.clip(img.detach().cpu().numpy(), 0., 1.)
            new_im_out = (new_im_out * 255).astype(np.uint8)
            new_im_out = torch.from_numpy(new_im_out/255)
            generated_list.append(new_im_out)
        return generated_list
        