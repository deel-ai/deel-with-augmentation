# Progressive Growing of GANS (ProGAN)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1lYkKxrJ6Bfz0VspQpi6TtLVcYXw-tZVH?authuser=3) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/deel-with-augmentation/blob/main/augmentare/methods/gan/progan.py) |
📰 [Paper](https://arxiv.org/abs/1710.10196)

Progressive Growing GAN also known as ProGAN is an extension of the GAN training process that allows training generating models with stability that can produce large-high-quality images.

It involves training by starting with a very small image and then layer blocks are added gradually so that the output size of the generator model increases and the input size of the discriminator model increases until the desired image size is obtained. This approach has proven to be very effective in creating highly realistic, high-quality synthetic images.

It basically includes 4 steps:
- Progressive growing (of model and layers)

- Minibatch std on Discriminator

- Normalization with PixelNorm

 - Equalized Learning Rate

<div class=figure>
  <p align="center" width="100%"> <img width="100%" src="../images/progan.gif">
  <p align="center"> Simplified view of ProGAN <a href="https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2"> (Image source) </a>
</div>

Here we can see in the above figure that Progressive Growing GAN involves using a generator and discriminator model with the traditional GAN structure and its starts with very small images, such as 4×4 pixels.

During training, it systematically adds new convolutional blocks to both the generator model and the discriminator model. This gradual addition of convolutional layers allows models to effectively learn coarse-level details early on and then learn even finer details, both on the generator and discriminator.

**ProGAN goals:**
- Produce high-quality, high-resolution images.
- Greater diversity of images in the output.
- Improve stability in GANs.
- Increase variation in the generated images

## NETWORK ARCHITECTURE : ProGAN

### GENERATOR NETWORK
A generator to incrementally size the output by starting with a very small image, then the blocks of layers added incrementally and increasing the input size of the discriminant model until the desired image size is obtained.

### DISCRIMINATOR NETWORK
A discriminator for discerning real from generated images.

## LOSS FUNCTIONS
ProGAN use one of the common loss functions in GANs, the **Wasserstein** loss function, also known as **WGAN-GP** from the paper [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf?ref=blog.paperspace.com). 

$$Loss_{G} = -D(x')$$
$$GP = (||\nabla D(ax' + (1-a)x))||_2 - 1)^2$$
$$Loss_{D} = -D(x) + D(x') + \lambda * GP$$

Where:
- x' is the generated image.
- x is an image from the training set.
- D is the discriminator.
- GP is a gradient penalty that helps stabilize training.
- The a term in the gradient penalty refers to a tensor of random numbers between 0 and 1, chosen uniformly at random.
- The parameter λ is common to set to 10.

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.gan import *

# Create GAN Generator
net_gen = PROGANGenerator(
    latent_size=128,
    in_channels=128,
    img_channels=3,
    alpha=1e-5,
    steps=4
)

# Create GAN Discriminator
net_dis = PROGANDiscriminator(
    in_channels=128,
    img_channels=3,
    alpha=1e-5,
    steps=4
)

# Optimizers and Loss functions
optimizer_gen = Adam(net_gen.parameters(), lr=1e-3, betas=(0.0, 0.999))
optimizer_dis = Adam(net_dis.parameters(), lr=1e-3, betas=(0.0, 0.999))
loss_fn_gen =  torch.cuda.amp.GradScaler()
loss_fn_dis =  torch.cuda.amp.GradScaler()

# Create GAN network
gan = PROGAN(
    net_gen,
    net_dis,
    optimizer_gen,
    optimizer_dis,
    loss_fn_gen,
    loss_fn_dis,
    device,
    latent_size=128
)

# Training the ProGAN network
gen_losses, dis_losses = gan.train(
    subset_a=dataloader,
    num_epochs=5,
    num_decay_epochs=None,
    num_classes = None,
    batch_size = [32, 32, 32, 16, 16, 16, 16, 8, 4],
    subset_b = None
)

# Sample images from the Generator
img_list = gan.generate_samples(
    nb_samples = 36,
    num_classes = None,
    real_image_a = None,
    real_image_b = None
)
```

## Notebooks

- [**CycleGAN**: Tutorial](https://colab.research.google.com/drive/1lYkKxrJ6Bfz0VspQpi6TtLVcYXw-tZVH?authuser=3)
- [**CycleGAN**: Apply in CelebA](https://colab.research.google.com/drive/14GZnuvUij3UOMNFcQr6Fu_WVg8VeZXRx?authuser=3)

{{augmentare.methods.gan.progan.PROGAN}}

{{augmentare.methods.gan.progan.PROGANGenerator}}

{{augmentare.methods.gan.progan.PROGANDiscriminator}}

[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) by Tero Karras & al (2018).
