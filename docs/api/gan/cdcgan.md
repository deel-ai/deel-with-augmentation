# CDCGAN (Conditional Deep Convolutional GAN)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial]() |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source]() |
📰 [Paper]()


`Conditional Deep Convolutional GAN` is a conditional GAN that use the same convolution layers as [`DCGAN`]() that is described previously. `CDCGAN` generate more realistic images than `CGAN` thanks to convolutional layers.

## NETWORK ARCHITECTURE : CDCGAN

### GENERATOR NETWORK
The CDCGAN Generator is parameterized to learn and produce realistic samples for each label in the training dataset. It receives an input noise vector of size $batch\ size \times latent\ size$. It outputs a tensor of $batch\ size \times channel \times height \times width$ corresponding to a batch of generated image samples.

The intermediate layers use the **ReLU** activation function to kill gradients and slow down convergence. We can also use any other activation to ensure a good gradation flow. The last layer uses the **Tanh** activation to constrain the pixel values ​​to the range of $(- 1 \to 1)$. 

### DISCRIMINATOR NETWORK
The CDCGAN Discriminator learns to distinguish fake and real samples, given the label information. It has a symmetric architecture to the generator. It maps the image with a confidence score to classify whether the image is real (i.e. comes from the dataset) or fake (i.e. sampled by the generator) 

We use the **LeakyReLU** activation for Discriminator.

The last layer of CDCGAN's Discriminator has a **Sigmoid** layer that makes the confidence score between $(0 \to 1)$ and allows the confidence score to be easily interpreted in terms of the probability that the image is real. However, this interpretation is restricted only to the Minimax Loss proposed in the original GAN paper, and losses such as the Wasserstein Loss require no such interpretation. However, if required, one can easily set last layer activation to **Sigmoid** by passing it as a parameter during initialization time.

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.gan import *

# Create GAN Generator
net_gen = CDCGANGenerator(
    num_classes=10,
    latent_size=10,
    label_embed_size=5,
    channels=3,
    conv_dim=64
)

# Create GAN Discriminator
net_dis = CDCGANDiscriminator(
    num_classes=10,
    channels=3,
    conv_dim=64,
    image_size=image_size
)

# Optimizers and Loss functions
optimizer_gen = Adam(net_gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_dis = Adam(net_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss_fn_gen =  nn.BCELoss()
loss_fn_dis =  nn.BCELoss()

# Create GAN network
gan = CDCGAN(
    net_gen,
    net_dis,
    optimizer_gen,
    optimizer_dis,
    loss_fn_gen,
    loss_fn_dis,
    device,
    latent_size=10,
    init_weights=False
)

# Training the CDCGAN network
gen_losses, dis_losses = gan.train(
    subset_a=dataloader,
    num_epochs=200,
    num_decay_epochs = None,
    num_classes = None,
    batch_size=256,
    subset_b = None
)

# Sample images from the Generator
img_list = gan.generate_samples(
    nb_samples=32,
    num_classes=8,
    real_image_a = None,
    real_image_b = None
)
```

## Notebooks

- [**CDCGAN**: Tutorial]()

{{augmentare.methods.gan.cdcgan.CDCGANGenerator}}

{{augmentare.methods.gan.cdcgan.CDCGANDiscriminator}}

{{augmentare.methods.gan.cdcgan.CDCGAN}}

[Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks](https://arxiv.org/abs/1511.06434) by Radford & al (2015).
