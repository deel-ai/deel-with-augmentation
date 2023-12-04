# DCGAN (Deep Convolutional GAN)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial]() |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source]() |
📰 [Paper]()

<div class=figure>
  <p align="center" width="100%"> <img width="55%" src="/home/vuong.nguyen/vuong/augmentare/docs/assets/dcgan.png">
  <p align="center"> DCGAN Generator
</div>

<div class=figure>
  <p align="center" width="100%"> <img width="75%" src="/home/vuong.nguyen/vuong/augmentare/docs/assets/full_dcgan.png">
  <p align="center"> Full DCGAN Architecture
</div>

## NETWORK ARCHITECTURE : DCGAN

### GENERATOR NETWORK
The DCGAN Generator receives an input noise vector of size $batch\ size \times latent\ size$. It outputs a tensor of $batch\ size \times channel \times height \times width$ corresponding to a batch of generated image samples. The generator transforms the noise vector into images in the following manner:

1. **Channel Dimension**: $encoding\ dims \rightarrow d \rightarrow \frac{d}{2} \rightarrow \frac{d}{4} \rightarrow \frac{d}{8} \rightarrow 1$.
2. **Image size**: $(1 \times 1) \rightarrow (4 \times 4) \rightarrow (8 \times 8) \rightarrow (16 \times 16) \rightarrow (32 \times 32) \rightarrow (64 \times 64)$.

The intermediate layers use the **ReLU** activation function to kill gradients and slow down convergence. We can also use any other activation to ensure a good gradation flow. The last layer uses the **Tanh** activation to constrain the pixel values ​​to the range of $(- 1 \to 1)$. We can easily change the nonlinearity of the intermediate and last layers to their preferences by passing them as parameters during Generator object initialization. 

### DISCRIMINATOR NETWORK
The DCGAN Discriminator has a symmetric architecture to the generator. It maps the image with a confidence score to classify whether the image is real (i.e. comes from the dataset) or fake (i.e. sampled by the generator) 

We use the **Leaky ReLU** activation for Discriminator. The conversion of image tension to confidence score is as follows: 

1. **Channel Dimension**: $1 \rightarrow d \rightarrow 2 \times d \rightarrow 4 \times d \rightarrow 8 \times d \rightarrow 1$.
2. **Image size**: $(64 \times 64) \rightarrow (32 \times 32) \rightarrow (16 \times 16) \rightarrow (8 \times 8) \rightarrow (4 \times 4) \rightarrow (1 \times 1)$.

The last layer of DCGAN's Discriminator has a **Sigmoid** layer that makes the confidence score between $(0 \to 1)$ and allows the confidence score to be easily interpreted in terms of the probability that the image is real. However, this interpretation is restricted only to the Minimax Loss proposed in the original GAN paper, and losses such as the Wasserstein Loss require no such interpretation. However, if required, one can easily set last layer activation to **Sigmoid** by passing it as a parameter during initialization time.

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.gan import *

# Create GAN Generator
net_gen = DCGANGenerator(
    num_channels=1,
    latent_size=100,
    feature_map_size=64
)

# Create GAN Discriminator
net_dis = DCGANDiscriminator(
    num_channels=1,
    feature_map_size=64
)

# Optimizers and Loss functions
optimizer_gen = Adam(net_gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_dis = Adam(net_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss_fn_gen =  nn.BCELoss()
loss_fn_dis =  nn.BCELoss()

# Create GAN network
gan = DCGAN(
    net_gen,
    net_dis,
    optimizer_gen,
    optimizer_dis,
    loss_fn_gen,
    loss_fn_dis,
    device,
    latent_size=100,
    init_weights=True
)

# Training the DCGAN network
gen_losses, dis_losses = gan.train(
    subset_a=dataloader,
    num_epochs=10,
    num_decay_epochs = None,
    num_classes = None,
    batch_size = None,
    subset_b = None 
)

# Sample images from the Generator
img_list = gan.generate_samples(
    nb_samples=64,
    num_classes = None,
    real_image_a = None,
    real_image_b = None
)
```

## Notebooks

- [**DCGAN**: Tutorial]()
- [**DCGAN**: Apply in EuroSAT]()
- [**DCGAN**: Apply in CelebA]()

{{augmentare.methods.gan.dcgan.DCGANGenerator}}

{{augmentare.methods.gan.dcgan.DCGANDiscriminator}}

{{augmentare.methods.gan.dcgan.DCGAN}}

[Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks](https://arxiv.org/abs/1511.06434) by Radford & al (2015).
