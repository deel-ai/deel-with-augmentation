# Conditional GAN (CGAN)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1PRgSd3HfaNs1VtgKMYuqwJ5pMfr2GrTV?authuser=3) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/deel-with-augmentation/blob/main/augmentare/methods/gan/cgan.py) |
📰 [Paper](https://arxiv.org/abs/1411.1784)

<div class=figure>
  <p align="center" width="100%"> <img width="100%" height="100%" src="../images/cgan.png">
  <p align="center"> Conditional GAN Architecture
</div>

## NETWORK ARCHITECTURE : CGAN

### GENERATOR NETWORK
The CGAN Generator is parameterized to learn and produce realistic samples for each label in the training dataset. It receives an input noise vector of size $batch\ size \times latent\ size$. It outputs a tensor of $batch\ size \times channel \times height \times width$ corresponding to a batch of generated image samples.

The intermediate layers use the **LeakyReLU** activation function to kill gradients and slow down convergence. We can also use any other activation to ensure a good gradation flow. The last layer uses the **Tanh** activation to constrain the pixel values ​​to the range of $(- 1 \to 1)$. 

### DISCRIMINATOR NETWORK
The CGAN Discriminator learns to distinguish fake and real samples, given the label information. It has a symmetric architecture to the generator. It maps the image with a confidence score to classify whether the image is real (i.e. comes from the dataset) or fake (i.e. sampled by the generator) 

We use the **LeakyReLU** activation for Discriminator. We also use **Dropout** activation, it's an effective technique for regularization and preventing the co-adaptation of neurons.

The last layer of CGAN's Discriminator has a **Sigmoid** layer that makes the confidence score between $(0 \to 1)$ and allows the confidence score to be easily interpreted in terms of the probability that the image is real. However, this interpretation is restricted only to the Minimax Loss proposed in the original GAN paper, and losses such as the Wasserstein Loss require no such interpretation. However, if required, one can easily set last layer activation to **Sigmoid** by passing it as a parameter during initialization time.

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.gan import *

# Create GAN Generator
net_gen = CGANGenerator(
    latent_size=100,
    num_classes=10,
    image_shape=image_shape
)

# Create GAN Discriminator
net_dis = CGANDiscriminator(
    num_classes=10,
    image_shape=image_shape
)

# Optimizers and Loss functions
optimizer_gen = Adam(net_gen.parameters(), lr=0.0001)
optimizer_dis = Adam(net_dis.parameters(), lr=0.0001)
loss_fn_gen =  nn.BCELoss()
loss_fn_dis =  nn.BCELoss()

# Create GAN network
gan = CGAN(
    net_gen,
    net_dis,
    optimizer_gen,
    optimizer_dis,
    loss_fn_gen,
    loss_fn_dis,
    device,
    latent_size=100,
    init_weights=False
)

# Training the CGAN network
gen_losses, dis_losses = gan.train(
    subset_a=dataloader,
    num_epochs=20,
    num_decay_epochs = None,
    num_classes=10,
    batch_size=32,
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

- [**CGAN**: Tutorial](https://colab.research.google.com/drive/1PRgSd3HfaNs1VtgKMYuqwJ5pMfr2GrTV?authuser=3)

{{augmentare.methods.gan.cgan.CGAN}}

{{augmentare.methods.gan.cgan.CGANGenerator}}

{{augmentare.methods.gan.cgan.CGANDiscriminator}}

[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) by Mehdi Mirza & Simon Osindero (2014).
