# Adaptive Instance Normalization (AdaIN)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/13kF3YeJL37RkgQwDJm13JvSZrCQkFMad?authuser=3) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/deel-with-augmentation/blob/main/augmentare/methods/style_transfer/adain.py) |
📰 [Paper](https://arxiv.org/abs/1703.06868)

## NETWORK ARCHITECTURE : AdaIN
They use the first few layers of a fixed VGG-19 network to encode the content and style images. An AdaIN layer is used to perform style transfer in the feature space. A decoder is learned to invert the AdaIN output to the image spaces. They use the same VGG encoder to compute a content loss L<sub>c</sub> and a style loss L<sub>s</sub>.

<img src="../images/adain.png" width="100%" alt="Picture" style="display: block; margin: 0 auto" />

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *

# Create AdaIN network
model = ADAIN(device)

# Optimizers
optimizer = Adam(model.parameters(), lr=1e-4)

# Training the AdaIN network
loss_train = model.train_network(
            num_epochs=49,
            train_loader= train_loader,
            optimizer= optimizer
        )

# Styled image by AdaIN
gen_image = model.adain_generate(content_tensor, style_tensor, alpha=1.0)
```

## Notebooks

- [**AdaIN**: Tutorial](https://colab.research.google.com/drive/13kF3YeJL37RkgQwDJm13JvSZrCQkFMad?authuser=3)
- [**AdaIN**: Apply in EuroSAT](https://colab.research.google.com/drive/10jBKY2sIVxBFt3ENHUFQIRm0Z6EfBII8?authuser=3)

{{augmentare.methods.style_transfer.adain.ADAIN}}

[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf) by Xuan Huang & Serge Belongie (2017).
