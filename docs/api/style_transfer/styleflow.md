# Style Flow

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1IqTYUFowVQ3iSqsukLV0DxhVuU9LTr2y?authuser=3) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/deel-with-augmentation/blob/main/augmentare/methods/style_transfer/style_flow.py) |
📰 [Paper](https://arxiv.org/abs/2207.01909)

## NETWORK ARCHITECTURE : Style Flow

With the invertible network structure, StyleFlow first projects the input images into the feature space in the forward, while the backward uses the SAN module to perform the fixed feature transformation of the content, and then projects them into image space.

<img src="../images/flow.png" width="100%" alt="Picture" style="display: block; margin: 0 auto" />

The blue arrows indicate the forward pass to extract the features, while the red arrows represent the backward pass to reconstruct the images. StyleFlow consists of a series of reversible blocks, where each block has three components: the `Squeeze module`, the `Flow module`, and the `SAN module`. A pre-trained VGG encoder is used for domain feature extraction.
    <ul> <li> <span style="color:gold"> Squeeze module: </span> <span> The Squeeze operation serves as an interconnection between blocks for reordering features. It reduces the spatial size of the feature map by first dividing the input feature into small patches along the spatial dimension and then concatenating the patches along the channel dimension. </span> </li>
    <li> <span style="color:gold"> Flow module: </span> <span> The Flow module consists of three reversible transformations: Actnorm Layer, 1x1 Convolution Layer, and Coupling Layer. </span> </li>
    <li> <span style="color:gold"> SAN module: </span> <span> SAN module to perform fixed content feature transformation. Fixed content transfer means that content information before and after transformation should be retained. </span> </li> </ul>

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *

# Create StyleFlow method
vgg_path = '/home/vuong.nguyen/vuong/augmentare/augmentare/methods/style_transfer/model/vgg_normalised_flow.pth'
model = STYLEFLOW(in_channel=3, n_flow=15, n_block=2, vgg_path=vgg_path,
                            affine=False, conv_lu=False, keep_ratio=0.8, device=device)

# Training the StyleFlow network
loss_train = model.train_network(train_loader=train_loader,
            content_weight = 0.1, style_weight=1, type_loss="TVLoss"
        )

# Styled image by StyleFlow
gen_image = model.style_flow_generate(
    content_image= content_image,
    style_image= style_image
)
```

## Notebooks

- [**StyleFlow**: Tutorial](https://colab.research.google.com/drive/1IqTYUFowVQ3iSqsukLV0DxhVuU9LTr2y?authuser=3)
- [**StyleFlow**: Apply in EuroSAT](https://colab.research.google.com/drive/1DUzOwHdQCgiTf2wA7Da0U06dkRba5Kgg?authuser=3)

{{augmentare.methods.style_transfer.style_flow.STYLEFLOW}}

[StyleFlow For Content-Fixed Image to Image Translation](https://arxiv.org/pdf/2207.01909.pdf) by Weichen Fan & al (2022).
