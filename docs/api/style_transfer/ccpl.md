# Contrastive Coherence Preserving Loss for Versatile Style Transfer (CCPL)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial]() |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source]() |
📰 [Paper]()

## NETWORK ARCHITECTURE : CCPL

<img src="/home/vuong.nguyen/vuong/augmentare/docs/assets/ccpl1.png" alt="Picture" style="display: block; margin: 0 auto" />

<span style="font-family:Roboto; color:gold"> Inspirations for CCPL: </span> <span style="font-family:Roboto"> Regions denoted by red boxes from the first frame `(RA or R'A)` have the same location with corresponding patches in the second frame wrapped in a yellow box `(RB or R'B)`. `RC and R'C` (in the blue boxes) are cropped from the first frame but their style aligns with `RB and R'B`. The difference between two patches is denoted by `D` (for example, D(RA, RB)). Mutual information between `D(RA, RC)` and `D(R'A, R'C)`, `(D(RA, RB) and D(R'A, R'B))` is encouraged to be maximized to preserve consistency from the content source. </span> </li>

<img src="/home/vuong.nguyen/vuong/augmentare/docs/assets/ccpl2.png" alt="Picture" style="display: block; margin: 0 auto" />

<span style="font-family:Roboto; color:gold"> Details of CCPL: </span> <span style="font-family:Roboto"> `Cf` and `Gf` represent the encoded features of a specific layer of encoder `E`. `⊖` denotes vector subtraction, and `SCE` stands for softmax cross-entropy. The yellow dotted lines illustrate how the positive pair is produced. </span>

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *

# Create CCPL method
vgg_path = '/home/vuong.nguyen/vuong/augmentare/augmentare/methods/style_transfer/model/vgg_normalised_ccpl.pth'
model = CCPL(training_mode= "pho", vgg_path=vgg_path, device=device)

# Training the CCPL network
loss_train = model.train_network(content_images, style_images, num_s=8, num_l=3, max_iter=50000,
                        content_weight=1.0, style_weight=10.0, ccp_weight=5.0)

# Styled image by CCPL
gen_image = model.ccpl_generate(
    content_image, style_image,
    alpha=1.0, interpolation= False, preserve_color= True
)
```

## Notebooks

- [**CCPL**: Tutorial]()
- [**CCPL**: Apply in EuroSAT]()

{{augmentare.methods.style_transfer.ccpl.CCPL}}

[Contrastive Coherence Preserving Loss for Versatile Style Transfer](https://arxiv.org/pdf/2207.04808.pdf) by Zijie Wu & al (2022).
