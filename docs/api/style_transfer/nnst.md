# Neural Neighbor Style Transfer (NNST)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial]() |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source]() |
📰 [Paper]()

## NETWORK ARCHITECTURE : NNST

<img src="/home/vuong.nguyen/vuong/augmentare/docs/assets/nnst.png" alt="Picture" style="display: block; margin: 0 auto" />

The fast and slow variants of their method, NNST-D, and NNST-Opt, only differ in step 4; mapping from the target features to image pixels. This simplified diagram omits several details for clarity, namely: they apply steps 1-4 at multiple scales, coarse to fine; they repeat steps 1-4 several times at the finest scale; and they only apply step 5 once (optionally) at the very end.

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *

# Create NNST method
model = NNST(content_img, style_img, device)

# Styled image by NNST
gen_image = model.nnst_generate(
    max_scales=5, alpha=0.75,
    content_loss=False, flip_aug=False,
    zero_init = False, dont_colorize=False
)
```

## Notebooks

- [**NNST**: Tutorial]()
- [**NNST**: Apply in EuroSAT]()

{{augmentare.methods.style_transfer.nnst.NNST}}

[Neural Neighbor Style Transfer](https://arxiv.org/pdf/2203.13215v1.pdf) by Nick Kolkin & al (2022).
