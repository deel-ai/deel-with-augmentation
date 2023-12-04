# Fourier Domain Adaptation (FDA)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial]() |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source]() |
📰 [Paper]()

## NETWORK ARCHITECTURE : FDA
Simplified domain adaptation via style transfer thanks to the Fourier transformation. The FDA does not need deep networks for style transfer and adversarial training.

<img src="/home/vuong.nguyen/vuong/augmentare/docs/assets/fda_view.png" alt="Picture" style="display: block; margin: 0 auto" />

The scheme of the proposed Fourier domain adaptation method:
    <ul> <li> <span style="color:gold"> Step 1: </span> <span> Apply FFT to source and target images. </span> </li>
    <li> <span style="color:gold"> Step 2: </span> <span> Replace the low frequency part of the source amplitude with that of the target. </span> </li>
    <li> <span style="color:gold"> Step 3: </span> <span> Apply the inverse FFT to the modified source spectrum. </span> </li> </ul>

## Example

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *

# Create FDA method
model = FDA(im_src, im_trg)

# Styled image by FDA
src_in_trg = model.fda_source_to_target(beta=0.01)
```

## Notebooks

- [**FDA**: Tutorial]()
- [**FDA**: Apply in EuroSAT]()

{{augmentare.methods.style_transfer.fda.FDA}}

[Fourier Domain Adaptation for Semantic Segmentation](https://arxiv.org/pdf/2004.05498.pdf) by Yanchao Yang & Stefano Soatto (2020).
