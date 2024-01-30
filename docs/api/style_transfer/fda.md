# Fourier Domain Adaptation (FDA)

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1qbsmllY18RyHGNNaCn5zPNcRu6Pe_S_t?authuser=3) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/deel-with-augmentation/blob/main/augmentare/methods/style_transfer/fda.py) |
📰 [Paper](https://arxiv.org/abs/2004.05498)

## NETWORK ARCHITECTURE : FDA
Simplified domain adaptation via style transfer thanks to the Fourier transformation. The FDA does not need deep networks for style transfer and adversarial training.

<img src="../images/fda.png" width="100%" alt="Picture" style="display: block; margin: 0 auto" />

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

- [**FDA**: Tutorial](https://colab.research.google.com/drive/1qbsmllY18RyHGNNaCn5zPNcRu6Pe_S_t?authuser=3)
- [**FDA**: Apply in EuroSAT](https://colab.research.google.com/drive/1LCLevUaAERtI-K6xYSHCAuJ00lAPjFBe?authuser=3)

{{augmentare.methods.style_transfer.fda.FDA}}

[Fourier Domain Adaptation for Semantic Segmentation](https://arxiv.org/pdf/2004.05498.pdf) by Yanchao Yang & Stefano Soatto (2020).
