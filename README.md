<div align="center">
    <img src="docs/assets/Aug_lib.png" width="75%" alt="Augmentare Banner" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/python-3.9%2B-blue">
    </a>
    <!-- <a href="https://github.com/deel-ai/deel-with-augmentation/actions/workflows/python-lints.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/deel-with-augmentation/actions/workflows/python-lints.yml/badge.svg">
    </a>
    <a href="https://github.com/deel-ai/deel-with-augmentation/actions/workflows/python-tests-torch.yml">
        <img alt="Tests torch" src="https://github.com/deel-ai/deel-with-augmentation/actions/workflows/python-tests-torch.yml/badge.svg"> -->
    </a>
    <!-- <a href="https://github.com/deel-ai/xplique/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/deel-ai/xplique/actions/workflows/python-publish.yml/badge.svg">
    </a> -->
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<p align="center">
    <b>DEEL With Augmentation</b> is a repository which implements several ML techniques to generatively augment the data and that analyzes those methods from a Fairness perspective.
</p>

This repository includes two types of ML approaches to generate synthetic data to augment your training dataset: Generative Adversarial Networks (GANs) and Style Transfer (ST).

The main focus of this work is to analyze the ability of those methods to efficiently adress bias issues. More specifically, we wanted to check that augmenting a minority group (*i.e.* a group that is under represented) with those approaches could alleviate the drop in performance for the sensitive group that is usually observable when training a model on the unaugmented dataset. We made a number of experiments on the EuroSat dataset where the minority group is the satellite images with a *blue veil* and on the CelebA dataset on which multiple minority groups could be defined.

## 🧪 Experiment Results

- [**DEEL With Eurosat Bias**]() experiment's summary
- [**DEEL With CelebA Bias**]() experiment's summary

<details>
<summary><b>Experiments on Eurosat</b></summary>

**TODO: Add a table with all the experiments**
</details>

<details>
<summary><b>Experiments on CelebA</b></summary>

**TODO: Add a table with all the experiments**
</details>

## 🐍 The Augmentation Package

In this repository we packaged all the code used for our experiments to allow both: reproducibility and usability as a Python package of the augmentation methods.

<details>
<summary><b>Getting Started</b></summary>

**TODO: Explain the installation process**
</details>

<details>
<summary><b>Generative Adversarial Networks</b></summary>

The library includes a `gan` module where various GAN models are available. They all come with explanations, tutorials, and links to official articles:

| **GAN Method** | Source                                   | Tutorial                                                                     |
|:---------------|:-----------------------------------------|:----------------------------------------------------------------------------:|
| DCGAN          | [Paper](https://arxiv.org/abs/1511.06434)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()|


**TODO: Complete the table**

</details>

<details>
<summary><b>Style Transfer</b></summary>

The library includes a `style_transfer` module where various Style Transfer approaches are available. They all come with explanations, tutorials, and links to official articles:

| **ST Method**                      | Source                                   | Tutorial                                                                     |
|:-----------------------------------|:-----------------------------------------|:----------------------------------------------------------------------------:|
| Fourrier Domain Adaptation         | [Paper](https://arxiv.org/abs/2004.05498)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()|


**TODO: Complete the table**

</details>

<!-- <div class=figure>
  <p align="center" width="100%"> <img width="100%" src="/home/vuong.nguyen/vuong/augmentare/docs/assets/original_view.png"> </p>
</div> -->

<!-- ## Tutorials

We propose some hands-on tutorials to familiarize yourself with the library and its API:

- [**DCGAN**: Deep Convolutional GAN]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**CGAN**: Conditional GAN]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**CDCGAN**: Conditional Deep Convolutional GAN]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**CycleGAN**: Cycle GAN]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**ProGAN**: Progressive Growing of GANs]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

<p align="center" width="100%">
    <img width="65%" src="./docs/assets/fda_cityscape.png">
</p>

- [**AdaIN**: Adaptive Instance Normalization]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**FDA**: Fourier Domain Adaptation]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**NNST**: Neural Neighbor Style Transfer]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**StyleFlow**: StyleFlow For Content-Fixed Image to Image Translation]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub>

- [**CCPL**: Contrastive Coherence Preserving Loss for Versatile Style Transfer]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub> -->

<!-- ## Quick Start

**Augmentare** requires a version of python higher than 3.7 and several libraries including Pytorch and Numpy. Installation can be done using:

```bash
git clone https://github.com/deel-ai/augmentare.git
cd augmentare
python setup.py install
```

Now that Augmentare is installed.

### Generative Adversarial Networks Methods

```python
# Augmentare Imports
import augmentare
from augmentare.methods.gan import *
from augmentare.plots.plot_losses import plot_losses_gan
from augmentare.plots.plot_samples import plot_image
```

All GANs models share a common API. You can find out more about it [here]().

### Neural Style Transfer Methods

```python
# Augmentare Imports
import augmentare
from augmentare.methods.style_transfer import *
from augmentare.plots.plot_losses import plot_losses
```

All NST models share a common API. You can find out more about it [here]()

## What's Included

| **Generative Adversarial Networks** | Source| Year | Tutorial           |
| :--------------------- | :------------ | :----------------: | :----------------: |
| DCGAN | [Paper]()  | 2015 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CGAN              | [Paper](https://arxiv.org/abs/1610.02391) | 2014| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CDCGAN              | [Paper](https://arxiv.org/abs/1610.02391) | 2015| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CycleGAN              | [Paper](https://arxiv.org/abs/1610.02391) | 2017| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| ProGAN              | [Paper](https://arxiv.org/abs/1610.02391) | 2018| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |

| **Neural Style Transfer** | Source| Year | Tutorial           |
| :--------------------- | :------------ | :----------------: | :----------------: |
| AdaIN | [Paper]()  | 2017 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| FDA              | [Paper](https://arxiv.org/abs/1610.02391) | 2020| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| NNST              | [Paper](https://arxiv.org/abs/1610.02391) | 2022| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| StyleFlow              | [Paper](https://arxiv.org/abs/1610.02391) | 2022| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CCPL              | [Paper](https://arxiv.org/abs/1610.02391) | 2022| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |

## Experiments

| **Bias in EuroSAT** | Tutorial           |
| :--------------------- | :----------------: |
| Summary | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CGAN  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CDCGAN| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CycleGAN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| ProGAN| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |

<div class=figure>
  <p align="center" width="100%"> <img width="80%" src="/home/vuong.nguyen/vuong/augmentare/docs/assets/fda_eurosat.png"> </p>
</div>

| **Bias in CelebA** | Tutorial |
| :--------------------- | :----------------: |
| Summary | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| FDA   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| NNST  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| StyleFlow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |
| CCPL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() |

<div class=figure>
  <p align="center" width="100%"> <img width="90%" src="/home/vuong.nguyen/vuong/augmentare/docs/assets/new_male_gray_2_male_blond.png"> </p>
</div> -->

## 👀 See Also

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [LARD](https://github.com/deel-ai/LARD) Landing Approach Runway Detection (LARD) is a dataset of aerial front view images of runways designed for aircraft landing phase
- [PUNCC](https://github.com/deel-ai/puncc) Puncc (Predictive uncertainty calibration and conformalization) is an open-source Python library that integrates a collection of state-of-the-art conformal prediction algorithms and related techniques for regression and classification problems
- [OODEEL](https://github.com/deel-ai/oodeel) OODeel is a library that performs post-hoc deep OOD detection on already trained neural network image classifiers. The philosophy of the library is to favor quality over quantity and to foster easy adoption
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/deel_dark.png"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/deel_light.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="docs/assets/deel_dark.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creators

This repository was developed by [Vuong NGUYEN]() as part of his apprenticeship with the <a href="https://www.deel.ai/"> DEEL </a> Team under the supervision of [Lucas Hervier](https://github.com/lucashervier) and [Agustin PICARD](https://github.com/Agustin-Picard). He is currently a student in dual engineering degree ModIA program at INSA Toulouse and INP-ENSEEIHT supported by Artificial and Natural Intelligence Toulouse Institute (ANITI).

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
