"""
GAN methods available
"""

from .dcgan import DCGAN, DCGANGenerator, DCGANDiscriminator
from .cgan import CGAN, CGANGenerator, CGANDiscriminator
from .cyclegan import CYCLEGAN, CYCLEGANGenerator, CYCLEGANDiscriminator
from .progan import PROGAN, PROGANGenerator, PROGANDiscriminator
from .cdcgan import CDCGAN, CDCGANGenerator, CDCGANDiscriminator
