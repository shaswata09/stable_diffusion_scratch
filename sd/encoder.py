import torch
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from torch.nn import functional as F
from torch.nn import nn


class VAE_Encoder(nn.Seqential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
        )
