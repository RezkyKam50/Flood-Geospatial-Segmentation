import torch.nn.functional as F
import torch
import torch.nn as nn

import random

class PRCNPTNLayer(nn.Module):
    """
    Permanent Random Connectome Non-Parametric Transformation Network Layer
    
    Args:
        inch:       number of input channels
        outch:      number of output channels
        G:          number of filters per input channel (growth factor)
        CMP:        channel max pool size
        kernel_size: convolution kernel size
        padding:    convolution padding
        stride:     convolution stride
    """
    def __init__(self, inch, outch, G, CMP, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.G = G
        self.CMP = CMP
        self.expansion = G * inch  # total intermediate channels

        avg_pool_size = (inch * G) // (CMP * outch)
        assert avg_pool_size > 0, "avg_pool_size must be > 0; check inch, G, CMP, outch values"

        # Depthwise conv: each input channel gets G filters
        self.conv = nn.Conv2d(
            inch, 
            G * inch, 
            kernel_size=kernel_size, 
            groups=inch,          # depthwise
            padding=padding, 
            stride=stride,
            bias=False
        )
 
        # Channel max pool then average pool (across channel dim)
        self.max_pool = nn.MaxPool3d((CMP, 1, 1))
        self.avg_pool = nn.AvgPool3d((avg_pool_size, 1, 1))

        # Permanent random index (fixed at init, never changes)
        perm = list(range(self.expansion))
        random.shuffle(perm)
        self.register_buffer('perm_index', torch.LongTensor(perm))

    def forward(self, x):
        out = self.conv(x)                    # (B, G*inch, H, W)
        out = out[:, self.perm_index, :, :]   # permanent random shuffle
        out = self.max_pool(out)              # channel max pooling
        out = self.avg_pool(out)              # channel average pooling -> outch
        return out