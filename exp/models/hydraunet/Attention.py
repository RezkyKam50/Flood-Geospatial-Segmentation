import torch.nn.functional as F
import torch
import torch.nn as nn
from models.hydraunet.GhostNet import hard_sigmoid, _make_divisible


"""

CBAM

"""

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

"""

SE

"""

class CrossModalSqueezeExcite(nn.Module):
    def __init__(self, aux_chs, s_chs, se_ratio=0.25, divisor=4):
        super(CrossModalSqueezeExcite, self).__init__()
        self.gate_fn = hard_sigmoid
        reduced_chs = _make_divisible(s_chs * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(aux_chs, reduced_chs, 1, bias=True)   # aux -> reduced
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, s_chs, 1, bias=True)    # reduced -> s1 channels

    def forward(self, s_feat, aux):
        x_se = self.avg_pool(aux)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return s_feat * self.gate_fn(x_se)
    
'''

Sensor Fusion Wrapper

'''

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim, num_groups=None, method="CBAM"):
        super().__init__()
        
        self.method = method

        if self.method == "CBAM":
            self.ca1 = CBAM(embed_dim * 2)
            self.ca2 = CBAM(embed_dim * 2)


        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
 
         
    def forward(self, f1, f2):
        C = f1.shape[1]

        f1_cat = torch.cat([f1, f2], dim=1)   # (B, 2C, H, W) — f1 perspective
        f2_cat = torch.cat([f2, f1], dim=1)   # (B, 2C, H, W) — f2 perspective

        f1_out = self.norm1(f1 + self.ca1(f1_cat)[:, :C, :, :])
        f2_out = self.norm2(f2 + self.ca2(f2_cat)[:, :C, :, :])

        return f1_out, f2_out
    
