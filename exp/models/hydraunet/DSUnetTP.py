from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from models.prithvi_segmenter import PritviSegmenter
 
# Dual Stream UNet3+  

# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py
class DSUNet3P(nn.Module):
    def __init__(self, cfg, use_prithvi=None):
        super(DSUNet3P, self).__init__()
        assert (cfg.DATASET.MODE == 'fusion')
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        
        # sentinel-1 unet stream
        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS)
        s1_in = n_s1_bands   
        self.s1_stream = UNet3Plus(cfg, n_channels=s1_in, n_classes=out, enable_outc=False)
        self.n_s1_bands = n_s1_bands

        # sentinel-2 unet stream
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        s2_in = n_s2_bands  
        self.s2_stream = UNet3Plus(cfg, n_channels=s2_in, n_classes=out, enable_outc=False)
        self.n_s2_bands = n_s2_bands

        self.use_prithvi = use_prithvi
        # prithvi
        if self.use_prithvi:
            self.prithvi = PritviSegmenter(
                weights_path=cfg.MODEL.PRITHVI_PATH,
                device="cuda" if torch.cuda.is_available() else "cpu",
                prithvi_encoder_size=cfg.MODEL.TOPOLOGY[-1],
                output_channels=out
            ) # prithvi encoder 
            out_dim = 2 * cfg.MODEL.TOPOLOGY[0] + 2 # N channels x Topo First idx 
        else:
            out_dim = 2 * cfg.MODEL.TOPOLOGY[0] # N channels x Topo First idx 

        self.out_conv = OutConv(out_dim, out)

    def change_prithvi_trainability(self, trainable):
        if self.use_prithvi:
            self.prithvi.change_prithvi_trainability(trainable)

    def change_s1_trainability(self, trainable):
        for param in self.s1_stream.parameters():
            param.requires_grad = trainable

    def change_s2_trainability(self, trainable):
        for param in self.s2_stream.parameters():
            param.requires_grad = trainable

    def forward(self, s1_img, s2_img, dem_img):

        del dem_img

        '''Late fusion scheme'''
        s1_feature = self.s1_stream(s1_img)
        s2_feature = self.s2_stream(s2_img)
 

        if self.use_prithvi:
            prithvi_features = self.prithvi(s2_img)
            fusion = torch.cat((s1_feature, s2_feature, prithvi_features), dim=1) # 2 ch + 2 ch prithvi
        else:
            fusion = torch.cat((s1_feature, s2_feature), dim=1) # 2 ch

        out = self.out_conv(fusion)  
        return out
    

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if act:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Sequential(
            conv_block(in_c, out_c),
            conv_block(out_c, out_c)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1) # spectral norm, see https://arxiv.org/abs/1802.05957

    def forward(self, x):
        return self.conv(x)



# Ref https://github.com/nikhilroxtomar/UNET-3-plus-Implementation-in-TensorFlow-and-PyTorch/blob/main/pytorch/1-unet3plus.py
# 
class UNet3Plus(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, enable_outc=True):
        super().__init__()
        
        self._cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
         
        if hasattr(cfg.MODEL, 'TOPOLOGY'):
            topology = cfg.MODEL.TOPOLOGY
 
            f1, f2, f3, f4, f5 = topology
        else:
            # Legacy UNet3+ Topo
            f1, f2, f3, f4, f5 = 64, 128, 256, 512, 1024
        
        self.enable_outc = enable_outc
         
        self.e1 = encoder_block(n_channels, f1)
        self.e2 = encoder_block(f1, f2)
        self.e3 = encoder_block(f2, f3)
        self.e4 = encoder_block(f3, f4)
 
        self.e5 = nn.Sequential(
            conv_block(f4, f5),
            conv_block(f5, f5)
        )
 
        self.reduction_channels = f1
 
        self.e1_d4 = conv_block(f1, self.reduction_channels)
        self.e2_d4 = conv_block(f2, self.reduction_channels)
        self.e3_d4 = conv_block(f3, self.reduction_channels)
        self.e4_d4 = conv_block(f4, self.reduction_channels)
        self.e5_d4 = conv_block(f5, self.reduction_channels)
        self.d4 = conv_block(self.reduction_channels * 5, self.reduction_channels)
 
        self.e1_d3 = conv_block(f1, self.reduction_channels)
        self.e2_d3 = conv_block(f2, self.reduction_channels)
        self.e3_d3 = conv_block(f3, self.reduction_channels)
        self.e4_d3 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d3 = conv_block(f5, self.reduction_channels)
        self.d3 = conv_block(self.reduction_channels * 5, self.reduction_channels)
 
        self.e1_d2 = conv_block(f1, self.reduction_channels)
        self.e2_d2 = conv_block(f2, self.reduction_channels)
        self.e3_d2 = conv_block(self.reduction_channels, self.reduction_channels)  # from d3
        self.e4_d2 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d2 = conv_block(f5, self.reduction_channels)
        self.d2 = conv_block(self.reduction_channels * 5, self.reduction_channels)
 
        self.e1_d1 = conv_block(f1, self.reduction_channels)
        self.e2_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d2
        self.e3_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d3
        self.e4_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d1 = conv_block(f5, self.reduction_channels)
        self.d1 = conv_block(self.reduction_channels * 5, self.reduction_channels)
 
        if enable_outc:
            self.y1 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
        else:
            self.y1 = nn.Identity()

    def forward(self, inputs):
 
        e1, p1 = self.e1(inputs)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)

        """ Bottleneck """
        e5 = self.e5(p4)
 
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)
        d4 = self.d4(d4)
 
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)
 
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)
 
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)
 
        y1 = self.y1(d1)

        return y1


