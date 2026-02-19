import torch.nn as nn
import torch.nn.functional as F
import torch


class FFTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.out_channels = out_channels
         
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        
        self.fft_scale = nn.Parameter(torch.ones(1))

        self.projection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1) # pointwise linear transformation
        )

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):

        identity = x
         
        B, C, H, W = x.shape

        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32):
            x_ft = torch.fft.rfft2(x)
            
            out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, 
                                device=x.device, dtype=torch.cfloat)
            
            modes = min(self.modes, x_ft.shape[2], x_ft.shape[3])
            
            x_ft_modes = x_ft[:, :, :modes, :modes]
            x_real = x_ft_modes.real
            x_imag = x_ft_modes.imag
            
            weights = torch.complex(self.weights_real, self.weights_imag)
            weights = weights[:C, :, :modes, :modes]   
            
            out_ft_modes = torch.einsum('bixy,ioxy->boxy', 
                                    torch.complex(x_real, x_imag), 
                                    weights)
            
            out_ft[:, :, :modes, :modes] = out_ft_modes
            
            x_fourier = torch.fft.irfft2(out_ft, s=(H, W)) * self.fft_scale

        x = self.projection(x)
        
        return x_fourier + x + self.skip(identity)


class MultiscaleFFTEncoder(nn.Module):
    def __init__(self, cfg, n_channels=None):
        super().__init__()
        n_channels = len(cfg.DATASET.DEM_BANDS) if n_channels is None else n_channels
        topology = cfg.MODEL.TOPOLOGY
        f1 = topology[0]
          
        self.projection = nn.Sequential(
            nn.Conv2d(n_channels, f1, 3, padding=1),
            nn.BatchNorm2d(f1),
            nn.GELU()
        )
 
          
        self.fno1 = FFTBlock(f1, f1, modes=8)
        self.fno2 = FFTBlock(f1, f1, modes=4)
        self.fno3 = FFTBlock(f1, f1, modes=2)
        self.fno4 = FFTBlock(f1, f1, modes=1)

        self.residual_scaling = nn.Parameter(torch.ones(3))
         
        self.fusion_projection = nn.Sequential(
            nn.BatchNorm2d(4*f1),
            nn.GELU(),
            nn.Conv2d(4*f1, f1, 1),
            nn.BatchNorm2d(f1),
            nn.GELU()
        )
        
    def forward(self, x):
        x = self.projection(x)
        identity = x

        x1 = self.fno1(x)
        x2 = self.fno2(x1 + x * self.residual_scaling[0])  
        x3 = self.fno3(x2 + x1 * self.residual_scaling[1])
        x4 = self.fno4(x3 + x2 * self.residual_scaling[2])
        
        fusion = torch.cat([x1, x2, x3, x4], dim=1)
        fusion = self.fusion_projection(fusion)
        fusion = fusion + identity 

        return fusion, [x1, x2, x3, x4]


class HydraFQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MultiscaleFFTEncoder(cfg)
        
    def forward(self, x):
        fused_features, [x1, x2, x3, x4] = self.encoder(x)
         
        features = torch.stack([
            fused_features,  
            x1,  
            x2,  
            x3,  
            x4,  
        ], dim=1)
        
        return features