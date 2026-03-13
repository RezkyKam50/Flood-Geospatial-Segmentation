import torch.nn as nn

class DeepLabWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sar, optical, elevation, water_occur):
        return self.model(optical)['out'] 