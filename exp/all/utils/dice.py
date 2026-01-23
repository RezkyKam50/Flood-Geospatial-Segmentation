import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.device = device
        
    def forward(self, output, target):
        # Apply softmax to the output logits and use the class probabilities directly
        output = torch.softmax(output, dim=1)
        
        # Flatten the tensors
        output = output[:, 1].flatten()  # Assuming class 1 is the relevant one for binary dice
        target = target.flatten().float()
        
        # Ignore the '255' values in the target
        no_ignore = target.ne(255).to(self.device)
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        
        # Compute the intersection and union for Dice calculation
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        dice = (2 * intersection + 1e-7) / (union + 1e-7)
        
        return 1 - dice
    
class DiceLoss2(nn.Module):
    def __init__(self, device, epsilon=1e-7):
        super(DiceLoss2, self).__init__()
        self.device = device
        self.epsilon = epsilon
    
    def forward(self, output, target):
        # Apply softmax to get probabilities
        output = torch.softmax(output, dim=1)
        
        # Get foreground (class 1) probabilities
        p_foreground = output[:, 1].flatten()
        
        # Get background (class 0) probabilities  
        p_background = output[:, 0].flatten()
        
        # Flatten target
        target = target.flatten().float()
        
        # Ignore the '255' values in the target
        no_ignore = target.ne(255).to(self.device)
        p_foreground = p_foreground.masked_select(no_ignore)
        p_background = p_background.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        
        # Compute foreground term: [Σpn*rn + ε] / [Σ(pn + rn) + ε]
        foreground_intersection = torch.sum(p_foreground * target)
        foreground_sum = torch.sum(p_foreground + target)
        foreground_dice = (foreground_intersection + self.epsilon) / (foreground_sum + self.epsilon)
        
        # Compute background term: [Σ(1-pn)*(1-rn) + ε] / [Σ(2-pn-rn) + ε]
        background_intersection = torch.sum(p_background * (1 - target))
        background_sum = torch.sum(2 - p_foreground - target)
        background_dice = (background_intersection + self.epsilon) / (background_sum + self.epsilon)
        
        # DL2 = 1 - (foreground_dice + background_dice)
        dl2 = 1 - (foreground_dice + background_dice)
        
        return dl2