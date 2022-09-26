import torch.nn as nn
from models.modules.heads import SimCLRProjectionHead


class SimCLR(nn.Module):
    
    def __init__(self, backbone:nn.Module):
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x1, x2):
        
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(f1)

        f2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(f2)

        return z1, z2


