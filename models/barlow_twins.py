"""BarlowTwin Model"""

import torch.nn as nn
from models.modules.heads import BarlowTwinsProjectionHead


class BarlowTwins(nn.Module):
    
    def __init__(self, backbone_q: nn.Module):
        super(BarlowTwins, self).__init__()
        
        self.backbone = backbone_q
        self.projection_head = BarlowTwinsProjectionHead(512,2048,2048)

    def forward(self, x1, x2):

        y1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(y1)

        y2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(y2)

        return z1, z2