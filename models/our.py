"""BYOL Model"""

import torch.nn as nn
from models.modules.memory_bank import NNMemoryBankModule
from models.modules.heads import BarlowTwinsProjectionHead

class OUR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OUR, self).__init__()
        
        self.backbone = backbone_q
        self.backbone = backbone_q
        self.projection_head = BarlowTwinsProjectionHead(512,2048,2048)

        self.memory_bank = NNMemoryBankModule()
    def forward(self, x1, x2):

        y1 = self.backbone(x1).flatten(start_dim=1) # (B, Cf, Hf, Wf) -> (B, C, 1, 1) -> (B, C)
        z1 = self.projection_head(y1)

        y2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(y2)

        z1 = self.memory_bank(z1.detach(), update=False)
        z2 = self.memory_bank(z2.detach(), update=True)
        return z1.detach(), z2.detach()