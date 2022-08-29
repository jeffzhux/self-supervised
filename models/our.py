"""BYOL Model"""

import torch.nn as nn
import torch.nn.functional as F
from models.modules.heads import SimSiamProjectionHead, SimSiamPredictionHead

class OUR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OUR, self).__init__()
        
        self.backbone = backbone_q
        self.projection_head = SimSiamProjectionHead(512, 512, 512)
        self.prediction_head = SimSiamPredictionHead(512, 128, 512)


    def forward(self, x1, x2):

        y1 = self.backbone(x1).flatten(start_dim=1) # (B, Cf, Hf, Wf) -> (B, C, 1, 1) -> (B, C)
        z1 = self.projection_head(y1)
        p1 = self.prediction_head(z1)

        y2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(y2)
        p2 = self.prediction_head(z2)


        return p1, z1.detach(), p2, z2.detach()