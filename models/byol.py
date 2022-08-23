"""BYOL Model"""

import torch
import torch.nn as nn
from models.modules.heads import BYOLProjectionHead, BYOLPredictionHead
from models.utils import deactivate_requires_grad, update_momentum


class BYOL(nn.Module):
    def __init__(self, backbone_q: nn.Module, backbone_k: nn.Module):
        super(BYOL, self).__init__()
        
        self.backbone = backbone_q
        self.projection_head = BYOLProjectionHead(512,512,128)
        self.prediction_head = BYOLPredictionHead(128,512,128)

        self.backbone_momentum = backbone_k
        self.projection_head_momentum = BYOLProjectionHead(512,512,128)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x1, x2):
        with torch.no_grad():
            update_momentum(self.backbone, self.backbone_momentum, m=0.99)
            update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)

        y1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(y1)
        p1 = self.prediction_head(z1)

        y_m1 = self.backbone_momentum(x1).flatten(start_dim=1)
        z_m1 = self.projection_head_momentum(y_m1)

        y2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(y2)
        p2 = self.prediction_head(z2)

        y_m2 = self.backbone_momentum(x2).flatten(start_dim=1)
        z_m2 = self.projection_head_momentum(y_m2)
        return p1, z_m1.detach(), p2, z_m2.detach()