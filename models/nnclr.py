"""NNCLR Model"""
import torch.nn as nn
from models.modules.memory_bank import NNMemoryBankModule
from models.modules.heads import NNCLRProjectionHead, NNCLRPredictionHead


class NNCLR(nn.Module):
    
    def __init__(self, backbone_q: nn.Module):
        super(NNCLR, self).__init__()
        
        self.backbone = backbone_q
        self.projection_head = NNCLRProjectionHead(512,1024,128)
        self.prediction_head = NNCLRPredictionHead(128,1024,128)

        self.memory_bank = NNMemoryBankModule()
    def forward(self, x1, x2):

        y1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(y1)
        p1 = self.prediction_head(z1)

        y2 = self.backbone(x2).flatten(start_dim=1)
        z2 = self.projection_head(y2)
        p2 = self.prediction_head(z2)

        z1 = self.memory_bank(z1.detach(), update=False)
        z2 = self.memory_bank(z2.detach(), update=True)
        return p1, z1.detach(), p2, z2.detach() 