"""BYOL Model"""
import torch
import torch.nn as nn
from models.modules.memory_bank import NNMemoryBankModule
from models.modules.heads import BarlowTwinsProjectionHead

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam

class OUR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OUR, self).__init__()
        
        self.backbone = backbone_q
        self.ac_head = Disentangler(512)
        self.projection_head = BarlowTwinsProjectionHead(512,2048,2048)

    def forward(self, x1, x2):

        y1 = self.backbone(x1) # (B, Cf, Hf, Wf)
        fg_feats1, bg_feats1, ccam1 = self.ac_head(y1)
        fg_feats1 = self.projection_head(fg_feats1)
        bg_feats1 = self.projection_head(bg_feats1)

        y2 = self.backbone(x2)
        fg_feats2, bg_feats2, ccam2 = self.ac_head(y2)
        fg_feats2 = self.projection_head(fg_feats2)
        bg_feats2 = self.projection_head(bg_feats2)

        return fg_feats1, bg_feats1, fg_feats2, bg_feats2
        # return fg_feats1, bg_feats1, ccam1, fg_feats2, bg_feats2, ccam2