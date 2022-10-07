import torch
import torch.nn as nn
from models.modules.heads import SimCLRProjectionHead

class Disentangler_mutual(nn.Module):
    def __init__(self, cin):
        super(Disentangler_mutual, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x))) # [N, 1, H, W]

        fg_feats = x * ccam           # [N, C, H, W]
        bg_feats = x * (1-ccam)       # [N, C, H, W]

        return fg_feats, bg_feats, ccam

class OurSimCLR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OurSimCLR, self).__init__()
        
        self.backbone = backbone_q
        self.ac_head = Disentangler_mutual(512)
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x1, x2, is_train=True):
        #82.93
        y1 = self.backbone(x1) # (B, Cf, Hf, Wf)
        fg_feats1, bg_feats1, ccam1 = self.ac_head(y1)

        fg_z1 = self.projection_head(fg_feats1.flatten(start_dim=1))
        bg_z1 = self.projection_head(bg_feats1.flatten(start_dim=1))
        if not is_train:
            return fg_z1, ccam1

        y2 = self.backbone(x2)
        fg_feats2, bg_feats2, ccam2 = self.ac_head(y2)
        
        fg_z2= self.projection_head(fg_feats2.flatten(start_dim=1))
        bg_z2 = self.projection_head(bg_feats2.flatten(start_dim=1))

        z1 = torch.cat((fg_z1, bg_z1))
        z2 = torch.cat((fg_z2, bg_z2))

        return z1, z2
        
        # return fg_feats1, bg_feats1, ccam1, fg_feats2, bg_feats2, ccam2