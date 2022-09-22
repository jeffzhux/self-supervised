"""BYOL Model"""
import torch
import torch.nn as nn
from models.modules.heads import SimSiamProjectionHead, SimSiamPredictionHead

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

'''
class OUR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OUR, self).__init__()
        
        self.backbone = backbone_q
        self.ac_head = Disentangler_mutual(512)
        self.projection_head = SimSiamProjectionHead(512, 512, 512)
        self.prediction_head = SimSiamPredictionHead(512, 128, 512)
    def neg_coeff_constraint(self, x, mask, pred_foreground, pred_background):
        # computes 2-C(F,B), which is equivalent to -C(F,B) as a loss term
        # we have 2-C(F,B) = H(F|B)/H(F) + H(B|F)/H(B), where:
        # H(F|B) = H(X*M | X*(1-M)) = ||M * (X - phi(X*(1-M), 1-M))||_1 = ||M * (X - pred_foreground)||_1
        # H(B|F) = H(X*(1-M) | X*M) = ||(1-M) * (X - phi(X*M, M))||_1 = ||(1-M) * (X - pred_background)||_1
        # H(F) = ||1-M||
        # H(B) = ||M||

        H_foreground_given_background = (mask*torch.abs(x-pred_foreground)).mean(1).sum((1,2))
        H_background_given_foreground = ((1-mask)*torch.abs(x-pred_background)).mean(1).sum((1,2))
        
        H_foreground = mask.sum((1,2,3))
        H_background = (1-mask).sum((1,2,3))

        C_foreground_given_background = - H_foreground_given_background / H_foreground.clamp_(1e-6)
        C_background_given_foreground = - H_background_given_foreground / H_background.clamp_(1e-6)
        C = C_foreground_given_background + C_background_given_foreground

        return (-C).add_(-1)

    def forward(self, x1, x2, is_train=True):
        #82.03 acc

        y1 = self.backbone(x1) # (B, Cf, Hf, Wf)
        fg_feats1, bg_feats1, ccam1 = self.ac_head(y1)

        fg_z1 = self.projection_head(fg_feats1.flatten(start_dim=1))
        fg_p1 = self.prediction_head(fg_z1)

        if not is_train:
            return fg_z1, fg_p1, ccam1

        y2 = self.backbone(x2)
        fg_feats2, bg_feats2, ccam2 = self.ac_head(y2)
        fg_z2= self.projection_head(fg_feats2.flatten(start_dim=1))
        fg_p2 = self.prediction_head(fg_z2)

        c = 0.5 * (
            self.neg_coeff_constraint(y1, ccam1, fg_feats1, bg_feats1).mean() \
            + self.neg_coeff_constraint(y2, ccam2, fg_feats2, bg_feats2).mean()
        )
        return fg_z1, fg_p1, fg_z2, fg_p2, c 
        
        # return fg_feats1, bg_feats1, ccam1, fg_feats2, bg_feats2, ccam2
'''

class OUR(nn.Module):
    def __init__(self, backbone_q: nn.Module):
        super(OUR, self).__init__()
        
        self.backbone = backbone_q
        self.ac_head = Disentangler_mutual(512)
        self.projection_head = SimSiamProjectionHead(512, 512, 512)
        self.prediction_head = SimSiamPredictionHead(512, 128, 512)

    def forward(self, x1, x2, is_train=True):
        #82.93
        y1 = self.backbone(x1) # (B, Cf, Hf, Wf)
        fg_feats1, bg_feats1, ccam1 = self.ac_head(y1)

        fg_z1 = self.projection_head(fg_feats1.flatten(start_dim=1))
        fg_p1 = self.prediction_head(fg_z1)

        bg_z1 = self.projection_head(bg_feats1.flatten(start_dim=1))
        bg_p1 = self.prediction_head(bg_z1)
        if not is_train:
            return fg_z1, fg_p1, ccam1

        y2 = self.backbone(x2)
        fg_feats2, bg_feats2, ccam2 = self.ac_head(y2)
        
        fg_z2= self.projection_head(fg_feats2.flatten(start_dim=1))
        fg_p2 = self.prediction_head(fg_z2)

        bg_z2 = self.projection_head(bg_feats2.flatten(start_dim=1))
        bg_p2 = self.prediction_head(bg_z2)
        return fg_z1, fg_p1, fg_z2, fg_p2, bg_z1, bg_p1, bg_z2, bg_p2
        
        # return fg_feats1, bg_feats1, ccam1, fg_feats2, bg_feats2, ccam2