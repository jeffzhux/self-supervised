import torch
import torch.nn as nn
from models.modules.heads import SimSiamProjectionHead, SimSiamPredictionHead


class SimSiam(nn.Module):
    """Implementation of SimSiam[0] network
    Recommended loss: :py:class:`lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss`
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    
    Attributes:
        backbone:
            Backbone model to extract features from images.

    Examples:
        >>> # single input, single output
        >>> p0, z0 = model(x0)
        >>> 
        >>> # single input, single output
        >>> p1, z1 = model(x1) 
        >>> loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0)).mean()
    """
    def __init__(self, backbone:nn.Module):
        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead()
        self.prediction_head = SimSiamPredictionHead()

    def forward(self, x):
        f = self.backbone(x)
        z = self.projection_head(f)
        p = self.prediction_head(z)

        return p, z.detach()


