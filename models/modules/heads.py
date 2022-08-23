
from ast import List
from typing import Optional, Tuple
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer, non_linearity_layer).
    Examples:
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """
    def __init__(self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]) -> None:
        super(ProjectionHead, self).__init__()
        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.

        "The projection MLP (in f) has BN applied to each fully-connected (fc)
        layer, including its output fc. Its output fc has no ReLU. The hidden fc is
        2048-d. This MLP has 3 layers." [0]

        [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """
    def __init__(
        self, 
        input_dim:int = 512,
        hidden_dim:int = 2048,
        output_dim:int = 2048):
        super(SimSiamProjectionHead, self).__init__([
            (input_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , output_dim, nn.BatchNorm1d(hidden_dim), None)
        ])

class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam.
    
    "The prediction MLP (h) has BN applied to its hidden fc layers. Its output
    fc does not have BN (...) or ReLU. This MLP has 2 layers." [0]
    
    [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 2048):
        super(SimSiamPredictionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, output_dim, None, None)
        ])

class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]
    
    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733

    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 4096,
                 output_dim: int = 256):
        super(BYOLProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, output_dim, None, None),
        ])

class BYOLPredictionHead(ProjectionHead):
    """Prediction head used for BYOL.
    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]
    [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 4096,
                 output_dim: int = 256):
        super(BYOLPredictionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, output_dim, None, None),
        ])


class NNCLRProjectionHead(ProjectionHead):
    def __init__(
        self,
        input_dim:int = 2048,
        hidden_dim:int = 2048,
        output_dim:int = 256) -> None:
        super(NNCLRProjectionHead, self).__init__(
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
        )

class NNCLRPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 4096,
        output_dim: int = 256) -> None:

        super(NNCLRPredictionHead, self).__init__(
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
                (hidden_dim, output_dim, None, None),
            )