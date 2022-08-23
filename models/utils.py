import torch
import torch.nn as nn

def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.
    
    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.
    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False

@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`
    Momentum encoders are a crucial component fo models such as MoCo or BYOL. 
    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1. - m)
