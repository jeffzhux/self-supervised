
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dist import get_rank, gather 


class NTXentLoss(nn.Module):
    def __init__(self, temperature: int = 0.5) -> None:
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    

    def forward(self, z, p):
        z = F.normalize(z, dim=-1)
        p = F.normalize(p, dim=-1)

        p = gather(p)

        logits = z @ p.T / self.temperature
        rank = get_rank()

        z_dim = z.size(0)
        labels = torch.arange(z_dim * rank, z_dim * (rank + 1), device=p.device)
        loss = F.cross_entropy(logits, labels)
        return loss

class NegativeCosineSimilarity(nn.Module):
    def __init__(self) -> None:
        super(NegativeCosineSimilarity, self).__init__()

    def forward(self, p, z, version: str='simplified'):
        if version == 'original':
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
            
        elif version == 'simplified':
            # same thing, much faster. Scroll down, speed test in __main__
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

class BYOLLoss(nn.Module):
    def __init__(self, version: str = 'simplified') -> None:
        super(BYOLLoss, self).__init__()
        self.version = version
        self.criterion = NegativeCosineSimilarity()

    def forward(
        self,
        p1: torch.Tensor,
        z1: torch.Tensor,
        p2: torch.Tensor,
        z2: torch.Tensor):
        
        return 0.5 * (self.criterion(p1, z2, self.version) + self.criterion(p2, z1, self.version)).mean()

class NNCLRLoss(nn.Module):
    def __init__(self) -> None:
        super(NNCLRLoss, self).__init__()
        self.criterion = NTXentLoss()
    def forward(
        self,
        p1: torch.Tensor,
        z1: torch.Tensor,
        p2: torch.Tensor,
        z2: torch.Tensor):

        return 0.5 * (self.criterion(z1, p2) + self.criterion(z2, p1))

