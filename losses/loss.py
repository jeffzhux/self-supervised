
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.config import ConfigDict
from utils.dist import get_rank, gather 


class NTXentLoss(nn.Module):
    def __init__(self, temperature: int = 0.5) -> None:
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    

    def forward(self, p, z):
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
            return -F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

class CosineSimilarity(nn.Module):
    def __init__(self) -> None:
        super(NegativeCosineSimilarity, self).__init__()

    def forward(self, p, z, version: str='simplified'):
        if version == 'original':
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return (p * z).sum(dim=1).mean()
            
        elif version == 'simplified':
            # same thing, much faster. Scroll down, speed test in __main__
            return F.cosine_similarity(p, z, dim=-1).mean()
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
        
        return 0.5 * (self.criterion(p1, z2.detach(), self.version) + self.criterion(p2, z1.detach(), self.version))

class SimSiamLoss(nn.Module):
    def __init__(self, version: str = 'simplified') -> None:
        super(SimSiamLoss, self).__init__()
        self.version = version
        self.criterion = NegativeCosineSimilarity()

    def forward(
        self,
        p1: torch.Tensor,
        z1: torch.Tensor,
        p2: torch.Tensor,
        z2: torch.Tensor):
        
        return 0.5 * (self.criterion(p1, z2.detach(), self.version) + self.criterion(p2, z1.detach(), self.version)).mean()

class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float, version: str = 'simplified') -> None:
        super(SimCLRLoss, self).__init__()
        self.version = version
        self.temperature = temperature
        self.criterion = NTXentLoss(self.temperature)
        
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor):
        
        return self.criterion(z1, z2)

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

        return 0.5 * (self.criterion(p2, z1) + self.criterion(p1, z2))

class BarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230
        Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self, 
        lambda_param: float = 5e-3, 
        gather_distributed : bool = False
    ):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param: 
                Parameter for importance of redundancy reduction term. 
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are 
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow_(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss
'''
class OurLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230
        Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self, 
        lambda_param: float = 5e-3,
        gather_distributed : bool = False):

        super(OurLoss, self).__init__()
        self.alpha_param = 0.999
        self.gather_distributed = gather_distributed

        self.criterion = NegativeCosineSimilarity()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m, f'{n}, {m}'
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cross_corr(self, c):
        c_on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        c_off_diag = self.off_diagonal(c).pow_(2).sum()
        c_loss = c_on_diag + self.lambda_param * c_off_diag
        return c_loss

    def forward(
        self,
        fg_z1: torch.Tensor,
        fg_p1: torch.Tensor,
        fg_z2: torch.Tensor,
        fg_p2: torch.Tensor,
        c : torch.Tensor) -> torch.Tensor:
        fg_loss = 0.5 * (self.criterion(fg_p2, fg_z1.detach()) + self.criterion(fg_p1, fg_z2.detach())).mean()
        return self.alpha_param * fg_loss + (1 - self.alpha_param) * c
'''

class OurLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230
        Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self, 
        lambda_param: float = 5e-3,
        gather_distributed : bool = False):

        super(OurLoss, self).__init__()
        self.alpha_param = 0.999
        self.gather_distributed = gather_distributed

        self.criterion = NTXentLoss()

    def forward(
        self,
        fg_z1: torch.Tensor,
        fg_p1: torch.Tensor,
        fg_z2: torch.Tensor,
        fg_p2: torch.Tensor,
        bg_z1: torch.Tensor,
        bg_p1: torch.Tensor,
        bg_z2: torch.Tensor,
        bg_p2: torch.Tensor) -> torch.Tensor:

        z1 = torch.cat((fg_z1, bg_z1))
        p1 = torch.cat((fg_p1, bg_p1))
        z2 = torch.cat((fg_z2, bg_z2))
        p2 = torch.cat((fg_p2, bg_p2))

        loss = 0.5 * (self.criterion(p2, z1) + self.criterion(p1, z2)).mean()
        return loss