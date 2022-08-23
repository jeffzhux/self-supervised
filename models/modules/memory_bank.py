
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBankModule(nn.Module):
    """Memory bank implementation
    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if 
    desired.
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: int = 2 ** 16):
        >>>         super(MyLossFunction, self).__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: torch.Tensor,
        >>>                 labels: torch.Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples
    """

    def __init__(self, size: int = 2 ** 16):

        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f'Illegal memory bank size {size}, must be non-negative.'
            raise ValueError(msg)

        self.size = size

        self.bank = None
        self.bank_ptr = None
    
    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """Initialize the memory bank if it's empty
        Args:
            dim:
                The dimension of the which are stored in the bank.
        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        self.bank = torch.randn(dim, self.size)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.LongTensor([0])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one
        Args:
            batch:
                The latest batch of keys to add to the memory bank.
        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[:self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr:ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(self,
                output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False):
        """Query memory bank for additional negative samples
        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.
        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.
        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank is None:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank

class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>>
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
    """
    def __init__(self, size:int = 2 ** 12) -> None:
        super(NNMemoryBankModule, self).__init__(size)
        
    
    def forward(self, input: torch.Tensor, update: bool = False):
        input, bank = super(NNMemoryBankModule, self).forward(input, update=update)
        bank = bank.to(input.device).t()

        output_normed = F.normalize(input, dim=1)
        bank_normed = F.normalize(bank, dim=1)

        similarity_matrix = torch.einsum("nd, md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(bank, dim=0, index = index_nearest_neighbours)
        
        return nearest_neighbours