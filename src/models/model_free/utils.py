import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ObsSNP:
    """
    Standard observation bundle for SNP/Model-Free models.
    Contains spatial coordinates (xs, ys), measured values (values), and a mask.
    """
    xs: torch.Tensor
    ys: torch.Tensor
    values: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        return ObsSNP(
            xs=self.xs.to(device),
            ys=self.ys.to(device),
            values=self.values.to(device) if self.values is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None
        )

