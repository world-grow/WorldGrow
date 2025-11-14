import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super(SparseLinear, self).__init__(in_features, out_features, bias, device=device)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
