import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------------------------Main augmentations used for SOCK----------------------------


class DifferencingAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def shape_transform(num_channels: int, seq_len: int) -> tuple[int, int]:
        return num_channels * 2, seq_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape `(batch_size, seq_len, num_channels)`.
        Returns:
            Tensor: shape `(batch_size, seq_len, num_channels * 2)`.
        """
        dx = F.pad(torch.diff(x, dim=1), (0, 0, 1, 0))
        return torch.cat([x, dx], dim=-1)


class CumsumAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def shape_transform(num_channels: int, seq_len: int) -> tuple[int, int]:
        return num_channels * 2, seq_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape `(batch_size, seq_len, num_channels)`.
        Returns:
            Tensor: shape `(batch_size, seq_len, num_channels * 2)`.
        """
        cumx = torch.cumsum(x, dim=1)
        return torch.cat([x, cumx], dim=-1)


class PosNegAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def shape_transform(num_channels: int, seq_len: int) -> tuple[int, int]:
        return num_channels * 3, seq_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape `(batch_size, seq_len, num_channels)`.
        Returns:
            Tensor: shape `(batch_size, seq_len, num_channels * 3)`.
        """
        pos = torch.relu(x)
        neg = -torch.relu(-x)
        return torch.cat([x, pos, neg], dim=-1)


AUGMENTATIONS = {
    "diff": DifferencingAugmentation,
    "cumsum": CumsumAugmentation,
    "posneg": PosNegAugmentation,
}
