import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sock.augmentations import AUGMENTATIONS

EPS = 1e-6

# --------------------------------helpers-----------------------------------------------


def register_or_fill_buffer(module: nn.Module, name: str, x: Tensor) -> None:
    if hasattr(module, name) and isinstance(getattr(module, name), Tensor):
        getattr(module, name).copy_(x)
    elif hasattr(module, name):
        delattr(module, name)
        module.register_buffer(name, x)
    else:
        module.register_buffer(name, x)


def normalize_kernel_(w: Tensor, center: bool = True) -> None:
    # normalization ensures that all kernels are equally strong a priori
    dims = (-2, -1)  # -2 is kernel width and -1 is kernel length
    if center:
        w.sub_(w.mean(dim=dims, keepdim=True))
    norm = w.abs().sum(dim=dims, keepdim=True).add_(1e-8)
    w.div_(norm)


# ------------------------------------Poolings------------------------------------------
# We discuss three choices for the pooling function in the paper.
# But there are many other plausible choices here.
# This includes concatenating different poolings.
#
# The first two poolings below are soft versions of Hydra's poolings.


def softmax(x: Tensor, dim: int, temp: float) -> Tensor:
    return F.softmax(x / temp, dim=dim)


def soft_count(x: Tensor, temp: float) -> Tensor:
    soft = softmax(x, dim=2, temp=temp)
    return soft.mean(dim=-1)


def neg_soft_weighted_mean(x: Tensor, temp: float) -> Tensor:
    soft = softmax(-x, dim=2, temp=temp)
    return (soft * x).mean(dim=-1)


def soft_std(x: Tensor, temp: float) -> Tensor:
    soft = softmax(x, dim=2, temp=temp)
    return soft.std(dim=-1)


POOLINGS = {
    "soft_count": soft_count,
    "neg_soft_weighted_mean": neg_soft_weighted_mean,
    "soft_std": soft_std,
}

# ------------------------------------SOCK----------------------------------------------


class SOCK(nn.Module):
    """
    Random convolution feature map.

    Input shape is `(bs, n_steps, num_channels)`.

    Minimal usage:
    1. `sock = SOCK(...)`
    2. `f_real = sock.forward_and_fit_scales(real_paths)`
        - this fits input and feature scales on real or training data paths.
    3. `f = sock(x)` to compute features for any input `x`.

    Notes:
    - If `resample()` is called, output feature scales become stale and must be re-fit
    by calling `forward_and_fit_scales()` on real data again.
    """

    def __init__(
        self,
        num_channels: int,
        n_steps: int,
        k: int = 8,  # kernels per group
        mix_dim: int = 256,
        kernel_len: int = 9,
        kernel_width: int = 2,
        scale_input: bool = True,
        scale_features: bool = True,
        # augmentation defaults to "diff" augmentation; pass `[]` for no augmentations
        augmentations: Sequence[str] = ("diff",),
        pooling: str = "soft_std",
        softmax_temp: float | tuple = 1e-1,
        max_dilation_exp: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.num_channels = num_channels  # the raw number of input channels
        self.n_steps = n_steps

        if len(augmentations) == 0:
            self.augs = None
        else:
            self.augs = nn.Sequential(*[AUGMENTATIONS[k]() for k in augmentations])
            for aug in self.augs:  # compute number of channels after augs
                num_channels, _ = aug.shape_transform(num_channels, 1)

        self.augmented_dim = num_channels
        self.mix_dim = mix_dim
        self.conv_dim = mix_dim
        if not (1 <= kernel_width <= mix_dim and mix_dim % kernel_width == 0):
            raise ValueError("invalid kernel_width/mix_dim")
        self.kernel_width = kernel_width
        self.k = k
        self.kernel_len = kernel_len

        pool = POOLINGS[pooling]
        self.pool = partial(pool, temp=softmax_temp) if "soft" in pooling else pool

        self.scale_input = scale_input
        self.scale_features = scale_features
        self._ft_scales_fresh = False

        # choice of dilation follows Hydra
        if max_dilation_exp is None:
            max_dilation_exp = max(0, int(math.log2((n_steps - 1) / (kernel_len - 1))))
        self.dilations = (2 ** torch.arange(int(max_dilation_exp) + 1)).int().tolist()

        self.proj = nn.Linear(num_channels, mix_dim, bias=False, device=device)

        self.convs = nn.ModuleList()
        for d in self.dilations:
            self.convs.append(
                nn.Conv1d(
                    self.conv_dim,
                    k * (self.conv_dim // kernel_width),
                    kernel_len,
                    padding="same",
                    dilation=d,
                    groups=self.conv_dim // kernel_width,
                    bias=False,
                    device=device,
                )
            )

        for p in self.parameters():
            p.requires_grad = False

        self.resample()

    @torch.no_grad()
    def resample(self) -> None:
        # 1 - resample the projection matrix
        self.proj.weight.normal_(std=(1 / self.augmented_dim**0.5))
        w_scl = self.proj.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        self.proj.weight.div_(w_scl)

        # 2 - resample the convolution kernels
        for conv in self.convs:
            conv.weight.normal_()
            normalize_kernel_(conv.weight)

        self._ft_scales_fresh = False

    @torch.no_grad()
    def fit_input_scales(self, x: Tensor) -> None:
        assert x.ndim == 3 and x.size(2) == self.num_channels
        x = self.augs(x) if self.augs is not None else x
        m = x.mean(dim=(0, 1), keepdim=True)
        s = x.std(dim=(0, 1), keepdim=True).clamp_min(EPS)
        self._register_input_scales(m, s)

    @torch.no_grad()
    def _register_input_scales(self, mean: Tensor, scl: Tensor) -> None:
        assert mean.ndim == 3 and mean.size(0) == 1 and mean.size(1) == 1
        assert scl.ndim == 3 and scl.size(0) == 1 and scl.size(1) == 1
        register_or_fill_buffer(self, "input_mean", mean)
        register_or_fill_buffer(self, "input_scl", scl)

    @torch.no_grad()
    def fit_feature_scales(self, x: Tensor) -> None:
        assert x.ndim == 2
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True).clamp(min=1e-4, max=1e4)
        self._register_feature_scales(m, s)

    @torch.no_grad()
    def _register_feature_scales(self, mean: Tensor, scl: Tensor) -> None:
        register_or_fill_buffer(self, "ft_mean", mean)
        register_or_fill_buffer(self, "ft_scl", scl)
        self._ft_scales_fresh = True

    def _scale_augmented_input(self, x: Tensor) -> Tensor:
        if not self.scale_input:
            return x
        assert hasattr(
            self, "input_mean"
        ), "You set scale_input=True. Call fit_input_scales() first on real data to compute input scales. Only then can you call forward()."
        return (x - self.input_mean) / self.input_scl

    def _scale_output(self, x: Tensor) -> Tensor:
        if not self.scale_features:
            return x
        assert self._ft_scales_fresh
        return (x - self.ft_mean) / self.ft_scl

    def _forward(self, x: Tensor, scale: bool = True) -> Tensor:
        assert x.ndim == 3
        assert x.size(1) == self.n_steps
        assert x.size(2) == self.num_channels

        x = x if self.augs is None else self.augs(x)
        x = self._scale_augmented_input(x)
        x = self.proj(x).permute(0, 2, 1)

        out = []
        x = x.contiguous()
        for conv in self.convs:
            z = conv(x)  # (bs, n_groups*k, n_steps) n_groups = mix_dim // kernel_width
            z = z.view(x.size(0), -1, self.k, x.size(-1))
            f = self.pool(z)  # (bs, n_groups, k, n_steps)
            out.append(f.view(f.size(0), -1))  # (bs, n_groups * k)
        out = torch.cat(out, dim=-1)
        return self._scale_output(out) if scale else out

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape `(bs, n_steps, num_channels)`
        Returns:
            Tensor: shape `(bs, num_features)`
        """
        return self._forward(x, scale=True)

    @torch.no_grad()
    def forward_and_fit_scales(self, x: Tensor) -> Tensor:
        """
        Calibration pass on a reference set (typically real/train paths).

        If input scaling is enabled and missing, input scales are fitted from `x`.
        Then unscaled features are computed, output feature scales are fitted, and
        scaled features for `x` are returned.
        """
        if self.scale_input and not hasattr(self, "input_mean"):
            self.fit_input_scales(x)
        f = self._forward(x, scale=False)
        self.fit_feature_scales(f)
        return self._scale_output(f)
