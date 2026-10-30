from functools import wraps

import torch
from torch import Tensor

# NOTE: this is a simplified version of all the metrics used in the paper.


def _check_shapes(x_fake: Tensor, x_real: Tensor) -> None:
    assert x_fake.ndim == 3
    assert x_fake.shape[1:] == x_real.shape[1:], f"{x_fake.shape} != {x_real.shape}"
    if x_fake.shape[0] != x_real.shape[0]:
        print(f"WARNING: different batch sizes: {x_fake.shape[0]} vs {x_real.shape[0]}")


def metric_guard(fn):
    @wraps(fn)
    def wrapped(x_fake, x_real, *args, **kwargs):
        _check_shapes(x_fake, x_real)
        c = torch.isfinite(x_fake).all().item() and torch.isfinite(x_real).all().item()
        if not c:
            return torch.tensor(float("inf"))
        return fn(x_fake, x_real, *args, **kwargs)

    return wrapped


def _expected_shortfall(x: Tensor, alpha: float = 0.05) -> Tensor:
    assert x.ndim == 3
    x = x.flatten(0, 1)
    q = x.quantile(alpha, dim=0)
    mask = x <= q
    return -(x * mask).sum(0) / mask.sum(0)


@metric_guard
def rel_expected_shortfall_diff(
    x_fake: Tensor, x_real: Tensor, alpha: float = 0.05
) -> Tensor:
    es_real = _expected_shortfall(x_real, alpha)
    es_fake = _expected_shortfall(x_fake, alpha)
    return ((es_fake - es_real).abs() / es_real.abs().clamp_min(1e-6)).mean()


def _acf(x: torch.Tensor, max_lag: int | None = 21) -> torch.Tensor:
    """
    Estimates the acf for each batch independently.
    Then averages over the batch dimension.
    """
    T = x.size(1)
    ml = T // 3  # suggested max lag
    max_lag = min(max_lag, ml) if max_lag is not None else ml
    if max_lag <= 0:
        return x.new_empty((0, x.size(2)))
    x = x - x.mean(1, keepdim=True)
    denom = x.square().sum(1).clamp_min(1e-12)
    out = []
    for k in range(1, max_lag + 1):
        num = (x[:, k:] * x[:, :-k]).sum(1)
        c = (num / denom).mean(0) * (T / (T - k))
        out.append(c)
    return torch.stack(out, 0)


@metric_guard
def acf_diff(x_fake: Tensor, x_real: Tensor, max_lag: int | None = 21) -> Tensor:
    diff = _acf(x_real, max_lag) - _acf(x_fake, max_lag)
    if diff.numel() == 0:
        return x_fake.new_zeros(())
    return diff.abs().mean()


def _corr_matrix(x: Tensor) -> Tensor:
    return torch.corrcoef(x.flatten(0, 1).T)


@metric_guard
def cross_corr_diff(x_fake: Tensor, x_real: Tensor) -> Tensor:
    if x_fake.size(-1) < 2:
        return x_fake.new_zeros(())
    corr_real = torch.nan_to_num(_corr_matrix(x_real), nan=0.0)
    corr_fake = torch.nan_to_num(_corr_matrix(x_fake), nan=0.0)
    diff = corr_real - corr_fake
    mask = ~torch.eye(diff.size(0), dtype=torch.bool, device=diff.device)
    return diff.abs()[mask].mean()


METRICS = dict(
    acf_diff=acf_diff,
    cross_corr_diff=cross_corr_diff,
    rel_es_diff=rel_expected_shortfall_diff,
)

TRAIN_METRICS = METRICS.copy()


@torch.no_grad()
def compute_metrics(
    x_real: Tensor,
    x_fake: Tensor,
    metrics: dict | None = None,
    prefix: str | None = None,
) -> dict:
    metrics = metrics if metrics is not None else METRICS
    x_fake, x_real = x_fake.cpu(), x_real.cpu()
    logs = {}
    for metric_name, metric_fn in metrics.items():
        mv = metric_fn(x_fake=x_fake, x_real=x_real)
        normalized_prefix = prefix.strip("/") if prefix is not None else ""
        key = f"{normalized_prefix}/{metric_name}" if normalized_prefix else metric_name
        if isinstance(mv, dict):
            for k, v in mv.items():
                logs[f"{key}_{k}"] = v
        else:
            logs[key] = mv.item()
    return logs
