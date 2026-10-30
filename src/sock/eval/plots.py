from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sock.eval.metrics import _acf


def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(-1)
    raise ValueError(
        f"Expected tensor with shape [B,T] or [B,T,C], got {tuple(x.shape)}"
    )


def _as_3d_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    return _ensure_3d(_to_tensor(x))


def create_acf_plot(
    x_real: np.ndarray | torch.Tensor,
    x_fake: np.ndarray | torch.Tensor,
    out_dir: str | Path,
    *,
    filename: str = "acf.png",
    max_lag: int = 14,
    channel_idx: int = 0,
) -> Path:
    x_real_t = _as_3d_tensor(x_real)
    x_fake_t = _as_3d_tensor(x_fake)

    acf_real = _acf(x_real_t, max_lag=max_lag)[:, channel_idx].float().cpu()
    acf_fake = _acf(x_fake_t, max_lag=max_lag)[:, channel_idx].float().cpu()

    lags = torch.arange(1, acf_real.size(0) + 1, dtype=torch.float32)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.grid(axis="y", alpha=0.3)

    offset = 0.16
    ax.vlines((lags - offset).numpy(), 0.0, acf_real.numpy(), color="black", lw=1.1)
    ax.vlines((lags + offset).numpy(), 0.0, acf_fake.numpy(), color="tab:blue", lw=1.1)
    ax.plot(
        (lags - offset).numpy(),
        acf_real.numpy(),
        ".",
        color="black",
        ms=4.5,
        label="Real (val)",
    )
    ax.plot(
        (lags + offset).numpy(),
        acf_fake.numpy(),
        ".",
        color="tab:blue",
        ms=4.5,
        label="Fake",
    )

    ax.set_xlim(0.4, max_lag + 0.6)
    ax.set_xticks(list(range(1, max_lag + 1)))
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(frameon=False)

    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Saved ACF plot: {out_path.resolve()}")
    return out_path


def create_marginals_plot(
    x_real_val: np.ndarray | torch.Tensor,
    x_fake: np.ndarray | torch.Tensor,
    out_dir: str | Path,
    *,
    x_real_train: np.ndarray | torch.Tensor | None = None,
    filename: str = "marginals.png",
    bins: int = 80,
    q_lo: float = 0.001,
    q_hi: float = 0.999,
) -> Path:
    x_real_val_t = _as_3d_tensor(x_real_val)
    x_fake_t = _as_3d_tensor(x_fake)
    x_real_train_t = _as_3d_tensor(x_real_train) if x_real_train is not None else None

    _, _, channels = x_real_val_t.shape
    fig, axes = plt.subplots(1, channels, figsize=(5.4 * channels, 3.8), squeeze=False)
    axes = axes[0]

    for c in range(channels):
        val_vals = x_real_val_t[:, :, c].reshape(-1).float()
        fake_vals = x_fake_t[:, :, c].reshape(-1).float()

        val_vals = val_vals[torch.isfinite(val_vals)]
        fake_vals = fake_vals[torch.isfinite(fake_vals)]

        pooled = [val_vals, fake_vals]

        train_vals = None
        if x_real_train_t is not None:
            train_vals = x_real_train_t[:, :, c].reshape(-1).float()
            train_vals = train_vals[torch.isfinite(train_vals)]
            pooled.append(train_vals)

        both = torch.cat(pooled, dim=0) if pooled else torch.empty(0)

        if both.numel() == 0:
            lo, hi = -1.0, 1.0
        else:
            qs = torch.quantile(both, torch.tensor([q_lo, q_hi], device=both.device))
            lo, hi = float(qs[0].item()), float(qs[1].item())
            if not lo < hi:
                lo, hi = lo - 1.0, hi + 1.0
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad

        ax = axes[c]
        if train_vals is not None:
            ax.hist(
                train_vals.cpu().numpy(),
                bins=bins,
                range=(lo, hi),
                density=True,
                alpha=0.22,
                color="0.6",
                label="Real (train)",
            )

        ax.hist(
            val_vals.cpu().numpy(),
            bins=bins,
            range=(lo, hi),
            density=True,
            alpha=0.28,
            color="black",
            label="Real (val)",
        )
        ax.hist(
            fake_vals.cpu().numpy(),
            bins=bins,
            range=(lo, hi),
            density=True,
            histtype="step",
            linewidth=1.3,
            color="tab:blue",
            label="Fake",
        )
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"Channel {c}")
        if c == 0:
            ax.set_ylabel("Density")
            ax.legend(frameon=False)

    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Saved marginals plot: {out_path.resolve()}")
    return out_path


def create_paths_plot(
    x_real: np.ndarray,
    x_fake: np.ndarray,
    out_dir: str | Path,
    *,
    filename: str = "real_vs_fake_paths.png",
    max_paths: int = 10,
    cmap_name: str = "plasma_r",
) -> Path:
    x_real_t = _as_3d_tensor(x_real)
    x_fake_t = _as_3d_tensor(x_fake)

    if x_real_t.shape[1:] != x_fake_t.shape[1:]:
        raise ValueError(
            f"Shape mismatch: {tuple(x_real_t.shape)} vs {tuple(x_fake_t.shape)}"
        )

    bs = min(x_real_t.shape[0], x_fake_t.shape[0])
    n_steps = x_real_t.shape[1]
    d = x_real_t.shape[2]
    n_plot = min(bs, max_paths)

    time = np.arange(n_steps)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_plot))

    rng = np.random.default_rng()
    idx_real = rng.choice(x_real_t.shape[0], size=n_plot, replace=False)
    idx_fake = rng.choice(x_fake_t.shape[0], size=n_plot, replace=False)
    x_real_sub = x_real_t[idx_real].cpu().numpy()
    x_fake_sub = x_fake_t[idx_fake].cpu().numpy()

    fig, axes = plt.subplots(
        nrows=d, ncols=2, figsize=(10, 2 * d), sharex=True, sharey="row"
    )
    axes = np.atleast_2d(axes).reshape(d, 2)

    for i in range(d):
        ax_r, ax_f = axes[i]
        for j in range(n_plot):
            ax_r.plot(time, x_real_sub[j, :, i], color=colors[j], alpha=0.8)
            ax_f.plot(time, x_fake_sub[j, :, i], color=colors[j], alpha=0.8)
        ax_r.set_ylabel(f"Channel {i}")
        if i == 0:
            ax_r.set_title("Real (val)")
            ax_f.set_title("Fake")

    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")

    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved paths plot: {out_path.resolve()}")
    return out_path


def plot_real_vs_fake_paths(
    x_real: np.ndarray,
    x_fake: np.ndarray,
    save_path: str,
    x_real_train: np.ndarray | None = None,
    max_paths: int = 10,
    cmap_name: str = "plasma_r",
) -> None:
    out_path = Path(save_path)
    create_paths_plot(
        x_real=x_real,
        x_fake=x_fake,
        out_dir=out_path.parent,
        filename=out_path.name,
        max_paths=max_paths,
        cmap_name=cmap_name,
    )

    stem = out_path.stem
    create_marginals_plot(
        x_real_val=x_real,
        x_fake=x_fake,
        x_real_train=x_real_train,
        out_dir=out_path.parent,
        filename=f"{stem}_marginals.png",
    )
    create_acf_plot(
        x_real=x_real,
        x_fake=x_fake,
        out_dir=out_path.parent,
        filename=f"{stem}_acf.png",
    )
