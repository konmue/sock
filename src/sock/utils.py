import torch
from torch import Tensor, nn


class StandardScaler(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def fit(self, x: Tensor) -> "StandardScaler":
        # compute per-feature mean & std over all dims except last
        dims = tuple(range(x.dim() - 1))
        m = x.mean(dim=dims)
        v = x.var(dim=dims, unbiased=False)
        s = torch.sqrt(v + self.eps)
        self.register_buffer("mean_", m)
        self.register_buffer("scale_", s)
        return self

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean_.to(x.device)) / self.scale_.to(x.device)

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x * self.scale_ + self.mean_


class TrainLogger:
    def __init__(
        self,
        *,
        use_wandb: bool,
        log_dir: str,
        **wandb_init_kwargs,
    ):
        self.use_wandb = use_wandb
        self._wandb = None
        self._run = None
        self._jsonl_path = None
        self.run_dir = log_dir

        if use_wandb:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                dir=log_dir,
                settings=wandb.Settings(init_timeout=120),
                **wandb_init_kwargs,
            )
            self.run_dir = self._run.dir
        else:
            from datetime import datetime
            from pathlib import Path

            run_dir = Path(log_dir) / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=False)
            self.run_dir = str(run_dir)
            self._jsonl_path = run_dir / "metrics.jsonl"

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.use_wandb:
            self._wandb.log(metrics, step=step)
            return
        import json

        payload = dict(metrics)
        if step is not None:
            payload["step"] = step
        with self._jsonl_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

    def save(self, path: str) -> None:
        if self.use_wandb:
            self._wandb.save(path)

    def finish(self) -> None:
        if self.use_wandb:
            self._wandb.finish()


@torch.no_grad()
def chunked_inference_gen(x0, generator, n_steps, device, chunk_size: int = 4096):
    generator.to(device)
    x_fake = []
    for i in range(0, x0.size(0), chunk_size):
        x0_c = x0[i : i + chunk_size].to(device)
        x_fake_c = generator(x0_c, n_steps=n_steps)
        x_fake.append(x_fake_c.cpu())
    return torch.cat(x_fake, dim=0)


def seed_all(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def parse_str_or_list_input(x: str) -> list[str]:
    if x.lower() == "none":
        return []
    return x.split(",") if "," in x else [x]


def lr_schedule(step: int, up: int, down: int, total: int) -> float:
    if up > 0 and step < up:
        return (step + 1) / up
    elif down <= 0 or step < total - down:
        return 1.0
    else:
        return (total - step) / down
