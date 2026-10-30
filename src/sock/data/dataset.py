import torch
from torch import Tensor

from sock.paths import data_dir


def bootstrap_segments(x: Tensor, k: int, stride: int = 1) -> Tensor:
    assert x.ndim == 2
    windows = x.unfold(dimension=0, size=k, step=stride)
    return windows.permute(0, 2, 1).contiguous()


def batch_bootstrap_segments(x: torch.Tensor, k: int, stride: int = 1) -> torch.Tensor:
    assert x.ndim == 3
    d = x.size(-1)
    windows = x.unfold(dimension=1, size=k, step=stride)
    windows = windows.permute(0, 1, 3, 2)
    return windows.reshape(-1, k, d).contiguous()


def sample_segments(x: Tensor, k: int, n_samples: int) -> Tensor:
    assert x.ndim == 2
    max_start = x.size(0) - k
    starts = torch.randint(0, max_start + 1, (n_samples,), device=x.device)
    offsets = torch.arange(k, device=x.device).unsqueeze(0)
    idx = starts.unsqueeze(1) + offsets
    return x[idx]


def batch_sample_segments(x: Tensor, k: int, n_samples: int, stride: int = 1) -> Tensor:
    assert x.ndim == 3
    bs, T, _ = x.shape
    starts_all = torch.arange(0, T - k + 1, stride, device=x.device)
    M = starts_all.numel()

    g = torch.randint(bs * M, (n_samples,), device=x.device)
    p_idx = g // M
    m_idx = g % M
    starts = starts_all[m_idx]

    offs = torch.arange(k, device=x.device)[None, :]
    t_idx = starts[:, None] + offs
    p_idx2 = p_idx[:, None].expand_as(t_idx)

    return x[p_idx2, t_idx, :]


class SinglePathDataset:
    def __init__(
        self,
        x_raw: Tensor,
        n_steps: int,
        n_cond_steps: int = 0,
        stride: int = 1,
        train: bool = True,
    ):
        assert x_raw.ndim == 2
        self.x_raw = x_raw
        self.num_channels = x_raw.size(1)
        self.n_steps = n_steps
        self.n_cond_steps = n_cond_steps
        self.stride = stride

        if train:
            self._split_into_train_paths()

    def _split_into_train_paths(self):
        segment_length = self.n_steps + self.n_cond_steps
        self.paths = bootstrap_segments(self.x_raw, segment_length, self.stride)

    def sample_x0_and_idx(self, bs: int) -> tuple:
        idx = torch.randint(0, self.paths.size(0), (bs,), device=self.paths.device)
        segs = self.paths[idx]
        return segs[:, : self.n_cond_steps], idx

    def sample_x0_and_segments(self, bs: int) -> tuple:
        x = sample_segments(self.x_raw, self.n_cond_steps + self.n_steps, bs)
        return x[:, : self.n_cond_steps], x[:, self.n_cond_steps :]

    def get_all_x0_and_segments(self) -> tuple:
        segs = bootstrap_segments(
            self.x_raw, self.n_cond_steps + self.n_steps, self.stride
        )
        return segs[:, : self.n_cond_steps], segs[:, self.n_cond_steps :]


class MultiPathDataset:
    """Dataset for when data consists of multiple paths.
    - we use this only for validation purposes on synthetic datasets
    - there we sample multiple continuations of the single training path
    - since we can generate as much validation data as we like, we don't use
    every window from each validation path; instead `stride = n_steps` by default
    """

    def __init__(
        self,
        paths: Tensor,
        n_steps: int,
        n_cond_steps: int,
        stride: int | None = None,
    ):
        assert paths.ndim == 3
        self.paths = paths
        self.n_steps = n_steps
        self.n_cond_steps = n_cond_steps
        self.stride = stride or n_steps
        self.seg_len = n_steps + n_cond_steps

    def sample_x0_and_segments(self, bs: int) -> tuple:
        segs = batch_sample_segments(self.paths, self.seg_len, bs, stride=self.stride)
        return segs[:, : self.n_cond_steps], segs[:, self.n_cond_steps :]

    def get_all_x0_and_segments(self):
        segs = batch_bootstrap_segments(self.paths, self.seg_len, self.stride)
        return segs[:, : self.n_cond_steps], segs[:, self.n_cond_steps :]


def setup_data(
    dataset_name: str,
    dataset_type: str,
    n_steps: int | tuple,  # if int, then train and val same length (this is default)
    n_cond_steps: int,
    device: torch.device,
    train_data_stride: int = 1,
    dataset_id: int | None = None,
):

    if isinstance(n_steps, tuple):
        n_steps_train, n_steps_val = n_steps
    else:
        n_steps_train = n_steps_val = n_steps

    fname = (
        f"{dataset_name}.pt"
        if dataset_type == "real"
        else f"{dataset_name}_{dataset_id}.pt"
    )
    f = data_dir() / dataset_name / fname
    data = torch.load(f, map_location="cpu")

    stride_train = min(train_data_stride, n_steps_train)
    stride_val = 1

    if dataset_type == "synth":
        assert all(k in data for k in ["train/path", "val/paths"])
        path = data["train/path"].to(torch.float32).to(device)
        val_paths = data["val/paths"].to(torch.float32)  # keep on cpu

        train_dataset = SinglePathDataset(
            path,
            n_steps=n_steps_train,
            n_cond_steps=n_cond_steps,
            stride=stride_train,
        )

        val_dataset = MultiPathDataset(
            val_paths, n_steps=n_steps_val, n_cond_steps=n_cond_steps
        )

        return train_dataset, val_dataset

    assert all(k in data for k in ["train/path", "val/path"])
    path = data["train/path"].to(torch.float32).to(device)
    val_path = data["val/path"].to(torch.float32)  # keep on cpu

    train_dataset = SinglePathDataset(
        path,
        n_steps=n_steps_train,
        n_cond_steps=n_cond_steps,
        stride=stride_train,
    )
    val_dataset = SinglePathDataset(
        val_path,
        n_steps=n_steps_val,
        n_cond_steps=n_cond_steps,
        stride=stride_val,
        train=False,
    )

    return train_dataset, val_dataset
