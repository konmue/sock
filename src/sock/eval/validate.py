import gc
import os

import torch
from torch import Tensor

from sock.data.dataset import MultiPathDataset, SinglePathDataset
from sock.eval.metrics import TRAIN_METRICS, compute_metrics
from sock.eval.plots import plot_real_vs_fake_paths
from sock.utils import chunked_inference_gen


@torch.no_grad()
def generate_fake_paths_on_train(dataset, generator, n_paths: int):
    x0, x_real = dataset.sample_x0_and_segments(n_paths)
    x_fake = generator(x0, n_steps=x_real.size(1))
    x_real = torch.cat((x0, x_real), dim=1)
    return x_real, x_fake


@torch.inference_mode()
def _validate_on_train(train_dataset, generator):
    """
    NOTE: validation on train may use a different n_steps than standard
    validation on val.
    """
    x_real, x_fake = generate_fake_paths_on_train(
        train_dataset, generator, n_paths=4096
    )
    logs = compute_metrics(x_real, x_fake, metrics=TRAIN_METRICS, prefix="train")

    del x_real, x_fake
    gc.collect()
    return logs


@torch.inference_mode()
def validation_step(
    generator,
    train_dataset,
    val_dataset,
    val_path_len: int,
) -> dict:
    generator.eval()
    device = next(generator.parameters()).device

    logs = _validate_on_train(train_dataset, generator)

    x0, x_real = val_dataset.sample_x0_and_segments(8192)
    x_fake = chunked_inference_gen(x0, generator, x_real.size(1), device)
    x_real = torch.cat((x0, x_real), dim=1)

    assert x_fake.size(1) == x_real.size(1)
    assert x_fake.size(1) == val_path_len

    vm = compute_metrics(x_real, x_fake, metrics=TRAIN_METRICS, prefix="val/")
    logs.update(vm)

    del x_fake
    gc.collect()
    generator.train()

    return logs


def final_validation(
    train_dataset: SinglePathDataset,
    val_dataset: SinglePathDataset | MultiPathDataset,
    plots_path: str | None,
    val_path_len: int,
    generator: torch.nn.Module | None = None,
    fake_samples: Tensor | None = None,
    save_dir: str | None = None,
) -> dict:
    assert sum(v is not None for v in [generator, fake_samples]) == 1

    print("===========================================================================")
    print("Running final validation...")
    print("===========================================================================")

    logs = {}
    x0, x_real = val_dataset.get_all_x0_and_segments()
    n_steps = x_real.size(1)
    x_real = torch.cat((x0, x_real), dim=1).cpu()

    if fake_samples is None:
        device = next(generator.parameters()).device
        x_fake = chunked_inference_gen(x0, generator, n_steps, device)
    else:
        x_fake = fake_samples.cpu()

    if save_dir is not None:  # save the fake_samples to file
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, "fake_samples.pt")
        torch.save(x_fake, fpath)
        print(f"Saved fake samples to {fpath}")
    x0_train, x_real_train = train_dataset.get_all_x0_and_segments()
    x_real_train = torch.cat((x0_train, x_real_train), dim=1)

    assert x_real.size(1) == x_fake.size(1)
    assert x_real.size(1) == val_path_len

    x_real_train = x_real_train.cpu().numpy()
    if plots_path is not None:
        plot_real_vs_fake_paths(
            x_real.numpy(),
            x_fake.numpy(),
            save_path=os.path.join(plots_path, "real_vs_fake_paths.png"),
            x_real_train=x_real_train,
        )

    metrics_logs = compute_metrics(
        x_real=x_real, x_fake=x_fake, prefix="big_val/"
    )
    logs.update(metrics_logs)

    print("===========================================================================")
    print("Final validation finished.")
    print("===========================================================================")

    return logs
