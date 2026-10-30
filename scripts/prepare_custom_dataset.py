"""
This is an example for how to prepare a custom dataset for training a generative model.

Our generative pipeline is setup to support two types of datasets:
    1) synthetic dataset: train/path [H_train, d], val/paths [J, H_val, d]
    2) real dataset:      train/path [H_train, d], val/path  [H_val, d]
where H is the path length.

For both real and synthetic datasets,
we assume that the training data consists of a *single path*.

For synthetic data, we assume that we have multiple validation paths (J).
For real data, we assume that we only have a single validation path.

Modify this code to prepare your own dataset.
"""

import os

import numpy as np
import torch

from sock.paths import data_dir

# Dummy constants for our dummy datasets.
SEED = 0
H = 512
J = 16
D = 3
SYNTH_NAME = "custom_synth_example"
REAL_NAME = "custom_real_example"


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Replace these with your own data loading logic in real usage.
    # Example:
    # x_train = np.load("/path/to/train.npy")
    # x_val_paths = np.load("/path/to/val_paths.npy")
    x_train = rng.normal(0.0, 1.0, size=(H, D)).astype(np.float32)
    x_val_paths = rng.normal(0.0, 1.0, size=(J, H, D)).astype(np.float32)

    synth_out_dir = data_dir() / SYNTH_NAME
    os.makedirs(synth_out_dir, exist_ok=True)
    synth_out_file = synth_out_dir / f"{SYNTH_NAME}_0.pt"
    torch.save(
        {
            "train/path": torch.from_numpy(x_train),
            "val/paths": torch.from_numpy(x_val_paths),
        },
        synth_out_file,
    )

    # Real dataset has a single out-of-sample path (no leading J dimension).
    x_val_path = rng.normal(0.0, 1.0, size=(H, D)).astype(np.float32)
    real_out_dir = data_dir() / REAL_NAME
    os.makedirs(real_out_dir, exist_ok=True)
    real_out_file = real_out_dir / f"{REAL_NAME}.pt"
    torch.save(
        {
            "train/path": torch.from_numpy(x_train),
            "val/path": torch.from_numpy(x_val_path),
        },
        real_out_file,
    )

    print(f"Saved synth example: {synth_out_file}")
    print(f"Saved real example:  {real_out_file}")


if __name__ == "__main__":
    main()
