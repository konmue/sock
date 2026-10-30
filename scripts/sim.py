"""
Using this script you can simulate the synthetic datasets that we use in the paper.

NOTE: we do not include the FGN dataset here, which we used in the paper.
Simulating the fractional gaussian noise is a bit more complicated.
We used the package `stocastic` for this:
https://github.com/crflynn/stochastic/blob/90ffba84523751dd205e17c50c84be9904c13330/stochastic/processes/noise/fractional_gaussian_noise.py
"""

import os

import numpy as np
import torch
import typer
import yaml

from sock.utils import StandardScaler

app = typer.Typer(pretty_exceptions_enable=False)


def as_t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


@app.command()
def main(seed: int, config_path: str):
    """
    Simulate synthetic data.
    Args:
        seed (int): Random seed.
        config_path (str): Path to configuration file.
    """
    # we use the term `path`` for long time series and `segment` for the short ones

    dataset = os.path.basename(config_path).split(".")[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_real_data_points = config.get("n_real_data_points", 2048)
    if n_real_data_points != 2048:
        print(f"NOTE: you are using n_real_data_points={n_real_data_points}.")
    n_val_paths = 2048
    val_length = config.get("val_length", 2048)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data")
    dataset_dir = os.path.join(data_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    fpath = os.path.join(dataset_dir, f"{dataset}_{seed}.pt")

    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    data = dict()

    if dataset.startswith("heston"):
        from sock.data.heston import HESTON_DEFAULT_PARAMS, HestonSimulator

        simulator = HestonSimulator(HESTON_DEFAULT_PARAMS, n_real_data_points, rng=rng)

        s_raw, v_raw = simulator.simulate(1)
        x_real_path = simulator.transform(s_raw, v_raw).squeeze(0)
        v_raw = v_raw.squeeze(0)
        x_real_path = as_t(x_real_path)
        x_real_path = scaler.fit_transform(x_real_path)

        data["train/path"] = x_real_path

        n_steps_val = val_length
        vsim = HestonSimulator(HESTON_DEFAULT_PARAMS, n_steps_val, rng=rng)
        v0 = v_raw[-1]
        vpaths = vsim.simulate_transformed(n_val_paths, v0=v0)
        data["val/paths"] = scaler.transform(as_t(vpaths))

    elif dataset.startswith("var1"):
        from sock.data.var import VAR1Simulator

        simulator = VAR1Simulator(
            n_steps=n_real_data_points, rng=rng, **config["simulator_configs"]
        )
        x_raw_path = simulator.simulate(1)
        x_real_path = x_raw_path.squeeze(0)
        x_real_path = as_t(x_real_path)
        x_real_path = scaler.fit_transform(x_real_path)
        x_raw_path = as_t(x_raw_path).squeeze(0)

        data["train/path"] = x_real_path

        n_steps_val = val_length

        vsim = VAR1Simulator(
            n_steps=n_steps_val, rng=rng, **config["simulator_configs"]
        )
        x0 = np.repeat(x_raw_path[-1:], n_val_paths, axis=0)
        vpaths = vsim.simulate(n_val_paths, x0=x0)
        data["val/paths"] = scaler.transform(as_t(vpaths))

    elif dataset.startswith("varp"):
        from sock.data.varp import VARpSimulator, tgh_transform

        r = config["simulator_configs"]["r"]
        omega = config["simulator_configs"]["omega"]
        num_channels = config["simulator_configs"]["dims"]
        g = config["simulator_configs"].get("g", 0.0)
        h = config["simulator_configs"].get("h", 0.0)
        apply_tgh = (g != 0.0) or (h != 0.0)

        phi1 = 2 * r * np.cos(omega * np.pi)
        phi2 = -r * r
        l1 = config["simulator_configs"]["first_lag"]
        l2 = l1 * 2
        lags = {l1: np.eye(num_channels) * phi1, l2: np.eye(num_channels) * phi2}
        max_lag = max(lags.keys())
        simulator = VARpSimulator(n_steps=n_real_data_points, rng=rng, lags=lags)

        x_raw_path = simulator.simulate(1)
        x_raw_path = as_t(x_raw_path).squeeze(0)
        x_real_path = tgh_transform(x_raw_path, g, h) if apply_tgh else x_raw_path
        x_real_path = scaler.fit_transform(x_real_path)

        data["train/path"] = x_real_path

        vsim = VARpSimulator(
            n_steps=val_length, rng=rng, lags=lags, cov_trace=0.2, corr_weight=0.5
        )
        x0 = np.repeat(x_raw_path[-max_lag:][np.newaxis, :], n_val_paths, axis=0)
        vpaths = vsim.simulate(n_val_paths, x0=x0)
        vpaths = as_t(vpaths)
        vpaths = tgh_transform(vpaths, g, h) if apply_tgh else vpaths
        vpaths = scaler.transform(vpaths)
        data["val/paths"] = vpaths

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    torch.save(data, fpath)


if __name__ == "__main__":
    app()
