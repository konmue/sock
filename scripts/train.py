"""
This is the script for training generative models.
All default args are the ones used in the paper.
"""

import os
from functools import partial

import torch
import torch.nn.functional as F
import typer
from tqdm import trange

from sock.data.dataset import setup_data
from sock.eval.validate import final_validation, validation_step
from sock.generators import Generator
from sock.paths import logs_dir
from sock.sock import SOCK
from sock.utils import (
    TrainLogger,
    lr_schedule,
    parse_str_or_list_input,
    seed_all,
)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    dataset: str = "heston",
    dataset_type: str | None = None,  # None or ["real", "synth"]
    dataset_id: int = 0,
    torch_seed: int | None = None,
    n_gen_steps: int = 64,  # =: T, length of the generated path; excl. conditioning segment
    n_cond_steps: int = 5,  # =: q, length of the conditioning segment
    train_data_stride: int = 1,
    # configs for training
    opt_steps: int = 100_000,
    val_every: int = 2000,
    bs: int = 1024,
    bias_correction: bool = False,
    gen_lr: float = 3e-4,
    warm_up_frac: float = 0.05,
    warm_down_frac: float = 0.7,
    grad_clip: float | None = 1.0,
    weight_decay: float = 0.01,
    # configs for generator
    gen_hidden_dim: int = 128,
    # discriminator configs
    resample_d_every: int = 100,  # =: R, resample frequency
    rconv_k: int = 8,
    kernel_len: int = 9,
    kernel_width: int = 2,
    pooling: str = "soft_std",
    mix_dim: int = 256,  # note: just pass in int
    augs: str = "diff",  # NOTE only used by sock, otherwise is hardcoded for other models
    softmax_temp: float = 0.1,
    compile_discriminator: bool = True,
    cudnn_benchmark: bool = True,
    # configs for logging
    log_dir: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    tag: str | None = None,
    save_paths: bool = False,
):
    # ----- Setting up some configs, before we log them to wandb ----------------------

    # To validate, we condition on a segment of length n_cond_steps
    # and generate n_gen_steps after that.
    # we evaluate on the whole *concatenated* segment of length:
    val_path_len = n_gen_steps + n_cond_steps

    augs = parse_str_or_list_input(augs)
    if dataset_type is None:
        if any(k in dataset for k in ["heston", "var1", "varp", "fgn"]):
            dataset_type = "synth"
        else:
            dataset_type = "real"

    # LR schedule: linear warm up -> constant -> linear warm down
    assert warm_up_frac + warm_down_frac <= 1.0
    assert all((warm_up_frac >= 0, warm_down_frac >= 0))
    warm_up = int(opt_steps * warm_up_frac)
    warm_down = int(opt_steps * warm_down_frac)

    config = locals().copy()  # save configs into dict for wandb logging

    if torch_seed is not None:
        seed_all(torch_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # this enables autotuning of the conv algo
        torch.backends.cudnn.benchmark = cudnn_benchmark

    log_dir = log_dir or str(logs_dir())
    os.makedirs(log_dir, exist_ok=True)

    train_dataset, val_dataset = setup_data(
        dataset_name=dataset,
        dataset_type=dataset_type,
        n_steps=n_gen_steps,
        n_cond_steps=n_cond_steps,
        device=device,
        train_data_stride=train_data_stride,
        dataset_id=dataset_id,
    )
    num_channels = train_dataset.num_channels

    # -------------------- Setting up the discriminator --------------------------------
    discriminator = SOCK(
        num_channels=num_channels,
        k=rconv_k,
        mix_dim=mix_dim,
        kernel_len=kernel_len,
        kernel_width=kernel_width,
        scale_features=True,
        augmentations=augs,
        pooling=pooling,
        softmax_temp=softmax_temp,
        device=device,
        n_steps=n_gen_steps + n_cond_steps,
    )
    if compile_discriminator:
        discriminator = torch.compile(discriminator)

    def resample_now(step: int) -> bool:
        if resample_d_every is None:
            return False
        return (step > 0) and (step % resample_d_every == 0)

    # -------------------- Setting up the generator ------------------------------------
    generator = Generator(
        dim=num_channels, hidden_dim=gen_hidden_dim, n_cond_steps=n_cond_steps
    )
    generator.to(device, non_blocking=True)

    # -------------------- Optimization, logging, helpers ------------------------------
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=gen_lr, weight_decay=weight_decay
    )
    get_lr_decay = partial(lr_schedule, up=warm_up, down=warm_down, total=opt_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_decay)

    logger = TrainLogger(
        use_wandb=wandb_project is not None,
        log_dir=log_dir,
        entity=wandb_entity,
        project=wandb_project,
        config=config,
        tags=[tag] if tag is not None else [],
    )

    f_real = None
    v_step = partial(
        validation_step,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_path_len=val_path_len,
    )

    print("===========================================================================")
    print("Starting training...")
    print("===========================================================================")
    iterator = trange(opt_steps, desc="Training", smoothing=0.1)
    generator.train()
    for step in iterator:
        optimizer.zero_grad()

        if resample_now(step):
            discriminator.resample()
            f_real = None  # old target features no longer valid

        if f_real is None:
            f_real = discriminator.forward_and_fit_scales(train_dataset.paths)

        x0, idx = train_dataset.sample_x0_and_idx(bs)
        x_fake = generator(x0, n_steps=n_gen_steps)
        f_fake = discriminator(x_fake)

        f_real_batch = f_real[idx]
        if bias_correction:
            delta = f_fake - f_real_batch
            loss = delta.mean(0).square().mean() - (
                delta.var(dim=0, correction=1).mean() / delta.size(0)
            )
        else:
            loss = F.mse_loss(f_fake.mean(0), f_real_batch.mean(0))

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if step > 0 and step % 100 == 0:
            logs = {"train/loss": loss.item()}
            iterator.set_postfix(logs)
            logger.log(logs, step=step)

        if step > 0 and step % val_every == 0:
            logs = v_step(generator)
            logger.log(logs, step=step)

    print("===========================================================================")
    print("Training finished.")
    print("===========================================================================")

    torch.save(generator.state_dict(), os.path.join(logger.run_dir, "generator.pth"))
    generator.eval()

    final_logs = final_validation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        generator=generator,
        plots_path=logger.run_dir,
        val_path_len=val_path_len,
        save_dir=logger.run_dir if save_paths else None,
    )

    logger.log(final_logs, step=opt_steps)
    for f in os.listdir(logger.run_dir):
        if f.endswith(".png"):
            logger.save(os.path.join(logger.run_dir, f))
    logger.finish()


if __name__ == "__main__":
    app()
