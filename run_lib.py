# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models."""

import gc
import io
import os
import time
from datetime import datetime

import numpy as np
import logging

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training pipeline"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Work directory: {workdir}"
    )

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created sample directory: {sample_dir}"
    )

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created tensorboard directory: {tb_dir}"
    )
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model: {config.model.name}"
    )
    score_model = mutils.create_model(config)

    # Calculate model size
    total_params = sum(p.numel() for p in score_model.parameters())
    trainable_params = sum(
        p.numel() for p in score_model.parameters() if p.requires_grad
    )
    model_size_mb = sum(
        p.numel() * p.element_size() for p in score_model.parameters()
    ) / (1024 * 1024)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model created: {config.model.name}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params:,} total parameters, {trainable_params:,} trainable parameters"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model memory size: {model_size_mb:.2f} MB"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model device: {config.device}"
    )

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EMA initialized with decay rate: {config.model.ema_rate}"
    )

    optimizer = losses.get_optimizer(config, score_model.parameters())
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimizer created: {config.optim.optimizer}, learning rate: {config.optim.lr}"
    )

    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    # Build data iterators
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets: {config.data.dataset}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset config - image_size: {config.data.image_size}, num_channels: {config.data.num_channels}, centered: {config.data.centered}"
    )
    train_ds, eval_ds, _ = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization
    )
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data scaler: {'centered [-1,1]' if config.data.centered else 'not centered [0,1]'}"
    )

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    num_train_steps = config.training.n_iters
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training configuration:"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Total steps: {num_train_steps}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Batch size: {config.training.batch_size}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Continuous: {config.training.continuous}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Likelihood weighting: {config.training.likelihood_weighting}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Log frequency: {config.training.log_freq}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Eval frequency: {config.training.eval_freq}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Snapshot frequency: {config.training.snapshot_freq}"
    )

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop at step {initial_step}."
    )

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to PyTorch tensors and normalize them.
        batch_dict = next(train_iter)
        batch = batch_dict["image"].to(config.device).float()
        # Data is already in [B, H, W, C] format from datasets.py, convert to [B, C, H, W]
        if batch.dim() == 4 and batch.shape[-1] == 3:
            batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] step: {step}, training_loss: {loss.item():.5e}"
            )
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving preemption checkpoint at step {step}"
            )
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch_dict = next(eval_iter)
            eval_batch = eval_batch_dict["image"].to(config.device).float()
            # Data is already in [B, H, W, C] format from datasets.py, convert to [B, C, H, W]
            if eval_batch.dim() == 4 and eval_batch.shape[-1] == 3:
                eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] step: {step}, eval_loss: {eval_loss.item():.5e}"
            )
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_{save_step}.pth"
            )
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint at step {step} to {checkpoint_path}"
            )
            save_checkpoint(checkpoint_path, state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating samples at step {step}"
                )
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Samples generated: shape={sample.shape}, n_steps={n}"
                )
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving samples to {this_sample_dir}"
                )
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(
                    sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255
                ).astype(np.uint8)
                sample_file = os.path.join(this_sample_dir, "sample.np")
                with open(sample_file, "wb") as fout:
                    np.save(fout, sample)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved numpy array to {sample_file}"
                )

                image_file = os.path.join(this_sample_dir, "sample.png")
                save_image(image_grid, image_file)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved image grid to {image_file}"
                )


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation pipeline"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Work directory: {workdir}, Eval folder: {eval_folder}"
    )

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created eval directory: {eval_dir}"
    )

    # Build data pipeline
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets for evaluation: {config.data.dataset}"
    )
    train_ds, eval_ds, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
    )

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model for evaluation: {config.model.name}"
    )
    score_model = mutils.create_model(config)

    # Calculate model size
    total_params = sum(p.numel() for p in score_model.parameters())
    trainable_params = sum(
        p.numel() for p in score_model.parameters() if p.requires_grad
    )
    model_size_mb = sum(
        p.numel() * p.element_size() for p in score_model.parameters()
    ) / (1024 * 1024)
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model created: {config.model.name}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model size: {total_params:,} total parameters, {trainable_params:,} trainable parameters"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model memory size: {model_size_mb:.2f} MB"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model device: {config.device}"
    )

    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(
            sde,
            train=False,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
        )

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(
        config, uniform_dequantization=True, evaluation=True
    )
    if config.eval.bpd_dataset.lower() == "train":
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == "test":
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    device = config.device if hasattr(config, "device") else "cuda"
    inception_model = evaluation.get_inception_model(
        inceptionv3=inceptionv3, device=device
    )

    begin_ckpt = config.eval.begin_ckpt
    end_ckpt = config.eval.end_ckpt
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation configuration:"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Checkpoint range: {begin_ckpt} to {end_ckpt}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Enable loss: {config.eval.enable_loss}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Enable BPD: {config.eval.enable_bpd}"
    )
    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Enable sampling: {config.eval.enable_sampling}"
    )
    if config.eval.enable_sampling:
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Number of samples: {config.eval.num_samples}"
        )
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Batch size: {config.eval.batch_size}"
        )
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing checkpoint {ckpt}"
        )
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for the arrival of checkpoint_{ckpt}"
                )
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading checkpoint from {ckpt_path}"
        )
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded successfully, step: {state['step']}"
            )
        except:
            logging.warning(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] First checkpoint load attempt failed, retrying..."
            )
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded on second attempt, step: {state['step']}"
                )
            except:
                logging.warning(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Second checkpoint load attempt failed, retrying..."
                )
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded on third attempt, step: {state['step']}"
                )
        ema.copy_to(score_model.parameters())
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EMA parameters copied to model"
        )
        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing loss on evaluation dataset for checkpoint {ckpt}"
            )
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            num_eval_batches = len(eval_ds)
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total evaluation batches: {num_eval_batches}"
            )
            for i, batch in enumerate(eval_iter):
                eval_batch = batch["image"].to(config.device).float()
                # Data is already in [B, H, W, C] format from datasets.py, convert to [B, C, H, W]
                if eval_batch.dim() == 4 and eval_batch.shape[-1] == 3:
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss.item())
                if (i + 1) % 1000 == 0:
                    logging.info(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished {i + 1}/{num_eval_batches} batches, mean loss: {np.mean(all_losses):.5e}"
                    )

            # Save loss values to disk
            all_losses = np.asarray(all_losses)
            mean_loss = all_losses.mean()
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loss computation completed for checkpoint {ckpt}: mean={mean_loss:.5e}, std={all_losses.std():.5e}"
            )
            loss_file = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving loss values to {loss_file}"
            )
            with open(loss_file, "wb") as fout:
                np.savez_compressed(fout, all_losses=all_losses, mean_loss=mean_loss)

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing bits per dimension (BPD) for checkpoint {ckpt}"
            )
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset: {config.eval.bpd_dataset}, Number of repeats: {bpd_num_repeats}"
            )
            bpds = []
            for repeat in range(bpd_num_repeats):
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BPD computation - Repeat {repeat + 1}/{bpd_num_repeats}"
                )
                bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = batch["image"].to(config.device).float()
                    # Data is already in [B, H, W, C] format from datasets.py, convert to [B, C, H, W]
                    if eval_batch.dim() == 4 and eval_batch.shape[-1] == 3:
                        eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ckpt: {ckpt}, repeat: {repeat}, batch: {batch_id}, mean bpd: {np.mean(np.asarray(bpds)):.6f}"
                    )
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk
                    with open(
                        os.path.join(
                            eval_dir,
                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz",
                        ),
                        "wb",
                    ) as fout:
                        np.savez_compressed(fout, bpd)

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting sampling for checkpoint {ckpt}"
            )
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Number of sampling rounds: {num_sampling_rounds}, samples per round: {config.eval.batch_size}"
            )
            for r in range(num_sampling_rounds):
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sampling -- ckpt: {ckpt}, round: {r + 1}/{num_sampling_rounds}"
                )

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                os.makedirs(this_sample_dir, exist_ok=True)
                samples, n = sampling_fn(score_model)
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generated {len(samples)} samples in {n} steps"
                )
                samples = np.clip(
                    samples.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                samples = samples.reshape(
                    (
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        config.data.num_channels,
                    )
                )
                # Write samples to disk
                samples_file = os.path.join(this_sample_dir, f"samples_{r}.npz")
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving samples to {samples_file}"
                )
                with open(samples_file, "wb") as fout:
                    np.savez_compressed(fout, samples=samples)

                # Force garbage collection before calling Inception network
                gc.collect()
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Inception network on {len(samples)} samples"
                )
                latents = evaluation.run_inception_distributed(
                    samples, inception_model, inceptionv3=inceptionv3, device=device
                )
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inception features extracted"
                )
                # Force garbage collection again
                gc.collect()
                # Save latent represents of the Inception network to disk
                stats_file = os.path.join(this_sample_dir, f"statistics_{r}.npz")
                logging.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving Inception statistics to {stats_file}"
                )
                with open(stats_file, "wb") as fout:
                    np.savez_compressed(
                        fout, pool_3=latents["pool_3"], logits=latents["logits"]
                    )

            # Compute inception scores, FIDs and KIDs.
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing IS/FID/KID metrics for checkpoint {ckpt}"
            )
            # Load all statistics that have been previously computed and saved for each host
            all_logits = []
            all_pools = []
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            import glob

            stats = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(stats)} statistics files"
            )
            for stat_file in stats:
                with open(stat_file, "rb") as fin:
                    stat = np.load(fin)
                    if not inceptionv3:
                        all_logits.append(stat["logits"])
                    all_pools.append(stat["pool_3"])

            if not inceptionv3:
                all_logits = np.concatenate(all_logits, axis=0)[
                    : config.eval.num_samples
                ]
            all_pools = np.concatenate(all_pools, axis=0)[: config.eval.num_samples]

            # Load pre-computed dataset statistics.
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading pre-computed dataset statistics"
            )
            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats["pool_3"]
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset statistics loaded, pool_3 shape: {data_pools.shape}"
            )

            # Compute FID/KID/IS on all samples together.
            if not inceptionv3:
                inception_score = evaluation.compute_inception_score(all_logits)
            else:
                inception_score = -1

            fid = evaluation.compute_fid(data_pools, all_pools)
            kid = evaluation.compute_kid(data_pools, all_pools)

            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ckpt-{ckpt} --- inception_score: {inception_score:.6e}, FID: {fid:.6e}, KID: {kid:.6e}"
            )

            report_file = os.path.join(eval_dir, f"report_{ckpt}.npz")
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving evaluation report to {report_file}"
            )
            with open(report_file, "wb") as f:
                np.savez_compressed(f, IS=inception_score, fid=fid, kid=kid)
