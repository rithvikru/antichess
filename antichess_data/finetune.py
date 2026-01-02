# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Fine-tune the 270M chess transformer on antichess data.

This script loads the pretrained 270M checkpoint and fine-tunes it on
antichess action-value data using the same supervised setup as the original
paper (cross-entropy on binned action-values with 128 buckets).

Usage:
    python -m antichess_data.finetune \
        --data-dir data/antichess \
        --checkpoint-dir checkpoints/270M \
        --output-dir checkpoints/antichess \
        --num-steps 100000 \
        --learning-rate 1e-5 \
        --batch-size 256

For TPU usage:
    The script automatically detects and uses all available TPU cores.
    JAX will handle distribution across the TPU pod.
"""

import copy
import functools
import os
import re
import sys
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
import grain.python as pygrain
import haiku as hk
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import bagz
from src import constants
from src import tokenizer
from src import training_utils
from src import transformer
from src import utils


# Command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    "data/antichess",
    "Directory containing antichess train/test .bag files",
)
flags.DEFINE_string(
    "checkpoint_dir",
    "checkpoints/270M",
    "Directory containing pretrained 270M checkpoint",
)
flags.DEFINE_string(
    "output_dir",
    "checkpoints/antichess",
    "Directory to save fine-tuned checkpoints",
)
flags.DEFINE_integer(
    "checkpoint_step",
    6_400_000,
    "Step of the pretrained checkpoint to load (-1 for latest)",
)
flags.DEFINE_float(
    "learning_rate",
    1e-5,
    "Learning rate for fine-tuning (lower than pretraining)",
)
flags.DEFINE_integer(
    "batch_size",
    256,
    "Global batch size (sharded across devices)",
)
flags.DEFINE_integer(
    "num_steps",
    100_000,
    "Number of fine-tuning steps",
)
flags.DEFINE_integer(
    "log_frequency",
    100,
    "Logging frequency in steps",
)
flags.DEFINE_integer(
    "ckpt_frequency",
    1000,
    "Checkpoint frequency in steps",
)
flags.DEFINE_integer(
    "save_frequency",
    5000,
    "Permanent save frequency in steps",
)
flags.DEFINE_float(
    "max_grad_norm",
    1.0,
    "Maximum gradient norm for clipping",
)
flags.DEFINE_integer(
    "num_return_buckets",
    128,
    "Number of return buckets (must match pretrained model)",
)
flags.DEFINE_bool(
    "use_ema_params",
    True,
    "Load EMA parameters from pretrained checkpoint",
)
flags.DEFINE_integer(
    "seed",
    42,
    "Random seed",
)
flags.DEFINE_integer(
    "worker_count",
    0,
    "Number of data loading workers",
)


def get_270m_config(num_return_buckets: int = 128) -> transformer.TransformerConfig:
    """Returns the 270M model configuration."""
    return transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=num_return_buckets,  # num_return_buckets for action_value
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,  # 79
        num_heads=8,
        num_layers=16,
        embedding_dim=1024,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
        seed=FLAGS.seed,
    )


def _resolve_bag_path(split_dir: Path, policy: str) -> str:
    """Resolve the .bag path, supporting sharded datasets via @N syntax."""
    expected = split_dir / f"{policy}_data.bag"
    if expected.exists():
        return str(expected)

    shard_files = sorted(split_dir.glob(f"{policy}-*-of-*_data.bag"))
    if shard_files:
        match = re.match(
            rf"{re.escape(policy)}-(\d+)-of-(\d+)_data\.bag$",
            shard_files[0].name,
        )
        if match:
            num_shards = int(match.group(2))
            return str(split_dir / f"{policy}@{num_shards:05d}_data.bag")
        return str(shard_files[0])

    fallback = sorted(split_dir.glob(f"*{policy}*.bag"))
    if fallback:
        return str(fallback[0])

    raise FileNotFoundError(f"No {policy} .bag files found in {split_dir}")


def build_antichess_data_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_return_buckets: int,
    policy: str = "action_value",
    shuffle: bool = True,
    seed: int = 0,
    worker_count: int = 0,
    num_records: int | None = None,
) -> pygrain.DataLoader:
    """Build a data loader for antichess .bag files.

    Args:
        data_dir: Base directory containing train/ and test/ subdirectories.
        split: "train" or "test".
        batch_size: Batch size.
        num_return_buckets: Number of return buckets (128).
        policy: Dataset policy name (default: "action_value").
        shuffle: Whether to shuffle data.
        seed: Random seed.
        worker_count: Number of data loading workers.
        num_records: Limit on number of records (None for all).

    Returns:
        PyGrain DataLoader for antichess action-value data.
    """
    if policy != "action_value":
        raise ValueError(
            "Only action_value policy is supported for fine-tuning."
        )
    # Resolve .bag file(s) in the split directory (supports sharded datasets).
    split_dir = Path(data_dir) / split
    bag_path = _resolve_bag_path(split_dir=split_dir, policy=policy)
    logging.info(f"Loading data from: {bag_path}")

    data_source = bagz.BagDataSource(bag_path)

    if num_records is not None:
        if len(data_source) < num_records:
            logging.warning(
                f"Requested {num_records} records but only {len(data_source)} available"
            )
            num_records = len(data_source)
    else:
        num_records = len(data_source)

    logging.info(f"Using {num_records:,} records from {split} split")

    sampler = pygrain.IndexSampler(
        num_records=num_records,
        shard_options=pygrain.NoSharding(),
        shuffle=shuffle,
        num_epochs=None,
        seed=seed,
    )

    # Use the action_value transformation from the original repo
    from src.data_loader import ConvertActionValueDataToSequence, FilterInvalidMoves

    transformations = [
        # Filter out king promotions (antichess-specific moves not in action vocab)
        FilterInvalidMoves(policy),
        ConvertActionValueDataToSequence(num_return_buckets=num_return_buckets),
        pygrain.Batch(batch_size, drop_remainder=True),
    ]

    return pygrain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=worker_count,
        read_options=None,
    )


def load_pretrained_params(
    predictor: constants.Predictor,
    checkpoint_dir: str,
    checkpoint_step: int,
    use_ema_params: bool,
) -> hk.Params:
    """Load pretrained parameters from checkpoint.

    Args:
        predictor: The predictor to initialize params for.
        checkpoint_dir: Directory containing the checkpoint.
        checkpoint_step: Step to load (-1 for latest).
        use_ema_params: Whether to load EMA parameters.

    Returns:
        Loaded parameters.
    """
    logging.info(f"Loading pretrained checkpoint from {checkpoint_dir}")

    # Initialize dummy params to get the structure
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )

    # Load pretrained params
    params = training_utils.load_parameters(
        params=dummy_params,
        step=checkpoint_step,
        use_ema_params=use_ema_params,
        checkpoint_dir=checkpoint_dir,
    )

    logging.info(f"Loaded checkpoint (use_ema={use_ema_params})")
    return params


def finetune(
    predictor_config: transformer.TransformerConfig,
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    checkpoint_step: int,
    learning_rate: float,
    batch_size: int,
    num_steps: int,
    log_frequency: int,
    ckpt_frequency: int,
    save_frequency: int,
    max_grad_norm: float,
    num_return_buckets: int,
    use_ema_params: bool,
    seed: int,
    worker_count: int,
) -> hk.Params:
    """Fine-tune the 270M model on antichess data.

    Args:
        predictor_config: Transformer configuration.
        data_dir: Directory with antichess .bag files.
        checkpoint_dir: Directory with pretrained 270M checkpoint.
        output_dir: Directory to save fine-tuned checkpoints.
        checkpoint_step: Step of pretrained checkpoint to load.
        learning_rate: Learning rate for fine-tuning.
        batch_size: Batch size per device.
        num_steps: Number of fine-tuning steps.
        log_frequency: Logging frequency.
        ckpt_frequency: Checkpoint frequency.
        save_frequency: Permanent save frequency.
        max_grad_norm: Maximum gradient norm.
        num_return_buckets: Number of return buckets.
        use_ema_params: Load EMA params from pretrained.
        seed: Random seed.
        worker_count: Data loading workers.

    Returns:
        Fine-tuned parameters.
    """
    logging.info("=" * 60)
    logging.info("ANTICHESS FINE-TUNING")
    logging.info("=" * 60)
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"Device count: {jax.device_count()}")
    logging.info(f"Process count: {jax.process_count()}")
    logging.info(f"Local device count: {jax.local_device_count()}")
    logging.info("=" * 60)

    if num_return_buckets != 128:
        raise ValueError(
            "The pretrained 270M checkpoint expects num_return_buckets=128."
        )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            "batch_size must be divisible by the total device count. "
            f"Got batch_size={batch_size}, device_count={jax.device_count()}."
        )

    # Build predictor
    predictor = transformer.build_transformer_predictor(predictor_config)

    # Build data loader
    # Adjust seed for multi-host
    data_seed = seed + jax.process_index()
    data_iter = build_antichess_data_loader(
        data_dir=data_dir,
        split="train",
        batch_size=batch_size,
        num_return_buckets=num_return_buckets,
        policy="action_value",
        shuffle=True,
        seed=data_seed,
        worker_count=worker_count,
    ).__iter__()

    # Load pretrained parameters
    params = load_pretrained_params(
        predictor=predictor,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=checkpoint_step,
        use_ema_params=use_ema_params,
    )

    # Initialize EMA params as copy of loaded params
    params_ema = copy.deepcopy(params)

    # Create optimizer with lower learning rate for fine-tuning
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)

    # Create loss and update functions
    loss_fn = training_utils.make_loss_fn(predictor=predictor)
    grad_fn = jax.value_and_grad(loss_fn)
    update_fn = functools.partial(
        training_utils.update_parameters,
        grad_fn=grad_fn,
        optimizer=optimizer,
    )

    # Create sharding for distributed training
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    sharding = jax.sharding.PositionalSharding(devices)
    sharding = sharding.reshape((jax.device_count(), 1))

    # Replicate params across devices
    params = training_utils.replicate(params, sharding)
    params_ema = training_utils.replicate(params_ema, sharding)
    opt_state = training_utils.replicate(opt_state, sharding)

    # Initialize checkpoint manager
    os.makedirs(output_dir, exist_ok=True)

    # Helper to unreplicate params for checkpointing (avoids sharding issues)
    def unreplicate_for_ckpt(x):
        """Convert sharded arrays to single-device for checkpointing."""
        return jax.tree.map(lambda arr: jax.device_get(arr), x)

    checkpoint_manager = training_utils.get_checkpoint_manager(
        ckpt_frequency=ckpt_frequency,
        max_to_keep=2,
        save_frequency=save_frequency,
        checkpoint_dir=output_dir,
    )

    # Check for existing fine-tuning checkpoint
    latest_step = 0
    if checkpoint_manager.latest_step() is not None:
        latest_step = checkpoint_manager.latest_step()
        logging.info(f"Resuming from fine-tuning checkpoint {latest_step}")
        # Restore unreplicated checkpoint
        restored = checkpoint_manager.restore(
            step=latest_step,
            items=dict(
                params=jax.device_get(params),
                params_ema=jax.device_get(params_ema),
                opt_state=jax.device_get(opt_state),
                data_iter=data_iter,
            ),
        )
        # Re-replicate across devices
        params = training_utils.replicate(restored['params'], sharding)
        params_ema = training_utils.replicate(restored['params_ema'], sharding)
        opt_state = training_utils.replicate(restored['opt_state'], sharding)
        data_iter = restored['data_iter']

    logging.info("=" * 60)
    logging.info("Starting fine-tuning loop")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Num steps: {num_steps}")
    logging.info(f"  Starting from step: {latest_step}")
    logging.info("=" * 60)

    # Main training loop
    for step in range(latest_step, num_steps + 1):
        # Save checkpoint
        if step % ckpt_frequency == 0:
            logging.info(f"Checkpointing step {step}")
            checkpoint_manager.save(
                step=step,
                items=dict(
                    params=unreplicate_for_ckpt(params),
                    params_ema=unreplicate_for_ckpt(params_ema),
                    opt_state=unreplicate_for_ckpt(opt_state),
                    data_iter=data_iter,
                ),
            )

        # Get batch
        sequences, loss_mask = next(data_iter)
        sequences = jax.lax.with_sharding_constraint(sequences, sharding)
        loss_mask = jax.lax.with_sharding_constraint(loss_mask, sharding)

        # Update parameters
        params, params_ema, opt_state, loss, grad_norm = update_fn(
            params=params,
            params_ema=params_ema,
            opt_state=opt_state,
            sequences=sequences,
            loss_mask=loss_mask,
        )

        # Log
        if step % log_frequency == 0:
            logging.info(
                f"step: {step:6d} | loss: {jax.device_get(loss):.4f} | "
                f"grad_norm: {jax.device_get(grad_norm):.4f}"
            )

    # Final checkpoint
    logging.info(f"Saving final checkpoint at step {num_steps}")
    checkpoint_manager.save(
        step=num_steps,
        items=dict(
            params=unreplicate_for_ckpt(params),
            params_ema=unreplicate_for_ckpt(params_ema),
            opt_state=unreplicate_for_ckpt(opt_state),
            data_iter=data_iter,
        ),
    )
    checkpoint_manager.close()

    logging.info("=" * 60)
    logging.info("FINE-TUNING COMPLETE")
    logging.info(f"Checkpoints saved to: {output_dir}")
    logging.info("=" * 60)

    return jax.device_get(params)


def main(_):
    """Main entry point."""
    # Resolve paths
    data_dir = os.path.join(os.getcwd(), FLAGS.data_dir)
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
    output_dir = os.path.join(os.getcwd(), FLAGS.output_dir)

    # Validate paths
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Get model config
    predictor_config = get_270m_config(num_return_buckets=FLAGS.num_return_buckets)

    # Run fine-tuning
    finetune(
        predictor_config=predictor_config,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        checkpoint_step=FLAGS.checkpoint_step,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.num_steps,
        log_frequency=FLAGS.log_frequency,
        ckpt_frequency=FLAGS.ckpt_frequency,
        save_frequency=FLAGS.save_frequency,
        max_grad_norm=FLAGS.max_grad_norm,
        num_return_buckets=FLAGS.num_return_buckets,
        use_ema_params=FLAGS.use_ema_params,
        seed=FLAGS.seed,
        worker_count=FLAGS.worker_count,
    )


if __name__ == "__main__":
    app.run(main)
