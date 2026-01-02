# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Evaluate the fine-tuned antichess model.

Computes metrics on the test set:
- Cross-entropy loss
- Action-value prediction accuracy (top-1, top-5)
- Kendall's tau correlation

Usage:
    python -m antichess_data.evaluate \
        --checkpoint-dir checkpoints/antichess \
        --data-dir data/antichess \
        --num-eval 10000
"""

import os
import sys
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants
from src import tokenizer
from src import training_utils
from src import transformer
from src import utils

from antichess_data.finetune import (
    build_antichess_data_loader,
    get_270m_config,
)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    "checkpoints/antichess",
    "Directory containing fine-tuned checkpoint",
)
flags.DEFINE_string(
    "data_dir",
    "data/antichess",
    "Directory containing antichess test .bag files",
)
flags.DEFINE_integer(
    "checkpoint_step",
    -1,
    "Checkpoint step to evaluate (-1 for latest)",
)
flags.DEFINE_integer(
    "num_eval",
    10000,
    "Number of examples to evaluate",
)
flags.DEFINE_integer(
    "batch_size",
    64,
    "Evaluation batch size",
)
flags.DEFINE_bool(
    "use_ema_params",
    True,
    "Use EMA parameters for evaluation",
)


def compute_metrics(
    predictor: constants.Predictor,
    params,
    data_iter,
    num_batches: int,
    num_return_buckets: int = 128,
) -> dict:
    """Compute evaluation metrics.

    Args:
        predictor: The predictor model.
        params: Model parameters.
        data_iter: Data iterator.
        num_batches: Number of batches to evaluate.
        num_return_buckets: Number of return buckets.

    Returns:
        Dictionary of metrics.
    """
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    # Get bucket values for computing expected return
    _, bucket_values = utils.get_uniform_buckets_edges_values(num_return_buckets)

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        sequences, loss_mask = next(data_iter)

        # sequences shape: [B, 79] for action_value
        # Last token is the return bucket target
        batch_size = sequences.shape[0]

        # Get model predictions
        conditionals = predictor.predict(params=params, targets=sequences, rng=None)

        # Extract predictions for the return bucket (last position)
        # conditionals shape: [B, T, num_buckets]
        logits = conditionals[:, -1, :]  # [B, num_buckets]

        # Get target return buckets (last token in sequence)
        targets = sequences[:, -1]  # [B]

        # Compute cross-entropy loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, targets[:, None], axis=-1
        ).squeeze(-1)
        batch_loss = -jnp.mean(target_log_probs)
        total_loss += float(batch_loss) * batch_size

        # Top-1 accuracy
        predictions = jnp.argmax(logits, axis=-1)
        correct_top1 = jnp.sum(predictions == targets)
        total_correct_top1 += int(correct_top1)

        # Top-5 accuracy
        top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
        correct_top5 = jnp.sum(jnp.any(top5_preds == targets[:, None], axis=-1))
        total_correct_top5 += int(correct_top5)

        total_samples += batch_size

        # Store for correlation
        all_predictions.extend(predictions.tolist())
        all_targets.extend(targets.tolist())

    # Compute aggregate metrics
    avg_loss = total_loss / total_samples
    top1_accuracy = total_correct_top1 / total_samples
    top5_accuracy = total_correct_top5 / total_samples

    # Compute Kendall's tau (rank correlation)
    from scipy import stats
    tau, tau_pvalue = stats.kendalltau(all_predictions, all_targets)

    # Mean absolute bucket error
    bucket_errors = np.abs(np.array(all_predictions) - np.array(all_targets))
    mean_bucket_error = np.mean(bucket_errors)

    return {
        "loss": avg_loss,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "kendall_tau": tau,
        "kendall_tau_pvalue": tau_pvalue,
        "mean_bucket_error": mean_bucket_error,
        "num_samples": total_samples,
    }


def main(_):
    """Main evaluation function."""
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
    data_dir = os.path.join(os.getcwd(), FLAGS.data_dir)

    logging.info("=" * 60)
    logging.info("ANTICHESS MODEL EVALUATION")
    logging.info("=" * 60)

    # Build model
    predictor_config = get_270m_config()
    predictor = transformer.build_transformer_predictor(predictor_config)

    # Load parameters
    logging.info(f"Loading checkpoint from {checkpoint_dir}")
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )

    params = training_utils.load_parameters(
        params=dummy_params,
        step=FLAGS.checkpoint_step,
        use_ema_params=FLAGS.use_ema_params,
        checkpoint_dir=checkpoint_dir,
    )
    logging.info("Checkpoint loaded successfully")

    # Build test data loader
    logging.info(f"Loading test data from {data_dir}")
    data_loader = build_antichess_data_loader(
        data_dir=data_dir,
        split="test",
        batch_size=FLAGS.batch_size,
        num_return_buckets=128,
        shuffle=False,
        seed=0,
        worker_count=0,
        num_records=FLAGS.num_eval,
    )
    data_iter = iter(data_loader)

    num_batches = FLAGS.num_eval // FLAGS.batch_size

    # Compute metrics
    logging.info(f"Evaluating on {FLAGS.num_eval} examples...")
    metrics = compute_metrics(
        predictor=predictor,
        params=params,
        data_iter=data_iter,
        num_batches=num_batches,
    )

    # Print results
    logging.info("=" * 60)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"  Samples evaluated: {metrics['num_samples']:,}")
    logging.info(f"  Cross-entropy loss: {metrics['loss']:.4f}")
    logging.info(f"  Top-1 accuracy: {metrics['top1_accuracy']*100:.2f}%")
    logging.info(f"  Top-5 accuracy: {metrics['top5_accuracy']*100:.2f}%")
    logging.info(f"  Kendall's tau: {metrics['kendall_tau']:.4f}")
    logging.info(f"  Mean bucket error: {metrics['mean_bucket_error']:.2f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    app.run(main)
