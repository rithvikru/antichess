# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Antichess neural engine using the fine-tuned model.

This module provides an engine interface for playing antichess using the
fine-tuned transformer model. It follows the same pattern as the original
neural_engines.py but uses chess.variant.GiveawayBoard for antichess rules.

Usage:
    from antichess_data.engine import build_antichess_engine

    engine = build_antichess_engine(
        checkpoint_dir="checkpoints/antichess",
        checkpoint_step=-1,
    )

    board = chess.variant.GiveawayBoard()
    best_move = engine.play(board)
"""

import os
import sys
from pathlib import Path
from typing import Callable

import chess
import chess.variant
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants
from src import tokenizer
from src import training_utils
from src import transformer
from src import utils

from antichess_data.finetune import get_270m_config


class AntichessEngine:
    """Neural engine for antichess using the fine-tuned transformer."""

    def __init__(
        self,
        predict_fn: Callable,
        return_buckets_values: np.ndarray,
    ):
        """Initialize the antichess engine.

        Args:
            predict_fn: Function that takes FEN and returns action-value logits.
            return_buckets_values: Center values for each return bucket.
        """
        self._predict_fn = predict_fn
        self._return_buckets_values = return_buckets_values

    def get_action_values(
        self,
        board: chess.variant.GiveawayBoard,
    ) -> dict[chess.Move, float]:
        """Get Q(s,a) for all legal moves.

        Args:
            board: Antichess board position.

        Returns:
            Dictionary mapping moves to their expected win probabilities.
        """
        fen = board.fen()
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return {}

        # Get action values from model
        action_values = self._predict_fn(fen, legal_moves)

        return {move: action_values[move] for move in legal_moves}

    def play(self, board: chess.variant.GiveawayBoard) -> chess.Move:
        """Select the best move for the current position.

        Args:
            board: Antichess board position.

        Returns:
            The move with the highest expected value.
        """
        action_values = self.get_action_values(board)

        if not action_values:
            raise ValueError("No legal moves available")

        # Select move with highest expected value
        best_move = max(action_values.keys(), key=lambda m: action_values[m])
        return best_move

    def get_win_probability(
        self,
        board: chess.variant.GiveawayBoard,
    ) -> float:
        """Get the win probability for the current position.

        This is the expected value of the best move.

        Args:
            board: Antichess board position.

        Returns:
            Win probability in [0, 1].
        """
        action_values = self.get_action_values(board)

        if not action_values:
            return 0.5  # Draw-ish for no legal moves

        return max(action_values.values())


def wrap_predict_fn(
    predictor: constants.Predictor,
    params,
    return_buckets_values: np.ndarray,
    batch_size: int = 32,
) -> Callable:
    """Wrap the predictor into a function that returns action values.

    Args:
        predictor: The transformer predictor.
        params: Model parameters.
        return_buckets_values: Center values for return buckets.
        batch_size: Batch size for prediction.

    Returns:
        Function that takes (fen, moves) and returns action values.
    """

    @jax.jit
    def _predict_batch(sequences):
        conditionals = predictor.predict(params=params, targets=sequences, rng=None)
        # Get logits for the return bucket (last position)
        logits = conditionals[:, -1, :]  # [B, num_buckets]
        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        # Compute expected value (weighted sum of bucket values)
        expected_values = jnp.sum(
            probs * return_buckets_values[None, :], axis=-1
        )
        return expected_values

    def predict_fn(
        fen: str,
        moves: list[chess.Move],
    ) -> dict[chess.Move, float]:
        """Predict action values for given moves.

        Args:
            fen: FEN string of the position.
            moves: List of moves to evaluate.

        Returns:
            Dictionary mapping moves to win probabilities.
        """
        if not moves:
            return {}

        # Tokenize FEN
        state_tokens = tokenizer.tokenize(fen).astype(np.int32)

        # Build sequences for each move
        sequences = []
        for move in moves:
            action_token = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
            # For action_value, we need: state (77) + action (1) + return (1)
            # We'll use a dummy return token (0) since we're predicting it
            return_token = np.array([0], dtype=np.int32)
            sequence = np.concatenate([state_tokens, action_token, return_token])
            sequences.append(sequence)

        sequences = np.stack(sequences, axis=0)  # [num_moves, 79]

        # Predict in batches
        all_values = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            values = _predict_batch(batch)
            all_values.extend(values.tolist())

        return {move: value for move, value in zip(moves, all_values)}

    return predict_fn


def build_antichess_engine(
    checkpoint_dir: str,
    checkpoint_step: int = -1,
    use_ema_params: bool = True,
) -> AntichessEngine:
    """Build an antichess engine from a fine-tuned checkpoint.

    Args:
        checkpoint_dir: Directory containing the fine-tuned checkpoint.
        checkpoint_step: Step to load (-1 for latest).
        use_ema_params: Whether to use EMA parameters.

    Returns:
        AntichessEngine instance.
    """
    # Build model
    predictor_config = get_270m_config()
    predictor = transformer.build_transformer_predictor(predictor_config)

    # Load parameters
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )

    params = training_utils.load_parameters(
        params=dummy_params,
        step=checkpoint_step,
        use_ema_params=use_ema_params,
        checkpoint_dir=checkpoint_dir,
    )

    # Get return bucket values
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)

    # Create predict function
    predict_fn = wrap_predict_fn(
        predictor=predictor,
        params=params,
        return_buckets_values=return_buckets_values,
    )

    return AntichessEngine(
        predict_fn=predict_fn,
        return_buckets_values=return_buckets_values,
    )


# Example usage / testing
def main():
    """Test the antichess engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Test antichess engine")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/antichess",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="FEN to analyze (default: starting position)",
    )
    args = parser.parse_args()

    checkpoint_dir = os.path.join(os.getcwd(), args.checkpoint_dir)

    print("Building antichess engine...")
    engine = build_antichess_engine(checkpoint_dir)

    # Create board
    if args.fen:
        board = chess.variant.GiveawayBoard(args.fen)
    else:
        board = chess.variant.GiveawayBoard()

    print(f"\nPosition:\n{board}\n")
    print(f"FEN: {board.fen()}\n")

    # Get action values
    print("Action values for legal moves:")
    action_values = engine.get_action_values(board)
    sorted_moves = sorted(action_values.items(), key=lambda x: -x[1])

    for move, value in sorted_moves[:10]:
        print(f"  {move.uci()}: {value:.4f}")

    # Get best move
    best_move = engine.play(board)
    print(f"\nBest move: {best_move.uci()}")
    print(f"Win probability: {engine.get_win_probability(board):.4f}")


if __name__ == "__main__":
    main()
