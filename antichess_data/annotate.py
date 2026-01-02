# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Fairy-Stockfish annotation engine for antichess positions."""

import argparse
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import chess.engine
import chess.variant
import polars as pl
from tqdm import tqdm

from antichess_data.utils import (
    TIME_LIMIT_MS_DEFAULT,
    centipawns_to_win_prob_antichess,
    mate_to_centipawns,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionValue(NamedTuple):
    """Action value annotation for a single move."""
    fen: str
    move_uci: str
    win_prob: float


class StateValue(NamedTuple):
    """State value annotation for a position."""
    fen: str
    win_prob: float


class FairyStockfishAnnotator:
    """Annotator using Fairy-Stockfish for antichess evaluation."""

    def __init__(
        self,
        engine_path: str,
        time_limit_ms: int = TIME_LIMIT_MS_DEFAULT,
    ):
        """Initialize the Fairy-Stockfish annotator.

        Args:
            engine_path: Path to Fairy-Stockfish binary.
            time_limit_ms: Time limit per position in milliseconds.
        """
        self.engine_path = engine_path
        self.time_limit_ms = time_limit_ms
        self.limit = chess.engine.Limit(time=time_limit_ms / 1000)

        # Start engine
        logger.info(f"Starting Fairy-Stockfish from {engine_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        # Configure for antichess variant
        # Fairy-Stockfish uses "giveaway" for antichess
        self.engine.configure({"UCI_Variant": "giveaway"})
        logger.info("Configured engine for giveaway (antichess) variant")

    def __del__(self):
        """Clean up engine on destruction."""
        if hasattr(self, 'engine'):
            try:
                self.engine.quit()
            except Exception:
                pass

    def close(self):
        """Explicitly close the engine."""
        self.engine.quit()

    def _score_to_win_prob(self, score: chess.engine.PovScore) -> float:
        """Convert engine score to win probability for antichess.

        Args:
            score: Engine score from analysis.

        Returns:
            Win probability in [0, 1] for antichess (inverted).
        """
        relative_score = score.relative

        if relative_score.is_mate():
            # In antichess, getting "mated" (losing all pieces) is WINNING
            mate_in = relative_score.mate()
            cp = mate_to_centipawns(mate_in)
        else:
            cp = relative_score.score()
            if cp is None:
                # Fallback for edge cases
                cp = 0

        return centipawns_to_win_prob_antichess(cp)

    def get_state_value(self, board: chess.variant.GiveawayBoard) -> float:
        """Get V(s) - state value for a position.

        Args:
            board: Antichess board position.

        Returns:
            Win probability for the side to move.
        """
        info = self.engine.analyse(board, self.limit)
        return self._score_to_win_prob(info["score"])

    def get_action_value(
        self,
        board: chess.variant.GiveawayBoard,
        move: chess.Move,
    ) -> float:
        """Get Q(s,a) - action value for a specific move.

        Args:
            board: Antichess board position.
            move: Move to evaluate.

        Returns:
            Win probability after playing this move.
        """
        info = self.engine.analyse(board, self.limit, root_moves=[move])
        return self._score_to_win_prob(info["score"])

    def get_all_action_values(
        self,
        board: chess.variant.GiveawayBoard,
    ) -> list[tuple[str, float]]:
        """Get Q(s,a) for all legal moves.

        Args:
            board: Antichess board position.

        Returns:
            List of (move_uci, win_prob) tuples for all legal moves.
        """
        results = []
        for move in board.legal_moves:
            win_prob = self.get_action_value(board, move)
            results.append((move.uci(), win_prob))
        return results

    def annotate_position(
        self,
        fen: str,
        move_uci: str | None = None,
        include_all_moves: bool = False,
    ) -> dict:
        """Annotate a position with engine evaluation.

        Args:
            fen: FEN string of the position.
            move_uci: Optional move to evaluate (for action-value).
            include_all_moves: If True, evaluate all legal moves.

        Returns:
            Dictionary with annotation results.
        """
        board = chess.variant.GiveawayBoard(fen)

        result = {
            "fen": fen,
            "state_value": self.get_state_value(board),
        }

        if move_uci:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                result["move_uci"] = move_uci
                result["action_value"] = self.get_action_value(board, move)
            else:
                logger.warning(f"Move {move_uci} not legal in position {fen}")

        if include_all_moves:
            result["all_action_values"] = self.get_all_action_values(board)

        return result


def annotate_positions_batch(
    positions: list[tuple[str, str]],
    engine_path: str,
    time_limit_ms: int = TIME_LIMIT_MS_DEFAULT,
    policy: str = "action_value",
    progress: bool = True,
) -> Iterator[ActionValue | StateValue]:
    """Annotate a batch of positions.

    Args:
        positions: List of (fen, move_uci) tuples.
        engine_path: Path to Fairy-Stockfish binary.
        time_limit_ms: Time limit per position.
        policy: "action_value", "state_value", or "all_action_values".
        progress: Whether to show progress bar.

    Yields:
        Annotated positions.
    """
    annotator = FairyStockfishAnnotator(engine_path, time_limit_ms)

    try:
        iterator = positions
        if progress:
            iterator = tqdm(positions, desc="Annotating positions")

        for fen, move_uci in iterator:
            try:
                board = chess.variant.GiveawayBoard(fen)

                if policy == "state_value":
                    win_prob = annotator.get_state_value(board)
                    yield StateValue(fen=fen, win_prob=win_prob)

                elif policy == "action_value":
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        win_prob = annotator.get_action_value(board, move)
                        yield ActionValue(fen=fen, move_uci=move_uci, win_prob=win_prob)

                elif policy == "all_action_values":
                    for move_str, win_prob in annotator.get_all_action_values(board):
                        yield ActionValue(fen=fen, move_uci=move_str, win_prob=win_prob)

            except Exception as e:
                logger.debug(f"Failed to annotate {fen}: {e}")
                continue

    finally:
        annotator.close()


def annotate_positions_file(
    input_path: str | Path,
    output_path: str | Path,
    engine_path: str,
    time_limit_ms: int = TIME_LIMIT_MS_DEFAULT,
    policy: str = "action_value",
    limit: int | None = None,
) -> int:
    """Annotate positions from a Parquet file and save results.

    Args:
        input_path: Input Parquet with fen, move_uci columns.
        output_path: Output Parquet for annotated positions.
        engine_path: Path to Fairy-Stockfish binary.
        time_limit_ms: Time limit per position.
        policy: "action_value", "state_value", or "all_action_values".
        limit: Maximum positions to annotate.

    Returns:
        Number of annotated positions.
    """
    logger.info(f"Loading positions from {input_path}")
    df = pl.read_parquet(input_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Processing first {limit:,} positions")

    positions = list(zip(df["fen"].to_list(), df["move_uci"].to_list()))

    # Annotate
    results = []
    for annotation in annotate_positions_batch(
        positions,
        engine_path,
        time_limit_ms,
        policy,
    ):
        if isinstance(annotation, ActionValue):
            results.append({
                "fen": annotation.fen,
                "move_uci": annotation.move_uci,
                "win_prob": annotation.win_prob,
            })
        else:  # StateValue
            results.append({
                "fen": annotation.fen,
                "win_prob": annotation.win_prob,
            })

    # Save
    result_df = pl.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)
    logger.info(f"Saved {len(results):,} annotated positions to {output_path}")

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Annotate antichess positions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/antichess/positions.parquet",
        help="Input path for positions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/antichess/annotated_positions.parquet",
        help="Output path for annotated positions",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        required=True,
        help="Path to Fairy-Stockfish binary",
    )
    parser.add_argument(
        "--time-limit-ms",
        type=int,
        default=TIME_LIMIT_MS_DEFAULT,
        help=f"Time limit per position in ms (default: {TIME_LIMIT_MS_DEFAULT})",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["action_value", "state_value", "all_action_values"],
        default="action_value",
        help="Type of annotation to generate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of positions to annotate (for testing)",
    )

    args = parser.parse_args()

    annotate_positions_file(
        input_path=args.input,
        output_path=args.output,
        engine_path=args.engine_path,
        time_limit_ms=args.time_limit_ms,
        policy=args.policy,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
