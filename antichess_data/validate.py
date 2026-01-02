# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Validate generated antichess .bag files."""

import argparse
import logging
import sys
from pathlib import Path

import chess.variant
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bagz import BagReader
from src.constants import CODERS, ActionValueData, BehavioralCloningData, StateValueData
from src.utils import MOVE_TO_ACTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_bag_file(
    bag_path: str | Path,
    policy: str,
    num_samples: int = 10,
    check_all: bool = False,
) -> dict:
    """Validate a .bag file for correctness.

    Args:
        bag_path: Path to .bag file.
        policy: "action_value", "state_value", or "behavioral_cloning".
        num_samples: Number of samples to display.
        check_all: If True, validate ALL records (slow).

    Returns:
        Dictionary with validation statistics.
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        logger.error(f"File not found: {bag_path}")
        return {"error": "file_not_found"}

    logger.info(f"Validating {bag_path}")

    reader = BagReader(str(bag_path))
    coder = CODERS[policy]

    stats = {
        "path": str(bag_path),
        "policy": policy,
        "total_records": len(reader),
        "valid_records": 0,
        "invalid_records": 0,
        "win_prob_min": float("inf"),
        "win_prob_max": float("-inf"),
        "win_prob_mean": 0.0,
        "win_probs": [],
        "samples": [],
    }

    # Determine indices to check
    if check_all:
        indices = range(len(reader))
    else:
        # Check first few, last few, and random samples
        n = len(reader)
        indices = list(range(min(100, n)))  # First 100
        indices += list(range(max(0, n - 100), n))  # Last 100
        # Random samples
        if n > 200:
            rng = np.random.default_rng(42)
            indices += list(rng.choice(range(100, n - 100), min(1000, n - 200), replace=False))
        indices = sorted(set(indices))

    logger.info(f"Checking {len(indices):,} of {len(reader):,} records")

    for idx in indices:
        try:
            raw = reader[idx]
            record = coder.decode(raw)

            if policy == "action_value":
                fen, move, win_prob = record
                stats["win_probs"].append(win_prob)

                # Validate FEN
                board = chess.variant.GiveawayBoard(fen)

                # Validate move
                if move not in MOVE_TO_ACTION:
                    logger.warning(f"Unknown move {move} at index {idx}")

                # Validate win_prob
                if not 0.0 <= win_prob <= 1.0:
                    logger.warning(f"Invalid win_prob {win_prob} at index {idx}")

                stats["valid_records"] += 1

                if len(stats["samples"]) < num_samples:
                    stats["samples"].append({
                        "idx": idx,
                        "fen": fen,
                        "move": move,
                        "win_prob": win_prob,
                    })

            elif policy == "state_value":
                fen, win_prob = record
                stats["win_probs"].append(win_prob)

                # Validate FEN
                board = chess.variant.GiveawayBoard(fen)

                # Validate win_prob
                if not 0.0 <= win_prob <= 1.0:
                    logger.warning(f"Invalid win_prob {win_prob} at index {idx}")

                stats["valid_records"] += 1

                if len(stats["samples"]) < num_samples:
                    stats["samples"].append({
                        "idx": idx,
                        "fen": fen,
                        "win_prob": win_prob,
                    })

            elif policy == "behavioral_cloning":
                fen, move = record

                # Validate FEN
                board = chess.variant.GiveawayBoard(fen)

                # Validate move
                if move not in MOVE_TO_ACTION:
                    logger.warning(f"Unknown move {move} at index {idx}")

                stats["valid_records"] += 1

                if len(stats["samples"]) < num_samples:
                    stats["samples"].append({
                        "idx": idx,
                        "fen": fen,
                        "move": move,
                    })

        except Exception as e:
            stats["invalid_records"] += 1
            logger.debug(f"Failed to decode record {idx}: {e}")

    # Compute statistics
    if stats["win_probs"]:
        stats["win_prob_min"] = min(stats["win_probs"])
        stats["win_prob_max"] = max(stats["win_probs"])
        stats["win_prob_mean"] = sum(stats["win_probs"]) / len(stats["win_probs"])

    # Cleanup for output
    del stats["win_probs"]

    return stats


def validate_dataset(
    data_dir: str | Path,
    policy: str = "action_value",
    num_samples: int = 5,
) -> dict:
    """Validate a full dataset directory.

    Args:
        data_dir: Directory containing train/ and test/ subdirectories.
        policy: Policy type to validate.
        num_samples: Number of samples to show per file.

    Returns:
        Dictionary with validation results for all files.
    """
    data_dir = Path(data_dir)

    results = {"train": [], "test": []}

    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        # Find .bag files
        bag_files = list(split_dir.glob(f"*{policy}*.bag"))
        if not bag_files:
            bag_files = list(split_dir.glob("*.bag"))

        for bag_file in sorted(bag_files):
            stats = validate_bag_file(bag_file, policy, num_samples)
            results[split].append(stats)

    return results


def print_validation_results(results: dict):
    """Print validation results in a readable format."""
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    total_train = 0
    total_test = 0

    for split in ["train", "test"]:
        print(f"\n{split.upper()} SET:")
        print("-" * 40)

        for stats in results[split]:
            path = Path(stats["path"]).name
            total = stats["total_records"]
            valid = stats["valid_records"]
            invalid = stats["invalid_records"]

            if split == "train":
                total_train += total
            else:
                total_test += total

            print(f"  {path}:")
            print(f"    Total records: {total:,}")
            print(f"    Valid checked: {valid:,}")
            print(f"    Invalid: {invalid:,}")

            if "win_prob_mean" in stats and stats["win_prob_mean"] != float("inf"):
                print(f"    Win prob range: [{stats['win_prob_min']:.4f}, {stats['win_prob_max']:.4f}]")
                print(f"    Win prob mean: {stats['win_prob_mean']:.4f}")

            if stats.get("samples"):
                print(f"    Sample records:")
                for sample in stats["samples"][:3]:
                    if "win_prob" in sample:
                        print(f"      - FEN: {sample['fen'][:50]}...")
                        if "move" in sample:
                            print(f"        Move: {sample['move']}, WinProb: {sample['win_prob']:.4f}")
                        else:
                            print(f"        WinProb: {sample['win_prob']:.4f}")
                    else:
                        print(f"      - FEN: {sample['fen'][:50]}...")
                        print(f"        Move: {sample['move']}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_train:,} train + {total_test:,} test = {total_train + total_test:,} records")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate antichess .bag files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/antichess",
        help="Directory containing train/ and test/ subdirectories",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["action_value", "state_value", "behavioral_cloning"],
        default="action_value",
        help="Policy type to validate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample records to display per file",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check ALL records (slow for large files)",
    )

    args = parser.parse_args()

    results = validate_dataset(
        data_dir=args.data_dir,
        policy=args.policy,
        num_samples=args.num_samples,
    )

    print_validation_results(results)


if __name__ == "__main__":
    main()
