# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Write annotated antichess data to .bag files compatible with Searchless Chess."""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

# Add parent directory to path for imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bagz import BagWriter
from src.constants import (
    CODERS,
    ActionValueData,
    BehavioralCloningData,
    StateValueData,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_action_value_bag(
    df: pl.DataFrame,
    output_path: str | Path,
    shard_size: int = 500_000,
    progress: bool = True,
) -> int:
    """Write action value data to .bag file(s).

    Args:
        df: DataFrame with columns: fen, move_uci, win_prob
        output_path: Base output path (will add shard suffixes if needed).
        shard_size: Number of records per shard.
        progress: Whether to show progress bar.

    Returns:
        Total number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coder = CODERS["action_value"]
    total_records = len(df)
    num_shards = (total_records + shard_size - 1) // shard_size

    logger.info(f"Writing {total_records:,} action-value records to {num_shards} shard(s)")

    records_written = 0

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total_records)
        shard_df = df.slice(start, end - start)

        # Determine shard path
        if num_shards > 1:
            shard_path = str(output_path).replace(
                "_data.bag",
                f"-{shard_idx:05d}-of-{num_shards:05d}_data.bag"
            )
            if shard_path == str(output_path):
                # If replacement didn't work, append shard suffix
                stem = output_path.stem
                shard_path = output_path.parent / f"{stem}-{shard_idx:05d}-of-{num_shards:05d}.bag"
        else:
            shard_path = output_path

        logger.info(f"Writing shard {shard_idx + 1}/{num_shards}: {shard_path}")

        with BagWriter(str(shard_path)) as writer:
            rows = shard_df.iter_rows(named=True)
            if progress:
                rows = tqdm(rows, total=len(shard_df), desc=f"Shard {shard_idx + 1}")

            for row in rows:
                record = ActionValueData(
                    fen=row["fen"],
                    move=row["move_uci"],
                    win_prob=row["win_prob"],
                )
                encoded = coder.encode(record)
                writer.write(encoded)
                records_written += 1

    logger.info(f"Wrote {records_written:,} action-value records")
    return records_written


def write_state_value_bag(
    df: pl.DataFrame,
    output_path: str | Path,
    progress: bool = True,
) -> int:
    """Write state value data to .bag file.

    Args:
        df: DataFrame with columns: fen, win_prob
        output_path: Output path for .bag file.
        progress: Whether to show progress bar.

    Returns:
        Total number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coder = CODERS["state_value"]
    total_records = len(df)

    logger.info(f"Writing {total_records:,} state-value records to {output_path}")

    with BagWriter(str(output_path)) as writer:
        rows = df.iter_rows(named=True)
        if progress:
            rows = tqdm(rows, total=total_records, desc="Writing state values")

        records_written = 0
        for row in rows:
            record = StateValueData(
                fen=row["fen"],
                win_prob=row["win_prob"],
            )
            encoded = coder.encode(record)
            writer.write(encoded)
            records_written += 1

    logger.info(f"Wrote {records_written:,} state-value records")
    return records_written


def write_behavioral_cloning_bag(
    df: pl.DataFrame,
    output_path: str | Path,
    progress: bool = True,
) -> int:
    """Write behavioral cloning data to .bag file.

    Args:
        df: DataFrame with columns: fen, move_uci
        output_path: Output path for .bag file.
        progress: Whether to show progress bar.

    Returns:
        Total number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coder = CODERS["behavioral_cloning"]
    total_records = len(df)

    logger.info(f"Writing {total_records:,} behavioral-cloning records to {output_path}")

    with BagWriter(str(output_path)) as writer:
        rows = df.iter_rows(named=True)
        if progress:
            rows = tqdm(rows, total=total_records, desc="Writing BC data")

        records_written = 0
        for row in rows:
            record = BehavioralCloningData(
                fen=row["fen"],
                move=row["move_uci"],
            )
            encoded = coder.encode(record)
            writer.write(encoded)
            records_written += 1

    logger.info(f"Wrote {records_written:,} behavioral-cloning records")
    return records_written


def convert_parquet_to_bag(
    input_path: str | Path,
    output_dir: str | Path,
    policy: str = "action_value",
    train_split: float = 0.95,
    shard_size: int = 500_000,
) -> dict:
    """Convert annotated Parquet to .bag files with train/test split.

    Args:
        input_path: Input Parquet file with annotated positions.
        output_dir: Output directory for .bag files.
        policy: "action_value", "state_value", or "behavioral_cloning".
        train_split: Fraction of data for training (rest is test).
        shard_size: Records per shard for action_value.

    Returns:
        Dictionary with statistics about written data.
    """
    logger.info(f"Loading annotated data from {input_path}")
    df = pl.read_parquet(input_path)

    # Shuffle and split
    df = df.sample(fraction=1.0, shuffle=True, seed=42)
    train_size = int(len(df) * train_split)
    train_df = df.head(train_size)
    test_df = df.tail(len(df) - train_size)

    logger.info(f"Split: {len(train_df):,} train, {len(test_df):,} test")

    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stats = {"policy": policy, "train": 0, "test": 0}

    if policy == "action_value":
        stats["train"] = write_action_value_bag(
            train_df,
            train_dir / "action_value_data.bag",
            shard_size=shard_size,
        )
        stats["test"] = write_action_value_bag(
            test_df,
            test_dir / "action_value_data.bag",
            shard_size=shard_size,
        )

    elif policy == "state_value":
        stats["train"] = write_state_value_bag(
            train_df,
            train_dir / "state_value_data.bag",
        )
        stats["test"] = write_state_value_bag(
            test_df,
            test_dir / "state_value_data.bag",
        )

    elif policy == "behavioral_cloning":
        stats["train"] = write_behavioral_cloning_bag(
            train_df,
            train_dir / "behavioral_cloning_data.bag",
        )
        stats["test"] = write_behavioral_cloning_bag(
            test_df,
            test_dir / "behavioral_cloning_data.bag",
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert annotated data to .bag files")
    parser.add_argument(
        "--input",
        type=str,
        default="data/antichess/annotated_positions.parquet",
        help="Input Parquet with annotated positions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/antichess",
        help="Output directory for .bag files",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["action_value", "state_value", "behavioral_cloning"],
        default="action_value",
        help="Type of data to write",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Records per shard for action_value (default: 500000)",
    )

    args = parser.parse_args()

    stats = convert_parquet_to_bag(
        input_path=args.input,
        output_dir=args.output_dir,
        policy=args.policy,
        train_split=args.train_split,
        shard_size=args.shard_size,
    )

    logger.info(f"Conversion complete: {stats}")


if __name__ == "__main__":
    main()
