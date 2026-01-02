# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Convert antichess_eval_worker JSONL files to .bag format for finetuning."""

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Add parent directory to path for imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bagz import BagWriter
from src.constants import CODERS, StateValueData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_jsonl_to_bags(
    input_dir: str | Path,
    output_dir: str | Path,
    train_split: float = 0.95,
    shard_size: int = 500_000,
) -> dict:
    """Convert JSONL files to .bag files for state_value training.

    The JSONL format is: ["Antichess: <FEN>", <win_prob>]

    Args:
        input_dir: Directory containing antichess_evals_*.jsonl files.
        output_dir: Output directory for .bag files.
        train_split: Fraction of data for training.
        shard_size: Records per shard.

    Returns:
        Dictionary with statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all JSONL files
    jsonl_files = sorted(input_dir.glob("antichess_evals_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No antichess_evals_*.jsonl files found in {input_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # Count total lines for progress
    total_lines = 0
    for f in jsonl_files:
        with open(f) as fp:
            total_lines += sum(1 for _ in fp)
    logger.info(f"Total records: {total_lines:,}")

    # Load all records
    logger.info("Loading records...")
    records = []
    for jsonl_file in jsonl_files:
        logger.info(f"  Reading {jsonl_file.name}")
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                fen_with_prefix = data[0]
                win_prob = data[1]

                # Strip "Antichess: " prefix
                if fen_with_prefix.startswith("Antichess: "):
                    fen = fen_with_prefix[len("Antichess: "):]
                else:
                    fen = fen_with_prefix

                records.append(StateValueData(fen=fen, win_prob=win_prob))

    logger.info(f"Loaded {len(records):,} records")

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(records)

    # Split into train/test
    train_size = int(len(records) * train_split)
    train_records = records[:train_size]
    test_records = records[train_size:]

    logger.info(f"Split: {len(train_records):,} train, {len(test_records):,} test")

    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    coder = CODERS["state_value"]

    def write_bags(records: list, output_path: Path, desc: str) -> int:
        """Write records to sharded .bag files."""
        total = len(records)
        num_shards = (total + shard_size - 1) // shard_size

        written = 0
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, total)
            shard_records = records[start:end]

            if num_shards > 1:
                shard_path = output_path.parent / f"state_value-{shard_idx:05d}-of-{num_shards:05d}_data.bag"
            else:
                shard_path = output_path

            logger.info(f"Writing {shard_path.name} ({len(shard_records):,} records)")

            with BagWriter(str(shard_path)) as writer:
                for record in tqdm(shard_records, desc=f"{desc} shard {shard_idx+1}/{num_shards}"):
                    encoded = coder.encode(record)
                    writer.write(encoded)
                    written += 1

        return written

    # Write train and test bags
    stats = {
        "train": write_bags(train_records, train_dir / "state_value_data.bag", "Train"),
        "test": write_bags(test_records, test_dir / "state_value_data.bag", "Test"),
    }

    logger.info(f"Done! Train: {stats['train']:,}, Test: {stats['test']:,}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert antichess JSONL evals to .bag format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/antichess",
        help="Directory with antichess_evals_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/antichess",
        help="Output directory for .bag files",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Records per .bag shard",
    )

    args = parser.parse_args()
    convert_jsonl_to_bags(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
