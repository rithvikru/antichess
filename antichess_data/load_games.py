# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Load and filter antichess games from HuggingFace dataset."""

import argparse
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from antichess_data.utils import MIN_ELO_DEFAULT, MAX_GAMES_DEFAULT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_filter_games(
    output_path: str | Path,
    min_elo: int = MIN_ELO_DEFAULT,
    max_games: int = MAX_GAMES_DEFAULT,
    termination: str = "Normal",
) -> pl.DataFrame:
    """Load antichess games from HuggingFace and filter by criteria.

    Args:
        output_path: Path to save filtered games as Parquet.
        min_elo: Minimum Elo for both players.
        max_games: Maximum number of games to sample.
        termination: Game termination type to filter for.

    Returns:
        Filtered DataFrame with game data.
    """
    logger.info(f"Loading antichess dataset from HuggingFace...")
    logger.info(f"Filters: min_elo={min_elo}, max_games={max_games}, termination={termination}")

    # Load dataset using polars with HuggingFace integration
    # The dataset is ~15.7GB so we use lazy evaluation
    try:
        # Try loading via HuggingFace datasets first for better caching
        from datasets import load_dataset

        logger.info("Loading dataset via HuggingFace datasets library...")
        dataset = load_dataset(
            "Lichess/antichess-chess-games",
            split="train",
            streaming=False,  # Need full dataset for filtering
        )

        # Convert to polars
        logger.info("Converting to Polars DataFrame...")
        df = pl.from_arrow(dataset.data.table)

    except Exception as e:
        logger.warning(f"HuggingFace loading failed: {e}")
        logger.info("Trying direct parquet loading...")
        # Fallback to direct parquet loading
        df = pl.read_parquet("hf://datasets/Lichess/antichess-chess-games/data/*.parquet")

    logger.info(f"Loaded {len(df):,} total games")

    # Apply filters
    logger.info("Applying filters...")
    filtered = df.filter(
        (pl.col("WhiteElo") >= min_elo) &
        (pl.col("BlackElo") >= min_elo) &
        (pl.col("Termination") == termination) &
        (pl.col("movetext").is_not_null()) &
        (pl.col("movetext").str.len_chars() > 10)  # Filter out very short games
    )

    logger.info(f"After Elo filter (>={min_elo}): {len(filtered):,} games")

    # Sample if we have more than max_games
    if len(filtered) > max_games:
        logger.info(f"Sampling {max_games:,} games from {len(filtered):,}...")
        filtered = filtered.sample(n=max_games, seed=42)
    else:
        logger.info(f"Using all {len(filtered):,} filtered games (less than {max_games:,})")

    # Select only columns we need
    filtered = filtered.select([
        "White",
        "Black",
        "WhiteElo",
        "BlackElo",
        "Result",
        "movetext",
        "Termination",
    ])

    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(output_path)
    logger.info(f"Saved {len(filtered):,} games to {output_path}")

    return filtered


def load_filtered_games(path: str | Path) -> pl.DataFrame:
    """Load previously filtered games from Parquet file.

    Args:
        path: Path to the filtered games Parquet file.

    Returns:
        DataFrame with filtered games.
    """
    return pl.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Load and filter antichess games")
    parser.add_argument(
        "--output",
        type=str,
        default="data/antichess/filtered_games.parquet",
        help="Output path for filtered games",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=MIN_ELO_DEFAULT,
        help=f"Minimum Elo for both players (default: {MIN_ELO_DEFAULT})",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=MAX_GAMES_DEFAULT,
        help=f"Maximum number of games to sample (default: {MAX_GAMES_DEFAULT:,})",
    )
    parser.add_argument(
        "--termination",
        type=str,
        default="Normal",
        help="Game termination type to filter for (default: Normal)",
    )

    args = parser.parse_args()

    load_and_filter_games(
        output_path=args.output,
        min_elo=args.min_elo,
        max_games=args.max_games,
        termination=args.termination,
    )


if __name__ == "__main__":
    main()
