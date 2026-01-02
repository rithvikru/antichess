# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Parse antichess PGN games into positions using python-chess."""

import argparse
import io
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import chess.pgn
import chess.variant
import polars as pl
from tqdm import tqdm

from antichess_data.utils import PositionData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GamePositions(NamedTuple):
    """Positions extracted from a single game."""
    positions: list[PositionData]
    result: str  # "1-0", "0-1", or "1/2-1/2"
    white_elo: int
    black_elo: int


def parse_antichess_movetext(movetext: str) -> list[PositionData]:
    """Parse antichess movetext to list of (fen, move_uci) positions.

    Args:
        movetext: PGN movetext string (e.g., "1. e3 e6 2. Qg4 Qh4")

    Returns:
        List of PositionData tuples with FEN and UCI move.
    """
    # Create antichess board (GiveawayBoard)
    board = chess.variant.GiveawayBoard()
    positions = []

    # Parse PGN - need to wrap movetext with variant header
    pgn_str = f'[Variant "Antichess"]\n\n{movetext}'
    pgn = io.StringIO(pgn_str)

    try:
        game = chess.pgn.read_game(pgn)
        if game is None:
            return []

        # Iterate through all moves
        for move in game.mainline_moves():
            # Record position BEFORE the move
            fen = board.fen()
            move_uci = move.uci()

            positions.append(PositionData(fen=fen, move_uci=move_uci))

            # Apply the move
            board.push(move)

    except Exception as e:
        # Log but don't fail on corrupted games
        logger.debug(f"Failed to parse game: {e}")
        return []

    return positions


def parse_games_from_dataframe(
    df: pl.DataFrame,
    progress: bool = True,
) -> Iterator[GamePositions]:
    """Parse all games from a DataFrame into positions.

    Args:
        df: DataFrame with columns: movetext, Result, WhiteElo, BlackElo
        progress: Whether to show progress bar.

    Yields:
        GamePositions for each successfully parsed game.
    """
    rows = df.iter_rows(named=True)
    if progress:
        rows = tqdm(rows, total=len(df), desc="Parsing games")

    parsed_count = 0
    failed_count = 0

    for row in rows:
        movetext = row["movetext"]
        if not movetext:
            failed_count += 1
            continue

        positions = parse_antichess_movetext(movetext)

        if not positions:
            failed_count += 1
            continue

        parsed_count += 1
        yield GamePositions(
            positions=positions,
            result=row["Result"],
            white_elo=row["WhiteElo"],
            black_elo=row["BlackElo"],
        )

    logger.info(f"Successfully parsed {parsed_count:,} games, {failed_count:,} failed")


def extract_all_positions(
    games_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """Extract all positions from filtered games and save to Parquet.

    Args:
        games_path: Path to filtered games Parquet file.
        output_path: Path to save extracted positions.
        limit: Maximum number of games to process (None for all).

    Returns:
        Total number of positions extracted.
    """
    logger.info(f"Loading games from {games_path}")
    df = pl.read_parquet(games_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Processing first {limit:,} games")

    # Collect all positions
    all_positions = []
    total_games = 0

    for game_positions in parse_games_from_dataframe(df):
        total_games += 1
        for pos in game_positions.positions:
            all_positions.append({
                "fen": pos.fen,
                "move_uci": pos.move_uci,
            })

    logger.info(f"Extracted {len(all_positions):,} positions from {total_games:,} games")

    # Save to parquet
    positions_df = pl.DataFrame(all_positions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions_df.write_parquet(output_path)
    logger.info(f"Saved positions to {output_path}")

    return len(all_positions)


def main():
    parser = argparse.ArgumentParser(description="Parse antichess games into positions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/antichess/filtered_games.parquet",
        help="Input path for filtered games",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/antichess/positions.parquet",
        help="Output path for extracted positions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of games to process (for testing)",
    )

    args = parser.parse_args()

    extract_all_positions(
        games_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
