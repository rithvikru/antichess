# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Main pipeline orchestration for antichess data generation."""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

import chess.variant
import polars as pl
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from antichess_data.annotate import FairyStockfishAnnotator
from antichess_data.load_games import load_and_filter_games, load_filtered_games
from antichess_data.parse_games import parse_antichess_movetext
from antichess_data.utils import MIN_ELO_DEFAULT, MAX_GAMES_DEFAULT, TIME_LIMIT_MS_DEFAULT
from antichess_data.write_bags import convert_parquet_to_bag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_game_chunk(
    games_chunk: list[dict],
    engine_path: str,
    time_limit_ms: int,
    policy: str,
) -> list[dict]:
    """Process a chunk of games: parse and annotate.

    This function is designed to be run in a worker process.

    Args:
        games_chunk: List of game dictionaries with 'movetext' key.
        engine_path: Path to Fairy-Stockfish binary.
        time_limit_ms: Time limit per position.
        policy: "action_value", "state_value", or "behavioral_cloning".

    Returns:
        List of annotated position dictionaries.
    """
    results = []

    # Create annotator for this worker (if needed for annotation)
    annotator = None
    if policy in ("action_value", "state_value", "all_action_values"):
        try:
            annotator = FairyStockfishAnnotator(engine_path, time_limit_ms)
        except Exception as e:
            logger.error(f"Failed to create annotator: {e}")
            return []

    try:
        for game in games_chunk:
            movetext = game.get("movetext", "")
            if not movetext:
                continue

            # Parse game into positions
            positions = parse_antichess_movetext(movetext)

            for pos in positions:
                try:
                    board = chess.variant.GiveawayBoard(pos.fen)

                    if policy == "behavioral_cloning":
                        # No annotation needed for BC
                        results.append({
                            "fen": pos.fen,
                            "move_uci": pos.move_uci,
                        })

                    elif policy == "state_value":
                        win_prob = annotator.get_state_value(board)
                        results.append({
                            "fen": pos.fen,
                            "win_prob": win_prob,
                        })

                    elif policy == "action_value":
                        # Annotate just the played move
                        move = chess.Move.from_uci(pos.move_uci)
                        if move in board.legal_moves:
                            win_prob = annotator.get_action_value(board, move)
                            results.append({
                                "fen": pos.fen,
                                "move_uci": pos.move_uci,
                                "win_prob": win_prob,
                            })

                    elif policy == "all_action_values":
                        # Annotate all legal moves
                        for move_uci, win_prob in annotator.get_all_action_values(board):
                            results.append({
                                "fen": pos.fen,
                                "move_uci": move_uci,
                                "win_prob": win_prob,
                            })

                except Exception as e:
                    logger.debug(f"Failed to process position: {e}")
                    continue

    finally:
        if annotator:
            annotator.close()

    return results


    def run_pipeline(
    output_dir: str | Path,
    engine_path: str,
    min_elo: int = MIN_ELO_DEFAULT,
    max_games: int = MAX_GAMES_DEFAULT,
    time_limit_ms: int = TIME_LIMIT_MS_DEFAULT,
    policy: str = "action_value",
    num_workers: int = 1,
    chunk_size: int = 100,
    skip_download: bool = False,
    skip_annotation: bool = False,
    train_split: float = 0.95,
    shard_size: int = 500_000,
    pgn_path: str | Path | None = None,
    pgn_url: str | None = None,
    lichess_month: str | None = None,
    lichess_months: list[str] | None = None,
    lichess_base_url: str | None = None,
    auto_backfill: bool = False,
    max_backfill_months: int = 120,
):
    """Run the full antichess data generation pipeline.

    Args:
        output_dir: Directory for all output files.
        engine_path: Path to Fairy-Stockfish binary.
        min_elo: Minimum Elo filter.
        max_games: Maximum games to process.
        time_limit_ms: Time limit per position for engine.
        policy: Data policy type.
        num_workers: Number of parallel workers.
        chunk_size: Games per worker chunk.
        skip_download: Skip Lichess download (use existing).
        skip_annotation: Skip annotation (use existing annotated data).
        train_split: Train/test split ratio.
        shard_size: Records per .bag shard.
        pgn_path: Local PGN or PGN.BZ2 path (optional).
        pgn_url: Direct URL to a Lichess PGN.BZ2 file (optional).
        lichess_month: Month for Lichess dump (YYYY-MM).
        lichess_months: Explicit list of months to fetch.
        lichess_base_url: Base URL for Lichess antichess database.
        auto_backfill: Fetch previous months until max_games reached.
        max_backfill_months: Max months to scan when auto_backfill is enabled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games_path = output_dir / "filtered_games.parquet"
    annotated_path = output_dir / f"annotated_{policy}.parquet"

    # Step 1: Download and filter games
    if not skip_download:
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading and filtering games from Lichess")
        logger.info("=" * 60)

        load_kwargs = dict(
            output_path=games_path,
            min_elo=min_elo,
            max_games=max_games,
            pgn_path=pgn_path,
            pgn_url=pgn_url,
            lichess_month=lichess_month,
            lichess_months=lichess_months,
        )
        if lichess_base_url is not None:
            load_kwargs["lichess_base_url"] = lichess_base_url
        if auto_backfill:
            load_kwargs["auto_backfill"] = auto_backfill
            load_kwargs["max_backfill_months"] = max_backfill_months
        load_and_filter_games(**load_kwargs)
    else:
        logger.info("Skipping download, using existing games file")
        if not games_path.exists():
            raise FileNotFoundError(f"Games file not found: {games_path}")

    # Step 2: Parse and annotate
    if not skip_annotation:
        logger.info("=" * 60)
        logger.info("STEP 2: Parsing games and annotating positions")
        logger.info("=" * 60)

        # Load games
        games_df = pl.read_parquet(games_path)
        games = games_df.to_dicts()
        logger.info(f"Loaded {len(games):,} games")

        # Split into chunks
        chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]
        logger.info(f"Processing {len(chunks):,} chunks with {num_workers} worker(s)")

        # Process chunks
        all_results = []

        if num_workers == 1:
            # Single-threaded processing
            for chunk in tqdm(chunks, desc="Processing chunks"):
                results = process_game_chunk(
                    chunk, engine_path, time_limit_ms, policy
                )
                all_results.extend(results)
        else:
            # Multi-process
            worker_fn = partial(
                process_game_chunk,
                engine_path=engine_path,
                time_limit_ms=time_limit_ms,
                policy=policy,
            )

            with mp.Pool(num_workers) as pool:
                for results in tqdm(
                    pool.imap(worker_fn, chunks),
                    total=len(chunks),
                    desc="Processing chunks",
                ):
                    all_results.extend(results)

        logger.info(f"Generated {len(all_results):,} annotated positions")

        # Save to parquet
        results_df = pl.DataFrame(all_results)
        results_df.write_parquet(annotated_path)
        logger.info(f"Saved annotated data to {annotated_path}")

    else:
        logger.info("Skipping annotation, using existing annotated file")
        if not annotated_path.exists():
            raise FileNotFoundError(f"Annotated file not found: {annotated_path}")

    # Step 3: Convert to .bag files
    logger.info("=" * 60)
    logger.info("STEP 3: Converting to .bag files")
    logger.info("=" * 60)

    stats = convert_parquet_to_bag(
        input_path=annotated_path,
        output_dir=output_dir,
        policy=policy,
        train_split=train_split,
        shard_size=shard_size,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training records: {stats['train']:,}")
    logger.info(f"Test records: {stats['test']:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate antichess training data for Searchless Chess",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/antichess",
        help="Output directory for all files",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        required=True,
        help="Path to Fairy-Stockfish binary",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=MIN_ELO_DEFAULT,
        help="Minimum Elo for both players",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=MAX_GAMES_DEFAULT,
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--time-limit-ms",
        type=int,
        default=TIME_LIMIT_MS_DEFAULT,
        help="Engine time limit per position in milliseconds",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["action_value", "state_value", "behavioral_cloning", "all_action_values"],
        default="action_value",
        help="Type of data to generate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (each spawns own engine)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Games per worker chunk",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading from Lichess (use existing file)",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation (use existing annotated file)",
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
        help="Records per .bag shard for action_value",
    )
    parser.add_argument(
        "--pgn-path",
        type=str,
        default=None,
        help="Path to local PGN or PGN.BZ2 file",
    )
    parser.add_argument(
        "--pgn-url",
        type=str,
        default=None,
        help="Direct URL to a Lichess PGN.BZ2 file",
    )
    parser.add_argument(
        "--lichess-month",
        type=str,
        default=None,
        help="Month for Lichess dump (YYYY-MM, default: current UTC month)",
    )
    parser.add_argument(
        "--lichess-months",
        type=str,
        default=None,
        help="Comma-separated list of months (YYYY-MM) to fetch",
    )
    parser.add_argument(
        "--lichess-base-url",
        type=str,
        default=None,
        help="Base URL for Lichess antichess database",
    )
    parser.add_argument(
        "--auto-backfill",
        action="store_true",
        help="Fetch previous months until max-games is reached",
    )
    parser.add_argument(
        "--max-backfill-months",
        type=int,
        default=120,
        help="Max months to scan when auto-backfill is enabled",
    )

    args = parser.parse_args()

    # Validate engine path
    if not os.path.isfile(args.engine_path):
        parser.error(f"Engine not found: {args.engine_path}")

    run_pipeline(
        output_dir=args.output_dir,
        engine_path=args.engine_path,
        min_elo=args.min_elo,
        max_games=args.max_games,
        time_limit_ms=args.time_limit_ms,
        policy=args.policy,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        skip_download=args.skip_download,
        skip_annotation=args.skip_annotation,
        train_split=args.train_split,
        shard_size=args.shard_size,
        pgn_path=args.pgn_path,
        pgn_url=args.pgn_url,
        lichess_month=args.lichess_month,
        lichess_months=(
            [m.strip() for m in args.lichess_months.split(",") if m.strip()]
            if args.lichess_months
            else None
        ),
        lichess_base_url=args.lichess_base_url,
        auto_backfill=args.auto_backfill,
        max_backfill_months=args.max_backfill_months,
    )


if __name__ == "__main__":
    main()
