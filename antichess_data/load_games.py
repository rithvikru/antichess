# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Load and filter antichess games from the Lichess database."""

import argparse
import bz2
import datetime as dt
import logging
import re
import urllib.request
from urllib import error as urlerror
from pathlib import Path

import chess.pgn
import polars as pl
from tqdm import tqdm

from antichess_data.utils import MIN_ELO_DEFAULT, MAX_GAMES_DEFAULT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LICHESS_BASE_URL = "https://database.lichess.org/antichess"


def _default_month() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m")


def _normalize_month(month: str | None) -> str:
    if month is None or month == "latest":
        return _default_month()
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"Invalid month format: {month!r}. Expected YYYY-MM.")
    return month


def _build_lichess_url(base_url: str, month: str) -> str:
    return f"{base_url}/lichess_db_antichess_rated_{month}.pgn.bz2"


def _download_pgn(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info(f"Using cached PGN: {output_path}")
        return output_path

    logger.info(f"Downloading PGN from {url}")
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as out_f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    logger.info(f"Saved PGN to {output_path}")
    return output_path


def _open_pgn(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_and_filter_games(
    output_path: str | Path,
    min_elo: int = MIN_ELO_DEFAULT,
    max_games: int = MAX_GAMES_DEFAULT,
    termination: str = "Normal",
    pgn_path: str | Path | None = None,
    pgn_url: str | None = None,
    lichess_month: str | None = None,
    lichess_months: list[str] | None = None,
    lichess_base_url: str = DEFAULT_LICHESS_BASE_URL,
    auto_backfill: bool = False,
    max_backfill_months: int = 120,
) -> pl.DataFrame:
    """Load antichess games from Lichess and filter by criteria.

    Args:
        output_path: Path to save filtered games as Parquet.
        min_elo: Minimum Elo for both players.
        max_games: Maximum number of games to sample.
        termination: Game termination type to filter for.
        pgn_path: Local PGN or PGN.BZ2 path (optional).
        pgn_url: Direct URL to a PGN.BZ2 file (optional).
        lichess_month: Month for Lichess dump (YYYY-MM). Defaults to current UTC month.
        lichess_months: Explicit list of months (YYYY-MM) to fetch.
        lichess_base_url: Base URL for Lichess antichess database.
        auto_backfill: Fetch previous months until max_games reached.
        max_backfill_months: Max months to scan when auto_backfill is True.

    Returns:
        Filtered DataFrame with game data.
    """
    logger.info("Loading antichess dataset from Lichess...")
    logger.info(
        "Filters: min_elo=%s, max_games=%s, termination=%s",
        min_elo,
        f"{max_games:,}",
        termination,
    )

    output_path = Path(output_path)
    if max_games <= 0:
        raise ValueError("max_games must be a positive integer.")

    months_to_fetch: list[str] = []

    if pgn_path is None and pgn_url is None:
        if lichess_months:
            months_to_fetch = [_normalize_month(m) for m in lichess_months]
        elif auto_backfill:
            start_month = _normalize_month(lichess_month)
            year, month = map(int, start_month.split("-"))
            current = dt.date(year, month, 1)
            for _ in range(max_backfill_months):
                months_to_fetch.append(current.strftime("%Y-%m"))
                if current.month == 1:
                    current = dt.date(current.year - 1, 12, 1)
                else:
                    current = dt.date(current.year, current.month - 1, 1)
        else:
            months_to_fetch = [_normalize_month(lichess_month)]

    exporter = chess.pgn.StringExporter(
        headers=False, variations=False, comments=False
    )
    records: list[dict] = []
    scanned = 0

    def process_pgn(pgn_file, label: str) -> bool:
        nonlocal scanned
        for game in tqdm(
            iter(lambda: chess.pgn.read_game(pgn_file), None),
            desc=f"Scanning games ({label})",
        ):
            scanned += 1

            headers = game.headers
            variant = headers.get("Variant")
            if variant and variant.lower() not in ("antichess", "giveaway"):
                continue

            white_elo = _parse_int(headers.get("WhiteElo"))
            black_elo = _parse_int(headers.get("BlackElo"))
            if white_elo is None or black_elo is None:
                continue
            if white_elo < min_elo or black_elo < min_elo:
                continue

            term = headers.get("Termination", "")
            if termination and term != termination:
                continue

            movetext = game.accept(exporter).strip()
            if not movetext or len(movetext) <= 10:
                continue

            records.append({
                "White": headers.get("White", ""),
                "Black": headers.get("Black", ""),
                "WhiteElo": white_elo,
                "BlackElo": black_elo,
                "Result": headers.get("Result", ""),
                "movetext": movetext,
                "Termination": term,
            })

            if len(records) >= max_games:
                return False
        return True

    if pgn_path is not None or pgn_url is not None:
        if pgn_path is None:
            pgn_path = output_path.parent / Path(pgn_url).name
            pgn_path = _download_pgn(pgn_url, Path(pgn_path))
        else:
            pgn_path = Path(pgn_path)
            if not pgn_path.exists():
                raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        logger.info(f"Parsing PGN from {pgn_path}")
        with _open_pgn(Path(pgn_path)) as pgn_file:
            process_pgn(pgn_file, Path(pgn_path).name)
    else:
        for month in months_to_fetch:
            url = _build_lichess_url(lichess_base_url, month)
            pgn_path = output_path.parent / Path(url).name
            try:
                pgn_path = _download_pgn(url, Path(pgn_path))
            except urlerror.HTTPError as exc:
                if exc.code == 404 and auto_backfill:
                    logger.warning(f"No PGN found for {month}; stopping backfill.")
                    break
                raise

            logger.info(f"Parsing PGN from {pgn_path}")
            with _open_pgn(Path(pgn_path)) as pgn_file:
                should_continue = process_pgn(pgn_file, month)
            if not should_continue:
                break

    logger.info(f"Scanned {scanned:,} games, kept {len(records):,}")

    filtered = pl.DataFrame(records)

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
        default=DEFAULT_LICHESS_BASE_URL,
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

    load_and_filter_games(
        output_path=args.output,
        min_elo=args.min_elo,
        max_games=args.max_games,
        termination=args.termination,
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
