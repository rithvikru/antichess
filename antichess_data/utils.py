# Copyright 2024
# Licensed under the Apache License, Version 2.0

"""Shared utilities for antichess data generation."""

import math
from typing import NamedTuple


class PositionData(NamedTuple):
    """A single position extracted from a game."""
    fen: str
    move_uci: str


class AnnotatedPosition(NamedTuple):
    """A position with engine annotations."""
    fen: str
    move_uci: str
    win_prob: float


def centipawns_to_win_prob(centipawns: int) -> float:
    """Convert centipawns to win probability [0, 1].

    Reference: https://lichess.org/page/accuracy
    This is the same formula used in the original Searchless Chess repo.

    Args:
        centipawns: The chess score in centipawns.

    Returns:
        Win probability in [0, 1].
    """
    return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


def centipawns_to_win_prob_antichess(centipawns: int) -> float:
    """Convert centipawns to win probability for antichess.

    In antichess, the goal is to lose all pieces. Fairy-Stockfish evaluates
    from a standard perspective where having more material is good.
    We need to INVERT the score: losing material should give high win_prob.

    Args:
        centipawns: The chess score in centipawns from Fairy-Stockfish.

    Returns:
        Win probability in [0, 1], where higher = better for antichess.
    """
    # Invert: negative cp (losing material) becomes positive (good in antichess)
    return centipawns_to_win_prob(-centipawns)


def mate_to_centipawns(mate_in: int) -> int:
    """Convert mate-in-N to centipawn equivalent.

    In antichess, "mate" means losing all pieces or being stalemated,
    which is the GOAL. So positive mate_in (we will "win") is good.

    Args:
        mate_in: Positive if we're mating, negative if being mated.

    Returns:
        Centipawn equivalent (before antichess inversion).
    """
    if mate_in > 0:
        # We will "win" (lose all pieces) - this is GOOD in antichess
        # Return negative cp so after inversion it becomes high win_prob
        return -10000
    else:
        # Opponent will "win" - BAD for us in antichess
        # Return positive cp so after inversion it becomes low win_prob
        return 10000


# Constants
MIN_ELO_DEFAULT = 1800
MAX_GAMES_DEFAULT = 1_000_000
TIME_LIMIT_MS_DEFAULT = 50
NUM_RETURN_BUCKETS = 128
