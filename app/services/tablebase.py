"""Endgame technique errors, verified against perfect play.

Everywhere else the app says a move "lost 40 points of expected score", which is
an estimate from a search at a fixed depth. Here it can say the game was drawn
and is now lost. That is not an evaluation, it is the result, and it is the one
place coaching can be completely certain.

Only the flips matter. A player grinding a lost position for thirty moves has
made no errors worth naming; the move that turned a draw into a loss is the whole
lesson, and it comes with a named ending the player can go and study.
"""

import logging

import chess

from app.clients.lichess_tablebase import complete, in_range, lichess_tablebase_client
from app.db import get_db

logger = logging.getLogger(__name__)

# Scored on a scale symmetric about zero, so switching whose point of view we are
# taking is exactly a negation. That symmetry is the whole point: a draw has to
# negate to a draw. Ranking these on 0..3 and inverting with `3 - rank` sends draw
# to blessed-loss, which reported "you threw away a draw" on moves that were the
# tablebase's own first choice.
_OUTCOME_VALUE = {
    "win": 2,
    "cursed-win": 1,   # winnable, but the fifty-move rule saves the defender
    "draw": 0,
    "blessed-loss": -1,  # lost, but the fifty-move rule rescues it
    "loss": -2,
}

# Named by material class rather than by exact signature. An exact lookup table
# misses almost everything -- a rook ending with three pawns a side is still a rook
# ending, and "KRPPkrp" is not a key anyone would think to add. Silman's endgame
# course is organised by rating for the same reason these names are kept plain:
# telling a 1500 about Vancura is not coaching, telling them Philidor is.
def name_ending(fen: str) -> str | None:
    """A studiable name for the material on the board, when there is one."""
    board = fen.split()[0]
    white = [c for c in board if c.isupper() and c != "K"]
    black = [c.upper() for c in board if c.islower() and c != "k"]

    if not white and not black:
        return "bare kings"

    # One side down to a bare king is a technique exercise, not an endgame type,
    # and it must be checked per side: looking at the combined material calls
    # king-and-rook-versus-king a "rook endgame".
    for mine, theirs in ((white, black), (black, white)):
        if theirs:
            continue
        heavy = sorted({c for c in mine if c in "QRBN"})
        if not heavy:
            return "king and pawn versus king"
        if mine.count("P"):
            names = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
            return f"{names[heavy[0]]} and pawn versus king" if len(heavy) == 1 else None
        return "mating with " + " and ".join(
            {"Q": "a queen", "R": "a rook", "B": "bishops", "N": "knights"}[c] for c in heavy
        )

    w_heavy = {c for c in white if c in "QRBN"}
    b_heavy = {c for c in black if c in "QRBN"}
    pawns = white.count("P") + black.count("P")

    if not w_heavy and not b_heavy:
        return "king and pawn endgame"
    if w_heavy != b_heavy:
        return None  # an imbalance, not a named ending type

    if w_heavy == {"R"}:
        if len(white) + len(black) - pawns == 2 and pawns == 1:
            return "rook and pawn versus rook (Lucena and Philidor)"
        return "rook endgame"
    if w_heavy == {"Q"}:
        return "queen endgame"
    if w_heavy == {"B"}:
        colours = _bishop_colours(fen)
        if len(colours) == 2 and len(colours[0] | colours[1]) == 2:
            return "opposite-coloured bishops"
        return "bishop endgame"
    if w_heavy == {"N"}:
        return "knight endgame"
    return None


def _bishop_colours(fen: str) -> list[set[bool]]:
    """The square colours each side's bishops stand on."""
    board = chess.Board(complete(fen))
    out = []
    for colour in (chess.WHITE, chess.BLACK):
        squares = board.pieces(chess.BISHOP, colour)
        if squares:
            out.append({bool((chess.square_rank(s) + chess.square_file(s)) % 2) for s in squares})
    return out


async def _cached(fen: str) -> str | None:
    db = get_db()
    async with db.execute(
        "SELECT category FROM tablebase_cache WHERE fen = ?", (fen,)
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None


async def _save(fen: str, category: str, dtz) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO tablebase_cache (fen, category, dtz) VALUES (?, ?, ?)",
        (fen, category, dtz),
    )
    await db.commit()


async def category(fen: str) -> str | None:
    """Theoretical result for the side to move: win, draw, loss, or a cursed variant."""
    if not in_range(fen):
        return None

    cached = await _cached(fen)
    if cached is not None:
        return cached or None

    try:
        result = await lichess_tablebase_client.lookup(fen)
    except Exception as exc:
        logger.warning("Tablebase lookup failed for %s: %s: %s", fen, type(exc).__name__, exc)
        return None
    if not result or not result.get("category"):
        return None

    await _save(fen, result["category"], result.get("dtz"))
    return result["category"]


def _standing(outcome: str | None, mover_is_white: bool, white_to_move: bool):
    """Scores an outcome from one fixed player's point of view, in [-2, +2]."""
    if outcome is None:
        return None
    value = _OUTCOME_VALUE.get(outcome)
    if value is None:  # "unknown", which is what castling rights produce
        return None
    # The category is always "for the side to move", so it is negated whenever the
    # side to move is not the player being coached.
    return value if mover_is_white == white_to_move else -value


async def review(moves: list[dict], colour: str) -> list[dict]:
    """Every move that changed the theoretical result, with the ending named.

    `moves` are the player's own annotated moves. Each needs `fen_before` and
    `fen_after`; anything outside tablebase range is skipped without a lookup.
    """
    mover_is_white = colour == "white"
    flips: list[dict] = []

    for move in moves:
        before_fen, after_fen = move.get("fen_before"), move.get("fen_after")
        if not before_fen or not after_fen:
            continue
        if not in_range(before_fen) or not in_range(after_fen):
            continue

        before = await category(before_fen)
        after = await category(after_fen)
        if before is None or after is None:
            continue

        was = _standing(before, mover_is_white, chess.Board(before_fen).turn == chess.WHITE)
        now = _standing(after, mover_is_white, chess.Board(after_fen).turn == chess.WHITE)
        if was is None or now is None or now >= was:
            continue

        flips.append(
            {
                "game_id": move.get("game_id"),
                "ply": move.get("ply"),
                "move": f"{move.get('move_number')}. {move.get('played_san')}",
                "was": "winning" if was > 0 else "drawn",
                "became": "drawn" if now == 0 else "lost",
                "ending": name_ending(before_fen),
                "engine_preferred": move.get("best_san"),
                "certainty": (
                    "This is the theoretical result with perfect play, not an "
                    "engine estimate. The result really did change on this move."
                ),
            }
        )

    return flips


def summary(flips: list[dict]) -> dict | None:
    """Aggregated technique errors, or None when there were none to find."""
    if not flips:
        return None
    return {
        "result_changing_endgame_moves": len(flips),
        "wins_drawn": sum(1 for f in flips if f["was"] == "winning" and f["became"] == "drawn"),
        "draws_lost": sum(1 for f in flips if f["was"] == "drawn" and f["became"] == "lost"),
        "wins_lost": sum(1 for f in flips if f["was"] == "winning" and f["became"] == "lost"),
        "examples": flips[:6],
    }
