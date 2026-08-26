"""Why a mistake happened, not just how much it cost.

A blunder is a symptom. Underneath it sits one of several different habits, and
they need opposite advice: a piece left hanging is a discipline problem, a
punishment that was already threatened before the move is a prophylaxis problem,
and a position where only one move held and the player found none of them is
simply a hard position. Centipawns cannot tell these apart. These probes can.

Two extra engine passes per position are needed, so they are run only on the
moments actually being sent to the model -- a dozen or so per report rather than
several thousand plies -- and cached in `move_probes` by (fen, depth) like every
other engine result, so a position probed once is never probed again.
"""

import asyncio
import logging

import chess

from app.config import settings
from app.db import get_db
from app.services.classify import PIECE_VALUES, static_exchange
from app.services.stockfish import stockfish_engine

logger = logging.getLogger(__name__)

# Expected-score gap between the best move and the second best, above which the
# position had essentially one answer. Expressed in centipawns because that is
# what the engine reports for the alternatives.
ONLY_MOVE_CP = 150


async def _cached(fen: str, depth: int) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT second_best_move, second_best_score_cp, null_move_threat "
        "FROM move_probes WHERE fen = ? AND depth = ?",
        (fen, depth),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "second_best_move": row[0],
        "second_best_score_cp": row[1],
        "null_move_threat": row[2],
    }


async def _save(fen: str, depth: int, probe: dict) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO move_probes "
        "(fen, depth, second_best_move, second_best_score_cp, null_move_threat) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            fen,
            depth,
            probe.get("second_best_move"),
            probe.get("second_best_score_cp"),
            probe.get("null_move_threat"),
        ),
    )
    await db.commit()


async def probe(fen: str, depth: int | None = None) -> dict:
    """Second-best move and standing threat for one position, cached."""
    depth = depth or settings.stockfish_default_depth
    cached = await _cached(fen, depth)
    if cached is not None:
        return cached

    probe_result: dict = {}
    try:
        alternatives = await stockfish_engine.analyse_alternatives(fen, depth)
        if len(alternatives) > 1:
            probe_result["second_best_move"] = alternatives[1]["move"]
            probe_result["second_best_score_cp"] = alternatives[1]["score_cp"]
        threat = await stockfish_engine.threat_after_null_move(fen, depth)
        if threat:
            probe_result["null_move_threat"] = threat["move"]
    except AssertionError:
        # The test suite's "no engine" guard. Swallowing it would let a test launch
        # Stockfish and still pass, which is the exact failure the guard exists for.
        raise
    except Exception as exc:  # engine unavailable or a position it will not take
        logger.warning("Probe failed for %s: %s: %s", fen, type(exc).__name__, exc)
        return {}

    await _save(fen, depth, probe_result)
    return probe_result


def etiology(move: dict, probe_result: dict) -> dict:
    """Names the habit behind a mistake from the probe and the move's own facts.

    Every field here is a fact about the position, not a diagnosis of the player.
    The model is told which cure each one implies; this only reports what is true.
    """
    out: dict = {}
    fen = move.get("fen_before")
    if not fen:
        return out

    board = chess.Board(fen)
    try:
        played = chess.Move.from_uci(move["played_uci"])
    except (ValueError, KeyError):
        return out
    if played not in board.legal_moves:
        return out

    # Counting: the capture simply loses material once the dust settles.
    if board.is_capture(played):
        exchange = static_exchange(board, played)
        out["exchange_value_cp"] = exchange
        if exchange < -PIECE_VALUES[chess.PAWN]:
            out["lost_a_counting_exchange"] = True

    # Criticality: was there another move that held, or was this the only one?
    best_score = move.get("eval_before_cp")
    second = probe_result.get("second_best_score_cp")
    if best_score is not None and second is not None:
        mover_is_white = board.turn == chess.WHITE
        gap = (best_score - second) if mover_is_white else (second - best_score)
        out["second_best_costs_cp"] = abs(gap)
        out["only_one_move_held"] = abs(gap) >= ONLY_MOVE_CP

    # Prophylaxis: was the punishment already threatened before this move?
    threat_uci = probe_result.get("null_move_threat")
    if threat_uci:
        try:
            after_null = board.copy(stack=False)
            after_null.push(chess.Move.null())
            threat_move = chess.Move.from_uci(threat_uci)
            if threat_move in after_null.legal_moves:
                out["standing_threat_san"] = after_null.san(threat_move)
                refutation = move.get("refutation_san")
                # Same punishment with or without the played move: the threat was
                # on the board already and the move simply ignored it.
                out["threat_was_already_there"] = (
                    refutation is not None
                    and refutation.rstrip("+#") == out["standing_threat_san"].rstrip("+#")
                )
        except ValueError:  # pragma: no cover - malformed cached uci
            pass

    return out


async def enrich(moments: list[dict], depth: int | None = None) -> list[dict]:
    """Attaches etiology to each moment, probing positions as needed.

    Probes go out together: each moment needs two searches, so a dozen moments is
    two dozen engine calls, and the pool can run several at once. Bounded by the
    pool size, since queueing more only waits on the same engines.
    """
    limit = asyncio.Semaphore(max(1, settings.stockfish_workers))

    async def one(moment: dict) -> dict:
        fen = moment.get("fen_before")
        if not fen:
            return moment
        async with limit:
            found = etiology(moment, await probe(fen, depth))
        return {**moment, "why": found} if found else moment

    return list(await asyncio.gather(*(one(m) for m in moments)))
