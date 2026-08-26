import asyncio
import io
import logging
import uuid

import chess
import chess.pgn

from app.config import settings
from app.db import get_db
from app.services import evaluation
from app.services.fen import normalize_fen
from app.services.games import get_game_history

logger = logging.getLogger(__name__)

JOB_FIELDS = [
    "job_id",
    "platform",
    "username",
    "max_games",
    "status",
    "games_total",
    "games_done",
    "error",
    "created_at",
    "updated_at",
]


async def create_job(platform: str, username: str, max_games: int) -> str:
    job_id = str(uuid.uuid4())
    db = get_db()
    await db.execute(
        "INSERT INTO jobs (job_id, platform, username, max_games, status) "
        "VALUES (?, ?, ?, ?, 'queued')",
        (job_id, platform, username, max_games),
    )
    await db.commit()
    return job_id


async def get_job(job_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        f"SELECT {', '.join(JOB_FIELDS)} FROM jobs WHERE job_id = ?",
        (job_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return dict(zip(JOB_FIELDS, row)) if row else None


async def _update_job(job_id: str, **fields) -> None:
    db = get_db()
    set_clause = ", ".join(f"{key} = ?" for key in fields)
    await db.execute(
        f"UPDATE jobs SET {set_clause}, updated_at = datetime('now') WHERE job_id = ?",
        (*fields.values(), job_id),
    )
    await db.commit()


def _walk_plies(pgn_text: str, increment: float = 0.0, initial: float | None = None):
    """Yields one dict per ply, including how long the mover spent on it.

    Walks nodes rather than mainline_moves() because the clock readings live in
    PGN comments (`[%clk 0:09:57.3]`), which mainline_moves() discards.

    Time spent is derived, not given: a clock reading is what the mover had left
    *after* moving, with the increment already credited. So the time they actually
    burned is their previous reading, plus the increment they earned, minus what
    they have now. The first move of each colour has no previous reading and falls
    back to the game's initial time.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return

    board = game.board()
    # Last clock reading seen for each colour, so a move can be diffed against that
    # player's own previous clock rather than their opponent's.
    previous_clock = {"white": initial, "black": initial}

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        fen_before = normalize_fen(board.fen())
        mover = "white" if board.turn == chess.WHITE else "black"
        move_san = board.san(move)
        board.push(move)

        clock_after = node.clock()
        seconds_spent = None
        if clock_after is not None and previous_clock[mover] is not None:
            spent = previous_clock[mover] + increment - clock_after
            # Negative means the clock reading disagrees with our increment model
            # (odd time controls, adjustments, corrupt comments). Drop it rather
            # than feed a nonsense duration into the coaching stats.
            seconds_spent = round(spent, 1) if spent >= 0 else None
        if clock_after is not None:
            previous_clock[mover] = clock_after

        yield {
            "ply": ply,
            "fen_before": fen_before,
            "fen_after": normalize_fen(board.fen()),
            "move_uci": move.uci(),
            "move_san": move_san,
            "mover": mover,
            "clock_after_seconds": clock_after,
            "seconds_spent": seconds_spent,
        }


async def _evaluate_plies(plies: list[dict]) -> None:
    """Evaluates a game's positions across the engine pool.

    Bounded by the pool size: queueing every ply at once would not go any faster,
    since they would all wait on the same engines, and it would make a failure
    mid-game harder to attribute to the position that caused it.

    Positions repeat across a game -- one ply's `fen_after` is the next one's
    `fen_before` -- so concurrent workers can ask for the same evaluation twice.
    That costs a duplicated search at worst; the cache write is idempotent.
    """
    limit = asyncio.Semaphore(max(1, settings.stockfish_workers))

    async def one(ply: dict) -> None:
        async with limit:
            after_eval, _ = await evaluation.get_or_analyze(ply["fen_after"])
            await evaluation.get_or_compute_diff(
                ply["fen_after"],
                ply["fen_before"],
                after_eval["depth"],
                after_eval["score_cp"],
            )

    await asyncio.gather(*(one(ply) for ply in plies))


async def _save_ply(job_id: str, game_id: str, platform: str, ply: dict) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO game_plies "
        "(job_id, game_id, ply, platform, fen_before, fen_after, move_uci, move_san, "
        "mover, clock_after_seconds, seconds_spent) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            game_id,
            ply["ply"],
            platform,
            ply["fen_before"],
            ply["fen_after"],
            ply["move_uci"],
            ply["move_san"],
            ply["mover"],
            ply["clock_after_seconds"],
            ply["seconds_spent"],
        ),
    )
    await db.commit()


def _user_perspective(game, username: str) -> tuple[str, float | None]:
    """Which colour the analyzed player had, and what they scored.

    Without this the per-game rows are anonymous: `mover` records that *a* side
    moved, not whether it was the player being coached, so no feedback above the
    single-move tier can tell their mistakes from their opponent's.
    """
    target = username.casefold()
    if (game.black_username or "").casefold() == target:
        color = "black"
    else:
        # Default to white when the name does not match either side: platform
        # exports occasionally omit a username, and a game is still worth storing.
        color = "white"

    scores = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    user_score = scores.get(game.result)
    if user_score is not None and color == "black":
        user_score = 1.0 - user_score

    return color, user_score


async def _save_game(job_id: str, game, username: str) -> None:
    color, user_score = _user_perspective(game, username)
    is_white = color == "white"

    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO games "
        "(game_id, job_id, platform, username, user_color, opponent, user_rating, "
        "opponent_rating, result, user_score, speed, url, played_at, eco, opening_name, "
        "opening_ply, initial_seconds, increment_seconds) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            game.game_id,
            job_id,
            game.platform,
            username,
            color,
            game.black_username if is_white else game.white_username,
            game.white_rating if is_white else game.black_rating,
            game.black_rating if is_white else game.white_rating,
            game.result,
            user_score,
            game.speed,
            game.url,
            game.played_at,
            game.eco,
            game.opening_name,
            game.opening_ply,
            game.initial_seconds,
            game.increment_seconds,
        ),
    )
    await db.commit()


async def run_analysis_job(job_id: str, platform: str, username: str, max_games: int) -> None:
    try:
        await _update_job(job_id, status="running")
        games = await get_game_history(platform, username, max_games)
        await _update_job(job_id, games_total=len(games))

        db = get_db()
        for game in games:
            await _save_game(job_id, game, username)

            if game.pgn:
                plies = _walk_plies(
                    game.pgn,
                    increment=game.increment_seconds or 0,
                    initial=game.initial_seconds,
                )
                # Every position in a game is independent, so they go to the
                # engine pool together rather than one at a time. A serial loop
                # leaves five of six engines idle and was what made a hundred-game
                # job an hour's work.
                await _evaluate_plies(plies)
                for ply in plies:
                    await _save_ply(job_id, game.game_id, platform, ply)

            # Only now is every ply of this game on disk. The games row is written
            # up front so progress is visible, so without this flag a half-walked
            # game is indistinguishable from a finished one and would be reviewed
            # as though it were complete.
            await db.execute(
                "UPDATE games SET analyzed = 1 WHERE game_id = ?", (game.game_id,)
            )
            await db.execute(
                "UPDATE jobs SET games_done = games_done + 1, updated_at = datetime('now') "
                "WHERE job_id = ?",
                (job_id,),
            )
            await db.commit()

        await _update_job(job_id, status="done")
    except asyncio.CancelledError:
        # BackgroundTasks die with the process, so a --reload restart or a Ctrl+C
        # cancels an in-flight job. CancelledError stringifies to "", which used to
        # land in the jobs table as a blank error and made this indistinguishable
        # from a real crash. Record it explicitly and let the cancellation continue.
        await _update_job(
            job_id,
            status="error",
            error="Analysis was cancelled -- the server stopped or reloaded mid-job.",
        )
        raise
    except Exception as exc:
        # str(exc) alone is empty for several exception types. The class name is
        # what makes a failed job diagnosable after the fact.
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        await _update_job(job_id, status="error", error=detail)
        logger.exception("Analysis job %s failed", job_id)
