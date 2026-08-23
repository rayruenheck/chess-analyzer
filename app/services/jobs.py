import io
import uuid

import chess
import chess.pgn

from app.db import get_db
from app.services import evaluation
from app.services.fen import normalize_fen
from app.services.games import get_game_history

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


def _walk_plies(pgn_text: str):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return

    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        fen_before = normalize_fen(board.fen())
        mover = "white" if board.turn == chess.WHITE else "black"
        board.push(move)
        fen_after = normalize_fen(board.fen())
        yield ply, fen_before, fen_after, move.uci(), mover


async def _save_ply(
    job_id: str,
    game_id: str,
    platform: str,
    ply: int,
    fen_before: str,
    fen_after: str,
    move_uci: str,
    mover: str,
) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO game_plies "
        "(job_id, game_id, ply, platform, fen_before, fen_after, move_uci, mover) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, game_id, ply, platform, fen_before, fen_after, move_uci, mover),
    )
    await db.commit()


async def run_analysis_job(job_id: str, platform: str, username: str, max_games: int) -> None:
    try:
        await _update_job(job_id, status="running")
        games = await get_game_history(platform, username, max_games)
        await _update_job(job_id, games_total=len(games))

        db = get_db()
        for game in games:
            if game.pgn:
                for ply, fen_before, fen_after, move_uci, mover in _walk_plies(game.pgn):
                    after_eval, _ = await evaluation.get_or_analyze(fen_after)
                    await evaluation.get_or_compute_diff(
                        fen_after, fen_before, after_eval["depth"], after_eval["score_cp"]
                    )
                    await _save_ply(
                        job_id, game.game_id, platform, ply, fen_before, fen_after, move_uci, mover
                    )

            await db.execute(
                "UPDATE jobs SET games_done = games_done + 1, updated_at = datetime('now') "
                "WHERE job_id = ?",
                (job_id,),
            )
            await db.commit()

        await _update_job(job_id, status="done")
    except Exception as exc:
        await _update_job(job_id, status="error", error=str(exc))
