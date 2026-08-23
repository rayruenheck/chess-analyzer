import json

from app.db import get_db


async def get_evaluation(fen: str, depth: int) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT best_move, principal_variation, score_cp, mate_in "
        "FROM evaluations WHERE fen = ? AND depth = ?",
        (fen, depth),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return None

    best_move, pv_json, score_cp, mate_in = row
    return {
        "fen": fen,
        "depth": depth,
        "best_move": best_move,
        "principal_variation": json.loads(pv_json) if pv_json else [],
        "score_cp": score_cp,
        "mate_in": mate_in,
    }


async def save_evaluation(evaluation: dict) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO evaluations "
        "(fen, depth, best_move, principal_variation, score_cp, mate_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            evaluation["fen"],
            evaluation["depth"],
            evaluation.get("best_move"),
            json.dumps(evaluation.get("principal_variation", [])),
            evaluation.get("score_cp"),
            evaluation.get("mate_in"),
        ),
    )
    await db.commit()


async def get_move_diff(fen: str, previous_fen: str, depth: int) -> int | None:
    db = get_db()
    async with db.execute(
        "SELECT eval_diff_cp FROM move_evaluations "
        "WHERE fen = ? AND previous_fen = ? AND depth = ?",
        (fen, previous_fen, depth),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None


async def save_move_diff(fen: str, previous_fen: str, depth: int, eval_diff_cp: int) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO move_evaluations (fen, previous_fen, depth, eval_diff_cp) "
        "VALUES (?, ?, ?, ?)",
        (fen, previous_fen, depth, eval_diff_cp),
    )
    await db.commit()
