import json

from app.db import get_db


async def get_response(
    fen: str, source: str, ratings: str = "", speeds: str = "", player: str = "", color: str = ""
) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT response_json FROM explorer_cache "
        "WHERE fen = ? AND source = ? AND ratings = ? AND speeds = ? AND player = ? AND color = ?",
        (fen, source, ratings, speeds, player, color),
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def save_response(
    fen: str,
    source: str,
    response: dict,
    ratings: str = "",
    speeds: str = "",
    player: str = "",
    color: str = "",
) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO explorer_cache "
        "(fen, source, ratings, speeds, player, color, response_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fen, source, ratings, speeds, player, color, json.dumps(response)),
    )
    await db.commit()
