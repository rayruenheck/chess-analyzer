from pathlib import Path

import aiosqlite

from app.config import settings

_connection: aiosqlite.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    fen TEXT NOT NULL,
    depth INTEGER NOT NULL,
    best_move TEXT,
    principal_variation TEXT,
    score_cp INTEGER,
    mate_in INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (fen, depth)
);

CREATE TABLE IF NOT EXISTS move_evaluations (
    fen TEXT NOT NULL,
    previous_fen TEXT NOT NULL,
    depth INTEGER NOT NULL,
    eval_diff_cp INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (fen, previous_fen, depth)
);

CREATE TABLE IF NOT EXISTS explorer_cache (
    fen TEXT NOT NULL,
    source TEXT NOT NULL,
    ratings TEXT NOT NULL DEFAULT '',
    speeds TEXT NOT NULL DEFAULT '',
    player TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '',
    response_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (fen, source, ratings, speeds, player, color)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    username TEXT NOT NULL,
    max_games INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    games_total INTEGER NOT NULL DEFAULT 0,
    games_done INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS game_plies (
    job_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    ply INTEGER NOT NULL,
    platform TEXT NOT NULL,
    fen_before TEXT NOT NULL,
    fen_after TEXT NOT NULL,
    move_uci TEXT NOT NULL,
    mover TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, ply)
);
"""


async def init_db() -> None:
    global _connection
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    _connection = await aiosqlite.connect(settings.db_path)
    await _connection.executescript(SCHEMA)
    await _connection.commit()


async def close_db() -> None:
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None


def get_db() -> aiosqlite.Connection:
    if _connection is None:
        raise RuntimeError("Database not initialized; call init_db() first")
    return _connection
