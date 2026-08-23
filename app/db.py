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
    move_san TEXT,
    mover TEXT NOT NULL,
    clock_after_seconds REAL,
    seconds_spent REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, ply)
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    username TEXT NOT NULL,
    user_color TEXT NOT NULL,
    opponent TEXT,
    user_rating INTEGER,
    opponent_rating INTEGER,
    result TEXT NOT NULL,
    user_score REAL,
    speed TEXT,
    url TEXT,
    played_at TEXT,
    eco TEXT,
    opening_name TEXT,
    opening_ply INTEGER,
    initial_seconds INTEGER,
    increment_seconds INTEGER,
    analyzed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS games_job_idx ON games (job_id);

CREATE TABLE IF NOT EXISTS llm_feedback (
    cache_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS llm_feedback_subject_idx ON llm_feedback (kind, subject);
"""


# Columns added to tables that predate them. CREATE TABLE IF NOT EXISTS is a no-op
# on an existing table, so new columns have to be ALTERed in for databases created
# before they were introduced.
ADDED_COLUMNS = {
    "game_plies": {
        "move_san": "TEXT",
        "clock_after_seconds": "REAL",
        "seconds_spent": "REAL",
    },
    "games": {
        "analyzed": "INTEGER NOT NULL DEFAULT 0",
        "eco": "TEXT",
        "opening_name": "TEXT",
        "opening_ply": "INTEGER",
        "initial_seconds": "INTEGER",
        "increment_seconds": "INTEGER",
    },
}


async def _migrate(connection: aiosqlite.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        async with connection.execute(f"PRAGMA table_info({table})") as cursor:
            existing = {row[1] for row in await cursor.fetchall()}
        for column, column_type in columns.items():
            if column not in existing:
                await connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )

    # games.analyzed defaults to 0, which would retroactively lock out every game
    # from a job that finished before the column existed. A job marked done walked
    # all of its games to completion, so its games are analyzed by definition.
    # Idempotent, so it is safe to re-run on every startup.
    await connection.execute(
        "UPDATE games SET analyzed = 1 "
        "WHERE analyzed = 0 AND job_id IN (SELECT job_id FROM jobs WHERE status = 'done')"
    )
    await connection.commit()


async def init_db() -> None:
    global _connection
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    _connection = await aiosqlite.connect(settings.db_path)
    await _connection.executescript(SCHEMA)
    await _connection.commit()
    await _migrate(_connection)


async def close_db() -> None:
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None


def get_db() -> aiosqlite.Connection:
    if _connection is None:
        raise RuntimeError("Database not initialized; call init_db() first")
    return _connection
