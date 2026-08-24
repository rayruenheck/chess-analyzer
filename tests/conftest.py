"""Shared fixtures.

Two rules govern this suite, and both exist to keep it safe to run on a laptop at
any time:

- **Nothing reaches the network.** No Anthropic call (they cost money), no Lichess
  or Chess.com call (rate limits and flakiness), no Stockfish process. Anything that
  would leave the machine is stubbed, and `no_network` fails the test loudly if
  something slips through rather than letting it quietly bill or hang.
- **Nothing touches the real database.** Every test gets a fresh SQLite file under
  tmp_path, so a test run can never disturb `data/chess_analyzer.db`.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import settings

# Two positions from the start of a real game, normalized the way the app stores them.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
AFTER_E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
AFTER_NF3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -"

DEPTH = 12


@pytest.fixture
def no_network(monkeypatch):
    """Makes any real outbound HTTP call fail loudly instead of silently happening."""

    async def forbidden(*args, **kwargs):
        raise AssertionError(
            "A test attempted a real network request. Stub the client instead."
        )

    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", str(path))
    return path


@pytest.fixture
def client(db_path, monkeypatch):
    """App client with a temp database and no API key configured.

    No key is the default so that any endpoint reaching the model without being
    stubbed returns a recognisable 503 rather than attempting a billable call.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "stockfish_default_depth", DEPTH)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def with_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key-not-real")


@pytest.fixture
def stub_llm(monkeypatch):
    """Replaces the model call with a canned response matching the requested schema.

    Returns the list of recorded calls so a test can assert on what the service
    actually asked for -- the payload is the interesting part, since that is where
    the deterministic analysis ends and the model begins.
    """
    calls = []

    canned = {
        "MoveFeedback": {
            "headline": "stub headline",
            "explanation": "stub explanation",
            "better_plan": "stub plan",
            "concept": "stub concept",
        },
        "GameFeedback": {
            "summary": "stub summary",
            "critical_moments": [],
            "takeaway": "stub takeaway",
        },
        "PlayerReport": {
            "headline": "stub headline",
            "strengths": [],
            "weaknesses": [],
            "openings": "stub openings",
            "clock": "stub clock",
            "by_time_control": [],
            "drills": ["stub drill"],
        },
        "OpeningCoach": {
            "opening": "stub opening",
            "theory_confidence": "stub confidence",
            "the_idea": "stub idea",
            "typical_middlegame": "stub middlegame",
            "pawn_structure": "stub structure",
            "your_version": "stub version",
            "divergences": [],
            "focus": "stub focus",
        },
    }

    async def fake_generate(kind, subject, payload, schema, effort=None, refresh=False):
        calls.append(
            {
                "kind": kind,
                "subject": subject,
                "payload": payload,
                "schema": schema.__name__,
                "effort": effort,
            }
        )
        return dict(canned[schema.__name__])

    from app.services import llm

    monkeypatch.setattr(llm, "generate", fake_generate)
    return calls


@pytest.fixture
def stub_explorer(monkeypatch):
    """Explorer responses without the network. Wide-open start, narrow after 1.b3."""

    async def fake_stats(fen, rating, speeds):
        if fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP"):
            return {
                "white": 900_000,
                "draws": 100_000,
                "black": 800_000,
                "moves": [
                    {"uci": "e2e4", "san": "e4", "white": 800_000, "draws": 90_000, "black": 700_000},
                    {"uci": "b2b3", "san": "b3", "white": 300, "draws": 40, "black": 260},
                ],
            }
        return {"white": 10, "draws": 2, "black": 8, "moves": []}

    from app.services import explorer

    monkeypatch.setattr(explorer, "get_lichess_stats", fake_stats)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def seed(client):
    """Inserts a finished two-game job directly, bypassing fetching and Stockfish.

    Writing rows rather than running the pipeline keeps the endpoint tests focused
    on the endpoints, and lets them assert on awkward states (an unanalyzed game, a
    job that errored) that a successful run would never produce.
    """
    import sqlite3

    def _seed(
        job_id="job-1",
        status="done",
        speeds=("rapid", "bullet"),
        analyzed=True,
    ):
        connection = sqlite3.connect(settings.db_path)
        connection.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, platform, username, max_games, status, games_total, games_done) "
            "VALUES (?, 'chesscom', 'tester', 10, ?, ?, ?)",
            (job_id, status, len(speeds), len(speeds) if status == "done" else 0),
        )

        for index, speed in enumerate(speeds):
            game_id = f"{job_id}-g{index}"
            colour = "white" if index % 2 == 0 else "black"
            connection.execute(
                "INSERT OR REPLACE INTO games (game_id, job_id, platform, username, "
                "user_color, opponent, user_rating, opponent_rating, result, user_score, "
                "speed, url, played_at, eco, opening_name, opening_ply, initial_seconds, "
                "increment_seconds, analyzed) VALUES (?, ?, 'chesscom', 'tester', ?, "
                "'rival', 1500, 1480, '1-0', ?, ?, 'http://x', ?, 'C20', "
                "'King''s Pawn Game: Wayward Queen', 3, 600, 0, ?)",
                (
                    game_id,
                    job_id,
                    colour,
                    1.0 if colour == "white" else 0.0,
                    speed,
                    f"2026-01-0{index + 1}T12:00:00+00:00",
                    1 if analyzed else 0,
                ),
            )

            plies = [
                (1, START_FEN, AFTER_E4, "e2e4", "e4", "white", 598.0, 2.0),
                (2, AFTER_E4, AFTER_E5, "e7e5", "e5", "black", 597.0, 3.0),
                (3, AFTER_E5, AFTER_NF3, "g1f3", "Nf3", "white", 590.0, 8.0),
            ]
            for ply, before, after, uci, san, mover, clock, spent in plies:
                connection.execute(
                    "INSERT OR REPLACE INTO game_plies (job_id, game_id, ply, platform, "
                    "fen_before, fen_after, move_uci, move_san, mover, "
                    "clock_after_seconds, seconds_spent) "
                    "VALUES (?, ?, ?, 'chesscom', ?, ?, ?, ?, ?, ?, ?)",
                    (job_id, game_id, ply, before, after, uci, san, mover, clock, spent),
                )

            for fen, score, best, pv in (
                (START_FEN, 20, "e2e4", '["e2e4","e7e5"]'),
                (AFTER_E4, 25, "e7e5", '["e7e5","g1f3"]'),
                # Deliberately NOT the move played from this position: the seeded
                # -120 diff on 3.Nf3 only registers as a mistake if the engine
                # preferred something else, since agreement with the engine
                # suppresses the severity label.
                (AFTER_E5, 30, "f1c4", '["f1c4","g8f6"]'),
                (AFTER_NF3, 35, "b8c6", '["b8c6"]'),
            ):
                connection.execute(
                    "INSERT OR REPLACE INTO evaluations "
                    "(fen, depth, best_move, principal_variation, score_cp, mate_in) "
                    "VALUES (?, ?, ?, ?, ?, NULL)",
                    (fen, DEPTH, best, pv, score),
                )

            for after, before, diff in (
                (AFTER_E4, START_FEN, 5),
                (AFTER_E5, AFTER_E4, -5),
                (AFTER_NF3, AFTER_E5, -120),
            ):
                connection.execute(
                    "INSERT OR REPLACE INTO move_evaluations "
                    "(fen, previous_fen, depth, eval_diff_cp) VALUES (?, ?, ?, ?)",
                    (after, before, DEPTH, diff),
                )

        connection.commit()
        connection.close()
        return job_id

    return _seed
