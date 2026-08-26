"""Endpoint tests.

Weighted toward the responses that are easy to get wrong and expensive when they
are: the guards that stop a billable model call running against data that cannot
support an answer, and the distinction between "no result" and "service broken".
"""

import json

from tests.conftest import AFTER_E5, DEPTH, START_FEN


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_the_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Chess Analyzer</title>" in response.text


def test_openapi_lists_the_feedback_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in (
        "/feedback/games/{job_id}",
        "/feedback/game/{game_id}",
        "/feedback/game/{game_id}/moves",
        "/feedback/move",
        "/feedback/openings/{job_id}",
        "/feedback/opening/{job_id}",
        "/feedback/report/{job_id}",
    ):
        assert route in paths


# --------------------------------------------------------------------------- #
# Engine-only endpoints -- these must never need a model or cost anything
# --------------------------------------------------------------------------- #

def test_games_list_returns_colour_and_opening(client, seed):
    job_id = seed()
    games = client.get(f"/feedback/games/{job_id}").json()

    assert len(games) == 2
    assert {g["user_color"] for g in games} == {"white", "black"}
    assert all(g["opening_name"] for g in games)
    assert all(g["initial_seconds"] == 600 for g in games)


def test_games_list_for_unknown_job_is_empty_not_an_error(client):
    response = client.get("/feedback/games/no-such-job")
    assert response.status_code == 200
    assert response.json() == []


def test_game_moves_include_evaluation_and_timing(client, seed):
    job_id = seed()
    body = client.get(f"/feedback/game/{job_id}-g0/moves?depth={DEPTH}").json()

    assert len(body["moves"]) == 3
    first = body["moves"][0]
    assert first["played_san"] == "e4"
    assert first["seconds_spent"] == 2.0
    assert first["clock_after_seconds"] == 598.0
    assert first["is_player_move"] is True

    # The seeded loss on the player's third ply must land as a mistake, and it is
    # the drop in expected score that decides that -- not the centipawn number,
    # which is still reported alongside it.
    third = body["moves"][2]
    assert third["severity"] == "mistake"
    assert third["cp_lost"] == 130
    assert 10 <= third["win_prob_lost"] < 15


def test_game_moves_statistics_cover_only_the_players_moves(client, seed):
    job_id = seed()
    stats = client.get(f"/feedback/game/{job_id}-g0/moves?depth={DEPTH}").json()["statistics"]
    # White played plies 1 and 3 of the seeded three.
    assert stats["moves_played"] == 2
    assert stats["severity_counts"]["mistake"] == 1


def test_game_moves_unknown_game_is_404(client, seed):
    seed()
    assert client.get("/feedback/game/nope/moves").status_code == 404


def test_openings_are_grouped_by_family(client, seed):
    job_id = seed()
    openings = client.get(f"/feedback/openings/{job_id}").json()

    # Both seeded games share one opening but are played from different colours,
    # so they stay separate entries -- advice for White is not advice for Black.
    assert {o["opening"] for o in openings} == {"King's Pawn Game"}
    assert {o["colour"] for o in openings} == {"white", "black"}
    assert all(o["games"] == 1 for o in openings)


# --------------------------------------------------------------------------- #
# Guards: never spend a model call on data that cannot answer the question
# --------------------------------------------------------------------------- #

def test_report_refuses_while_the_job_is_still_running(client, seed, with_api_key):
    job_id = seed(status="running")
    response = client.get(f"/feedback/report/{job_id}")

    assert response.status_code == 409
    assert "running" in response.json()["detail"]


def test_report_refuses_on_a_failed_job(client, seed, with_api_key):
    job_id = seed(status="error")
    assert client.get(f"/feedback/report/{job_id}").status_code == 409


def test_report_on_unknown_job_is_404(client, with_api_key):
    assert client.get("/feedback/report/nope").status_code == 404


def test_report_for_a_speed_with_no_games_is_404(client, seed, with_api_key):
    job_id = seed(speeds=("rapid",))
    response = client.get(f"/feedback/report/{job_id}?speed=bullet")

    assert response.status_code == 404
    assert "bullet" in response.json()["detail"]


def test_game_review_refuses_a_half_analyzed_game(client, seed, with_api_key):
    """A games row is written before its plies are walked, so an unanalyzed game
    looks complete unless the flag is honoured."""
    job_id = seed(analyzed=False)
    response = client.get(f"/feedback/game/{job_id}-g0")

    assert response.status_code == 409
    assert "still being analyzed" in response.json()["detail"]


def test_review_of_unknown_game_is_404(client, seed, with_api_key):
    seed()
    assert client.get("/feedback/game/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Missing API key degrades cleanly rather than crashing
# --------------------------------------------------------------------------- #

def test_model_endpoints_return_503_without_a_key(client, seed):
    job_id = seed()

    for url in (
        f"/feedback/report/{job_id}",
        f"/feedback/game/{job_id}-g0",
        f"/feedback/opening/{job_id}?name=King%27s%20Pawn%20Game",
    ):
        response = client.get(url)
        assert response.status_code == 503, url
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_engine_endpoints_still_work_without_a_key(client, seed):
    job_id = seed()
    assert client.get(f"/feedback/games/{job_id}").status_code == 200
    assert client.get(f"/feedback/game/{job_id}-g0/moves?depth={DEPTH}").status_code == 200
    assert client.get(f"/feedback/openings/{job_id}").status_code == 200


# --------------------------------------------------------------------------- #
# Model-backed endpoints, with the model stubbed
# --------------------------------------------------------------------------- #

def test_game_review_returns_feedback_and_engine_facts(client, seed, with_api_key, stub_llm):
    job_id = seed()
    body = client.get(f"/feedback/game/{job_id}-g0").json()

    assert body["feedback"]["summary"] == "stub summary"
    assert body["statistics"]["moves_played"] == 2
    assert stub_llm[0]["kind"] == "game"
    # Only the player's own moves are ever offered as critical moments.
    assert all(m["side_to_move"] == "white" for m in body["critical_moments"])


def test_report_sends_every_analysis_section_to_the_model(client, seed, with_api_key, stub_llm):
    job_id = seed()
    body = client.get(f"/feedback/report/{job_id}").json()

    assert body["feedback"]["headline"] == "stub headline"
    payload = stub_llm[0]["payload"]
    for section in (
        "overall",
        "rates",
        "by_time_control",
        "openings",
        "clock",
        "conversion",
        "missed_punishment",
        "sessions",
        "per_game",
        "critical_moments",
    ):
        assert section in payload, section

    # The report is the one call worth real reasoning.
    assert stub_llm[0]["effort"] == "high"


def test_report_splits_by_time_control(client, seed, with_api_key, stub_llm):
    job_id = seed(speeds=("rapid", "bullet"))
    client.get(f"/feedback/report/{job_id}")

    breakdown = stub_llm[0]["payload"]["by_time_control"]
    assert set(breakdown) == {"rapid", "bullet"}
    assert all("accuracy" in v for v in breakdown.values())


def test_speed_filter_narrows_the_report(client, seed, with_api_key, stub_llm):
    job_id = seed(speeds=("rapid", "bullet"))
    client.get(f"/feedback/report/{job_id}?speed=rapid")

    player = stub_llm[0]["payload"]["player"]
    assert player["games_reviewed"] == 1
    assert player["time_controls"] == ["rapid"]
    assert player["filtered_to_speed"] == "rapid"


def test_speed_filter_gets_its_own_cache_subject(client, seed, with_api_key, stub_llm):
    """A filtered report must not be served from the unfiltered report's cache."""
    job_id = seed()
    client.get(f"/feedback/report/{job_id}")
    client.get(f"/feedback/report/{job_id}?speed=rapid")

    assert stub_llm[0]["subject"] != stub_llm[1]["subject"]


def test_opening_coach_sends_both_sides_moves(client, seed, with_api_key, stub_llm):
    job_id = seed()
    body = client.get(
        f"/feedback/opening/{job_id}?name=King%27s%20Pawn%20Game&colour=white"
    ).json()

    assert body["feedback"]["the_idea"] == "stub idea"
    moves = stub_llm[0]["payload"]["games"][0]["moves"]
    # The opening is a dialogue; the player's moves make no sense alone.
    assert {m["yours"] for m in moves} == {True, False}
    assert stub_llm[0]["effort"] == "high"


def test_no_endpoint_shows_the_model_a_raw_game_id(client, seed, with_api_key, stub_llm):
    """Game ids are unreadable in prose and unlinkable, so the model gets G-numbers.

    Asserted over the whole serialized payload rather than the known sites: ids
    surface from a dozen aggregates, and one leak is enough for the model to quote
    it back at the player.
    """
    job_id = seed()
    client.get(f"/feedback/report/{job_id}")
    client.get(f"/feedback/game/{job_id}-g0")
    client.get(f"/feedback/opening/{job_id}?name=King%27s%20Pawn%20Game&colour=white")

    assert len(stub_llm) == 3
    for call in stub_llm:
        assert f"{job_id}-g" not in json.dumps(call["payload"]), call["kind"]


def test_the_report_tells_the_model_which_games_are_which(client, seed, with_api_key, stub_llm):
    job_id = seed(speeds=("rapid", "bullet"))
    client.get(f"/feedback/report/{job_id}")

    index = stub_llm[0]["payload"]["games_index"]
    assert [entry["ref"] for entry in index] == ["G1", "G2"]
    # The description is what makes a bare number mean something to the model.
    assert "rival" in index[0]["game"]


def test_game_numbers_survive_the_speed_filter(client, seed, with_api_key, stub_llm):
    """A filtered report keeps the sidebar's numbers rather than renumbering 1..n.

    Games are listed newest first, so the seeded rapid game (played first) is
    Game 2. Narrowing the report to rapid must still call it Game 2 -- renumbering
    it to Game 1 would point every citation at the wrong sidebar row.
    """
    job_id = seed(speeds=("rapid", "bullet"))
    client.get(f"/feedback/report/{job_id}?speed=rapid")

    payload = stub_llm[0]["payload"]
    assert [entry["ref"] for entry in payload["games_index"]] == ["G2"]
    assert payload["per_game"][0]["game_ref"] == "G2"


def test_the_report_resolves_the_citations_the_model_wrote(client, seed, with_api_key, monkeypatch):
    """A [G1#3] in the model's prose comes back as a marker plus a resolved link."""

    async def cited_report(kind, subject, payload, schema, effort=None, refresh=False):
        return {
            "headline": "you drop material after a long think [G1#3]",
            "strengths": [],
            "weaknesses": [],
            "openings": "nothing to say",
            "clock": "nothing to say",
            "by_time_control": [],
            "drills": ["look again at [G1#3]"],
        }

    from app.services import llm

    monkeypatch.setattr(llm, "generate", cited_report)
    job_id = seed()
    body = client.get(f"/feedback/report/{job_id}").json()

    assert body["feedback"]["headline"] == "you drop material after a long think [[c0]]"
    assert body["feedback"]["drills"] == ["look again at [[c0]]"]

    citation = body["citations"][0]
    assert citation["id"] == "c0"
    # Newest game first, so Game 1 is the second one seeded.
    assert citation["game_id"] == f"{job_id}-g1"
    assert citation["ply"] == 3
    assert citation["text"] == "Game 1 · 2.Nf3"


def test_opening_coach_unknown_opening_is_404(client, seed, with_api_key, stub_llm):
    job_id = seed()
    response = client.get(f"/feedback/opening/{job_id}?name=Latvian%20Gambit")
    assert response.status_code == 404


def test_move_endpoint_rejects_an_illegal_move(client, with_api_key, stub_llm):
    response = client.post(
        "/feedback/move", json={"fen": START_FEN, "move": "e2e5"}
    )
    assert response.status_code == 400
    assert "not legal" in response.json()["detail"]


def test_move_endpoint_explains_a_legal_move(client, with_api_key, stub_llm, monkeypatch):
    """Stockfish is stubbed: the endpoint's job is orchestration, not search."""

    async def fake_analyze(fen, depth=None, time_limit=None):
        return {
            "fen": fen,
            "depth": depth or DEPTH,
            "best_move": "e2e4",
            "principal_variation": ["e2e4", "e7e5"],
            "score_cp": 25,
            "mate_in": None,
        }

    from app.services.stockfish import stockfish_engine

    monkeypatch.setattr(stockfish_engine, "analyze_fen", fake_analyze)

    body = client.post(
        "/feedback/move", json={"fen": START_FEN, "move": "e2e4", "rating": 1500}
    ).json()

    assert body["feedback"]["headline"] == "stub headline"
    assert body["facts"]["played_san"] == "e4"
    assert body["facts"]["tags"] == ["engine_best"]
    assert stub_llm[0]["payload"]["player_rating"] == 1500


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

def test_job_status_unknown_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_job_status_reports_progress(client, seed):
    job_id = seed(status="running")
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "running"
    assert body["games_total"] == 2


def test_analyze_rejects_an_unknown_platform(client):
    response = client.post(
        "/jobs/analyze", json={"platform": "chess24", "username": "x", "max_games": 1}
    )
    assert response.status_code == 422


def test_matching_the_engine_is_never_graded_a_mistake(client, seed, monkeypatch):
    """Regression: the eval swing between two positions is not a pure quality
    measure -- search noise and forced recaptures can post a triple-digit "loss"
    on the engine's own first choice. Labelling that an inaccuracy would tell the
    player their best available move was bad."""
    import sqlite3

    from app.config import settings

    job_id = seed()
    connection = sqlite3.connect(settings.db_path)
    # Make the engine agree with the played move while keeping the -120 swing.
    connection.execute(
        "UPDATE evaluations SET best_move = 'g1f3' WHERE fen = ? AND depth = ?",
        (AFTER_E5, DEPTH),
    )
    connection.commit()
    connection.close()

    moves = client.get(f"/feedback/game/{job_id}-g0/moves?depth={DEPTH}").json()["moves"]
    third = moves[2]

    assert "engine_best" in third["tags"]
    assert third["severity"] == "ok"
    assert third["cp_lost"] == 130  # the number is still reported honestly
