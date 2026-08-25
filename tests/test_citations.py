"""The contract between the model's [G3#47] tokens and the links the UI renders.

The failure this file guards against is a link that points somewhere wrong. A
citation the model invented, a ply that was never analyzed, or a number that
drifted out of step with the sidebar all produce prose that looks authoritative
and sends the player to the wrong position -- worse than no link at all.
"""

import pytest

from app.services.citations import GameIndex

GAMES = [
    {
        "game_id": "aaa",
        "number": 1,
        "user_score": 1.0,
        "user_color": "white",
        "opponent": "rival",
        "opponent_rating": 1480,
        "speed": "rapid",
        "opening_name": "Caro-Kann Defense",
        "played_at": "2026-01-01T12:00:00+00:00",
        "url": "http://x/1",
    },
    {
        "game_id": "bbb",
        "number": 2,
        "user_score": 0.0,
        "user_color": "black",
        "opponent": "other",
        "opponent_rating": None,
        "speed": "bullet",
        "opening_name": None,
        "played_at": None,
        "url": None,
    },
]

PLIES = [
    {"ply": 47, "played_san": "Qxh7", "move_number": 24, "side_to_move": "white"},
    {"ply": 48, "played_san": "Kxh7", "move_number": 24, "side_to_move": "black"},
]


@pytest.fixture
def index():
    built = GameIndex(GAMES)
    built.record_plies("aaa", PLIES)
    return built


def test_the_model_never_sees_a_game_id(index):
    payload = index.swap_ids(
        {
            "per_game": [{"game_id": "aaa", "result": "1-0"}],
            "clock": {"slowest_moves": [{"game_id": "bbb", "seconds": 40}]},
        }
    )
    assert payload["per_game"][0] == {"game_ref": "G1", "result": "1-0"}
    assert payload["clock"]["slowest_moves"][0]["game_ref"] == "G2"
    assert "aaa" not in str(payload)


def test_a_citation_resolves_to_a_game_and_a_named_move(index):
    text, cited = index.resolve({"detail": "you took on h7 [G1#47] there"})

    assert text["detail"] == "you took on h7 [[c0]] there"
    assert cited[0]["game_id"] == "aaa"
    assert cited[0]["ply"] == 47
    assert cited[0]["text"] == "Game 1 · 24.Qxh7"


def test_black_moves_are_numbered_with_an_ellipsis(index):
    _, cited = index.resolve({"a": "[G1#48]"})
    assert cited[0]["text"] == "Game 1 · 24...Kxh7"


def test_the_same_position_cited_twice_is_one_citation(index):
    text, cited = index.resolve({"a": "[G1#47]", "b": ["also [G1#47]"]})
    assert len(cited) == 1
    assert text["b"] == ["also [[c0]]"]


def test_an_unanalyzed_ply_falls_back_to_the_game(index):
    _, cited = index.resolve({"a": "[G1#999]"})
    assert cited[0]["ply"] is None
    assert cited[0]["text"] == "Game 1"


def test_an_invented_game_becomes_plain_text_not_a_link(index):
    text, cited = index.resolve({"a": "as seen in [G9]."})
    assert text["a"] == "as seen in Game 9."
    assert cited == []


def test_a_bare_ply_uses_the_game_under_review(index):
    text, cited = index.resolve({"a": "here [#47]"}, default_game_id="aaa")
    assert text["a"] == "here [[c0]]"
    # No game name: a single game's review is already on that game.
    assert cited[0]["text"] == "24.Qxh7"


def test_a_bare_ply_outside_a_single_game_review_is_dropped(index):
    """Without a game to attach to, `[#47]` names no position and must not link."""
    text, cited = index.resolve({"a": "somewhere [#47] in there."})
    assert text["a"] == "somewhere in there."
    assert cited == []


def test_numbering_follows_the_job_not_the_slice():
    """A report filtered to one speed still cites the sidebar's numbers."""
    index = GameIndex(GAMES)
    only_second = [GAMES[1]]

    assert index.payload_index(only_second)[0]["ref"] == "G2"
    _, cited = index.resolve({"a": "[G2]"})
    assert cited[0]["game_id"] == "bbb"


def test_the_index_describes_a_game_without_naming_its_id(index):
    described = index.payload_index(GAMES)
    assert described[0]["game"] == (
        "won as white · vs rival (1480) · rapid · Caro-Kann Defense · 2026-01-01"
    )
    # A game missing everything optional still describes cleanly.
    assert described[1]["game"] == "lost as black · vs other · bullet"


def test_prose_without_citations_is_left_exactly_as_written(index):
    original = {"a": "  spacing   we did not touch.  "}
    assert index.resolve(original)[0] == original
