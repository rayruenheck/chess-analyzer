"""Naming the tactic that punished a move.

Every position here is asserted legal first. Three of these tests were wrong on
the first pass -- a "fork" that attacked nothing, an "undefended" queen its king
was defending, a "pin" played while in check -- and each looked like a bug in the
detector until the board was checked.
"""

import chess
import pytest

from app.services import insights, motifs


@pytest.mark.parametrize(
    "fen, uci, expected",
    [
        ("r3k3/8/8/3N4/8/8/8/4K3 w - -", "d5c7", "fork"),
        ("3q4/8/8/8/8/8/8/3RK2k w - -", "d1d8", "hangingPiece"),
        ("4k3/8/8/8/8/4n3/8/4R1K1 w - -", "e1e2", "pin"),
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - -", "a1a8", "backRankMate"),
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - -", "a1a8", "mate"),
        ("8/4P3/8/8/8/8/8/4K1k1 w - -", "e7e8q", "promotion"),
    ],
)
def test_motifs_are_named_from_board_geometry(fen, uci, expected):
    assert chess.Move.from_uci(uci) in chess.Board(fen).legal_moves, "test position is wrong"
    assert expected in motifs.tag(fen, uci)


def test_a_quiet_move_gets_no_motif_rather_than_a_guess():
    assert motifs.tag("4k3/8/8/8/8/8/4P3/4K3 w - -", "e2e3") == []


def test_a_defended_capture_is_not_a_hanging_piece():
    """The queen on d8 is defended by its own king, so taking it is a trade."""
    assert "hangingPiece" not in motifs.tag("3qk3/8/8/8/8/8/8/3RK3 w - -", "d1d8")


def test_an_illegal_or_unparseable_move_yields_nothing():
    assert motifs.tag(chess.STARTING_FEN, "e2e5") == []
    assert motifs.tag(chess.STARTING_FEN, "not-a-move") == []


# --------------------------------------------------------------------------- #
# Aggregation and drills
# --------------------------------------------------------------------------- #

def _move(severity, themes, cp=400):
    return {"severity": severity, "cp_lost": cp, "tags": [], "phase": "middlegame",
            "position_state": "competitive", "punished_by": list(themes)}


def test_a_common_motif_is_not_a_weakness_just_because_it_is_common():
    """Pins were a third of one player's blunders and a quarter of every move
    they played. Forks were a fifth as many and three times the signal. Counting
    without the base rate prescribes the wrong drill."""
    # Pins in 30 of 40 blunders (75%) and in 750 of 1000 moves (75%): a ratio of
    # exactly 1, i.e. no signal at all. Forks in 10 blunders but only 10 moves.
    moves = ([_move("blunder", ["pin"])] * 30 + [_move("blunder", ["fork"])] * 10
             + [_move("ok", ["pin"], 5)] * 720 + [_move("ok", [], 5)] * 240)
    themes = insights.pattern_rates(moves)["punished_by"]["themes"]

    assert themes["pin"]["in_blunders"] > themes["fork"]["in_blunders"]
    assert themes["pin"]["enrichment"] == pytest.approx(1.0, abs=0.05)
    assert themes["pin"]["verdict"] == "about average"
    assert themes["fork"]["verdict"] == "over-represented"
    assert themes["fork"]["enrichment"] > themes["pin"]["enrichment"]


def test_only_an_over_represented_tactic_gets_a_drill():
    moves = ([_move("blunder", ["fork"])] * 20 + [_move("blunder", ["pin"])] * 10
             + [_move("ok", ["pin"], 5)] * 300 + [_move("ok", [], 5)] * 700)
    themes = insights.pattern_rates(moves)["punished_by"]["themes"]

    assert themes["fork"]["drill"] == "https://lichess.org/training/fork"
    assert "drill" not in themes["pin"]


def test_blunders_with_no_named_motif_are_counted_not_hidden():
    """Not every mistake is a tactic, and the report has to be able to say so."""
    moves = ([_move("blunder", ["fork"])] * 10 + [_move("blunder", [])] * 20
             + [_move("ok", [], 5)] * 100)
    tactics = insights.pattern_rates(moves)["punished_by"]
    assert tactics["blunders_with_no_named_motif"] == 20

    # With no motif anywhere there is nothing to report at all.
    silent = [_move("blunder", [])] * 20 + [_move("ok", [], 5)] * 100
    assert insights.pattern_rates(silent)["punished_by"] is None


def test_only_themes_lichess_serves_a_puzzle_set_for_get_a_drill():
    assert insights.drill_for("fork") == "https://lichess.org/training/fork"
    assert insights.drill_for("mate") is None
