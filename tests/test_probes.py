"""Blunder etiology: which habit produced the mistake, not how much it cost.

`etiology` is deliberately pure -- it takes a move and an already-computed probe
result -- so the classification can be tested without launching an engine. The
engine calls that feed it live in stockfish.py and are stubbed out of this suite.
"""

import chess

from app.services import probes
from app.services.classify import static_exchange


# --------------------------------------------------------------------------- #
# Static exchange evaluation
# --------------------------------------------------------------------------- #

def test_see_prices_a_capture_sequence_out_to_the_end():
    free = chess.Board("rnb1kbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -")
    assert static_exchange(free, chess.Move.from_uci("e4d5")) == 100

    # Same capture, but Black's queen recaptures: pawn for pawn.
    even = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -")
    assert static_exchange(even, chess.Move.from_uci("e4d5")) == 0

    # Queen takes a pawn defended by a pawn: the counting error Heisman names.
    losing = chess.Board("rnbqkbnr/pp2pppp/2p5/3p4/8/8/PPP1PPPP/RNBQKBNR w KQkq -")
    assert static_exchange(losing, chess.Move.from_uci("d1d5")) == -800


def test_see_is_zero_for_a_move_that_captures_nothing():
    assert static_exchange(chess.Board(), chess.Move.from_uci("e2e4")) == 0


# --------------------------------------------------------------------------- #
# Etiology
# --------------------------------------------------------------------------- #

LOSING_GRAB = {
    "fen_before": "rnbqkbnr/pp2pppp/2p5/3p4/8/8/PPP1PPPP/RNBQKBNR w KQkq -",
    "played_uci": "d1d5",
    "eval_before_cp": 20,
}


def test_a_capture_that_loses_the_exchange_is_named_as_counting():
    found = probes.etiology(LOSING_GRAB, {})
    assert found["exchange_value_cp"] == -800
    assert found["lost_a_counting_exchange"] is True


def test_an_even_capture_is_not_a_counting_error():
    even = {
        "fen_before": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        "played_uci": "e4d5",
        "eval_before_cp": 20,
    }
    found = probes.etiology(even, {})
    assert found["exchange_value_cp"] == 0
    assert "lost_a_counting_exchange" not in found


def test_a_position_with_one_answer_is_told_apart_from_one_with_many():
    """Scolding a player for missing the only move is unfair unless the report
    knows it was the only move."""
    only = probes.etiology(LOSING_GRAB, {"second_best_score_cp": -400})
    assert only["second_best_costs_cp"] == 420
    assert only["only_one_move_held"] is True

    forgiving = probes.etiology(LOSING_GRAB, {"second_best_score_cp": 10})
    assert forgiving["only_one_move_held"] is False


def test_criticality_is_measured_from_the_movers_side():
    """Scores are stored from White's point of view; the gap must not flip sign
    when it is Black to move."""
    black_to_move = {
        "fen_before": "rnbqkbnr/pp2pppp/2p5/3p4/8/8/PPP1PPPP/RNBQKBNR b KQkq -",
        "played_uci": "d5d4",
        "eval_before_cp": -20,
    }
    found = probes.etiology(black_to_move, {"second_best_score_cp": 400})
    assert found["second_best_costs_cp"] == 420
    assert found["only_one_move_held"] is True


# Ruy Lopez after 3...a6. The b5 bishop is already attacked, so axb5 is on the
# board before White does anything; playing h3 here ignores a standing threat
# rather than creating a new one.
IGNORED_THREAT = {
    "fen_before": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
    "played_uci": "h2h3",
    "eval_before_cp": 20,
}


def test_a_threat_that_predates_the_move_is_distinguished_from_one_it_created():
    """Same punishment with or without the move means the player was looking at
    their own plan while the opponent's was already on the board -- prophylaxis,
    not a hung piece, and a different cure."""
    ignored = probes.etiology(
        {**IGNORED_THREAT, "refutation_san": "axb5"}, {"null_move_threat": "a6b5"}
    )
    assert ignored["standing_threat_san"] == "axb5"
    assert ignored["threat_was_already_there"] is True

    # A different punishment means the move created the problem it was punished for.
    created = probes.etiology(
        {**IGNORED_THREAT, "refutation_san": "Nxe4"}, {"null_move_threat": "a6b5"}
    )
    assert created["standing_threat_san"] == "axb5"
    assert created["threat_was_already_there"] is False


def test_a_threat_probe_is_skipped_when_the_mover_is_in_check():
    """There is no free move to give when you are in check, so the probe returns
    nothing and the classification simply omits it."""
    in_check = {
        "fen_before": "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq -",
        "played_uci": "c7c6",
        "eval_before_cp": 30,
    }
    assert "standing_threat_san" not in probes.etiology(in_check, {})


def test_an_empty_probe_still_yields_what_the_board_alone_can_say():
    found = probes.etiology(LOSING_GRAB, {})
    assert "lost_a_counting_exchange" in found
    assert "only_one_move_held" not in found
    assert "standing_threat_san" not in found


def test_a_move_that_cannot_be_parsed_is_skipped_rather_than_guessed():
    assert probes.etiology({"fen_before": chess.STARTING_FEN, "played_uci": "zzzz"}, {}) == {}
    assert probes.etiology({"played_uci": "e2e4"}, {}) == {}
    # Legal-looking but not legal here.
    assert probes.etiology(
        {"fen_before": chess.STARTING_FEN, "played_uci": "e2e5"}, {}
    ) == {}
