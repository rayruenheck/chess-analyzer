"""Positional features -- the evidence behind imbalance-based coaching.

These matter because the prompt now asks the model to teach in terms of
imbalances. A wrong feature here does not produce a wrong number in a table; it
produces a confident sentence about a bad bishop the player does not have.
"""

import chess

from app.services import position


def _board(fen):
    return chess.Board(fen)


# --------------------------------------------------------------------------- #
# Pawn structure
# --------------------------------------------------------------------------- #

def test_isolated_doubled_and_islands_are_counted_off_the_board():
    # White: a2, c2, c3 -- the c-pawns are doubled, and both a and c are isolated.
    s = position.pawn_structure(_board("6k1/8/8/8/8/2P5/P1P5/6K1 w - -"), chess.WHITE)
    assert s["doubled"] == 1
    assert s["isolated"] == 3
    assert s["islands"] == 2


def test_a_pawn_is_passed_only_when_nothing_can_stop_it_on_three_files():
    blocked = _board("6k1/3p4/8/3P4/8/8/8/6K1 w - -")
    assert position.pawn_structure(blocked, chess.WHITE)["passed"] == 0

    # An enemy pawn on an adjacent file ahead still stops it being passed.
    adjacent = _board("6k1/4p3/8/3P4/8/8/8/6K1 w - -")
    assert position.pawn_structure(adjacent, chess.WHITE)["passed"] == 0

    clear = _board("6k1/8/8/3P4/8/8/8/6K1 w - -")
    assert position.pawn_structure(clear, chess.WHITE)["passed"] == 1


# --------------------------------------------------------------------------- #
# Minor pieces
# --------------------------------------------------------------------------- #

def test_the_starting_position_has_no_bad_bishop():
    """Every bishop starts with four of its own pawns on its colour, so a
    threshold at four would flag every game before a move is played."""
    assert position.features(chess.STARTING_FEN, "white", "opening") == {}


def test_a_bishop_hemmed_in_by_its_own_pawns_is_reported():
    fen = "6k1/5pp1/4p2p/3pP3/3P1P2/2P3P1/5B1P/6K1 w - -"
    bad = position.features(fen, "white", "endgame")["your_bad_bishops"]
    assert bad[0]["square"] == "f2"
    assert bad[0]["own_pawns_on_its_colour"] >= position.BAD_BISHOP_PAWNS


def test_an_outpost_needs_a_pawn_defender_and_no_pawn_that_can_evict_it():
    defended = "r4rk1/pp3ppp/8/3N4/2P1P3/8/PP3PPP/R4RK1 w - -"
    assert position.outposts(_board(defended), chess.WHITE) == ["d5"]

    # A black pawn on e6 attacks d5, so the knight sits on a square it can be
    # thrown off -- not an outpost, however advanced it looks.
    evictable = "r4rk1/pp3ppp/4p3/3N4/2P1P3/8/PP3PPP/R4RK1 w - -"
    assert position.outposts(_board(evictable), chess.WHITE) == []

    # Undefended by a pawn is likewise not an outpost.
    unsupported = "r4rk1/pp3ppp/8/3N4/8/2P5/PP3PPP/R4RK1 w - -"
    assert position.outposts(_board(unsupported), chess.WHITE) == []


def test_the_bishop_pair_is_only_an_imbalance_when_one_side_has_it():
    both = position.features(
        "r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq -", "white", "middlegame"
    )
    assert "bishop_pair" not in both

    only_white = position.features(
        "r2qkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq -", "white", "middlegame"
    )
    assert only_white["bishop_pair"] == "you"


# --------------------------------------------------------------------------- #
# Files and king safety
# --------------------------------------------------------------------------- #

def test_files_distinguish_open_from_semi_open_and_spot_the_seventh():
    files = position.file_control(_board("6k1/R4ppp/8/8/8/8/5PPP/6K1 w - -"), chess.WHITE)
    assert "a" in files["open_files"]
    assert files["rooks_on_seventh"] == 1
    assert files["rooks_on_open_files"] == 1


def test_king_safety_reads_the_shield_and_the_file():
    tucked = position.king_safety(_board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - -"), chess.WHITE)
    assert tucked["pawn_shield"] == 3
    assert tucked["on_open_file"] is False

    stripped = position.king_safety(_board("6k1/5ppp/8/8/8/8/7P/6K1 w - -"), chess.WHITE)
    assert stripped["pawn_shield"] == 1
    assert stripped["on_open_file"] is True


def test_an_unremarkable_position_says_nothing_rather_than_reporting_zeroes():
    """An empty dict is what stops the model reading meaning into a page of
    zeroes and inventing an imbalance to explain a move."""
    assert position.features(chess.STARTING_FEN, "black", "opening") == {}
