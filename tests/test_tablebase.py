"""Endgame technique errors verified against perfect play.

The lookups themselves are network calls and are stubbed; what is tested here is
the arithmetic around them, because that is where a wrong answer becomes a
confident false accusation rather than a missing feature.
"""

import chess
import pytest

from app.services import tablebase


# --------------------------------------------------------------------------- #
# Naming the ending
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "fen, expected",
    [
        ("8/8/8/4k3/8/4K3/4P3/8 w - -", "king and pawn versus king"),
        ("8/5p2/8/4k3/8/4K3/4P3/8 w - -", "king and pawn endgame"),
        # One side on a bare king is a technique exercise, and must be read per
        # side: on the combined material this looks like a rook endgame.
        ("8/8/8/4k3/8/4K3/8/4R3 w - -", "mating with a rook"),
        ("8/8/4k3/8/8/4K3/8/4Q3 w - -", "mating with a queen"),
        ("8/8/4k3/8/8/4KB2/8/5B2 w - -", "mating with bishops"),
        ("8/5pk1/8/8/8/6K1/5P2/R6r w - -", "rook endgame"),
        ("8/6k1/8/8/8/6K1/5P2/R6r w - -", "rook and pawn versus rook (Lucena and Philidor)"),
        ("8/5pk1/8/8/8/5BK1/5P2/6b1 w - -", "opposite-coloured bishops"),
        ("8/8/4k3/8/8/4K3/8/8 w - -", "bare kings"),
    ],
)
def test_endings_are_named_by_material_class(fen, expected):
    assert tablebase.name_ending(fen) == expected


# --------------------------------------------------------------------------- #
# Whose point of view
# --------------------------------------------------------------------------- #

def test_a_draw_stays_a_draw_from_either_side():
    """The bug this guards against: outcomes ranked 0..3 and inverted with
    `3 - rank` send a draw to blessed-loss, so every drawn position looked worse
    from one side than the other. It reported "you threw away a draw" on moves
    that were the tablebase's own first choice."""
    for white_to_move in (True, False):
        for mover_is_white in (True, False):
            assert tablebase._standing("draw", mover_is_white, white_to_move) == 0


def test_win_and_loss_swap_when_the_point_of_view_does():
    assert tablebase._standing("win", True, True) == 2
    assert tablebase._standing("win", True, False) == -2
    assert tablebase._standing("loss", False, False) == -2
    assert tablebase._standing("loss", False, True) == 2


def test_the_fifty_move_variants_invert_into_each_other():
    assert tablebase._standing("cursed-win", True, True) == 1
    assert tablebase._standing("cursed-win", True, False) == -1
    assert tablebase._standing("blessed-loss", True, True) == -1


def test_castling_rights_put_a_position_out_of_scope():
    """Syzygy does not cover them and the endpoint answers "unknown"."""
    assert tablebase._standing("unknown", True, True) is None
    assert tablebase._standing(None, True, True) is None


# --------------------------------------------------------------------------- #
# Finding the flips
# --------------------------------------------------------------------------- #

DRAWN = "8/6k1/8/8/8/6K1/5P2/R6r b - -"
LOST = "8/6k1/8/8/8/5PK1/8/R6r w - -"


def _move(before, after, san="Rb1"):
    return {
        "game_id": "g", "ply": 98, "move_number": 49, "played_san": san,
        "fen_before": before, "fen_after": after, "best_san": "Rc3+",
    }


@pytest.fixture
def categories(monkeypatch):
    table: dict[str, str] = {}

    async def fake(fen):
        return table.get(fen)

    monkeypatch.setattr(tablebase, "category", fake)
    return table


async def test_a_move_that_changes_the_result_is_reported(categories):
    categories[DRAWN] = "draw"   # Black to move, drawn
    categories[LOST] = "win"     # White to move and now winning
    flips = await tablebase.review([_move(DRAWN, LOST)], "black")

    assert len(flips) == 1
    assert flips[0]["was"] == "drawn"
    assert flips[0]["became"] == "lost"
    assert flips[0]["engine_preferred"] == "Rc3+"


async def test_a_move_that_holds_the_draw_is_not_reported(categories):
    """Both positions drawn, read from opposite sides. This is the case the
    perspective bug turned into a false accusation."""
    categories[DRAWN] = "draw"
    categories[LOST] = "draw"
    assert await tablebase.review([_move(DRAWN, LOST)], "black") == []


async def test_grinding_an_already_lost_position_is_not_an_error(categories):
    categories[DRAWN] = "loss"
    categories[LOST] = "win"
    assert await tablebase.review([_move(DRAWN, LOST)], "black") == []


async def test_positions_out_of_tablebase_range_are_skipped_without_a_lookup(categories):
    full = chess.STARTING_FEN
    assert await tablebase.review([_move(full, full)], "white") == []


def test_summary_is_none_when_nothing_was_thrown_away():
    assert tablebase.summary([]) is None
    built = tablebase.summary([{"was": "drawn", "became": "lost"}])
    assert built["draws_lost"] == 1
    assert built["result_changing_endgame_moves"] == 1
