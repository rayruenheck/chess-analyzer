"""Facts menu and output validation -- the two halves of keeping prose honest.

The prompt asks the model to cite; these make citing checkable. The validator in
particular exists because chess is the rare domain where an exact checker is
possible: every move and number the coaching may mention was in the payload.
"""

from app.services import facts, validate


# --------------------------------------------------------------------------- #
# The facts menu
# --------------------------------------------------------------------------- #

def _payload(**over):
    base = {
        "overall": {
            "moves_played": 500,
            "average_centipawn_loss": 64.8,
            "average_centipawn_loss_competitive": 56.1,
            "moves_in_competitive_positions": 400,
            "engine_best_move_rate": 0.418,
            "severity_counts": {"blunder": 20, "mistake": 30, "inaccuracy": 40, "ok": 410},
        },
        "rates": {
            "scored_moves": 500,
            "blunder_rate_pct": 6.4,
            "by_phase": {"endgame": {"blunder_rate_pct": 7.3, "moves": 200}},
        },
    }
    base.update(over)
    return base


def test_every_fact_is_numbered_and_carries_its_sample():
    built = facts.build(_payload())
    assert [f["id"] for f in built] == [f"F{i+1}" for i in range(len(built))]
    rate = next(f for f in built if "endgame" in f["statement"])
    assert rate["value"] == 7.3
    assert rate["n"] == 200
    assert rate["sufficient"] is True


def test_a_thin_sample_is_flagged_so_abstention_is_not_a_judgement_call():
    """"Do not generalise from small samples" is an instruction a model can talk
    itself out of; a boolean on the data is not."""
    built = facts.build(_payload(openings={
        "as_white": [{"opening": "Vienna Game", "games": 2, "score_pct": 100}],
    }))
    vienna = next(f for f in built if "Vienna" in f["statement"])
    assert vienna["sufficient"] is False
    assert facts.summary(built)["insufficient_sample"] == [vienna["id"]]


def test_missing_sections_produce_no_facts_rather_than_nulls():
    assert facts.build({}) == []
    assert facts.build({"overall": {}, "rates": None}) == []


def test_an_unreachable_explorer_contributes_no_theory_fact():
    built = facts.build(_payload(book_exits={"unavailable": "could not be reached"}))
    assert not any("theory" in f["statement"] for f in built)


# --------------------------------------------------------------------------- #
# The validator
# --------------------------------------------------------------------------- #

PAYLOAD = {
    "critical_moments": [
        {"played_san": "Qxh7", "best_san": "Rf1", "best_line_san": ["Rf1", "Kg8", "Bd3"]}
    ],
    "rates": {"by_phase": {"endgame": {"blunder_rate_pct": 7.3, "moves": 654}}},
}


def test_prose_built_from_the_payload_passes():
    result = validate.check(
        {"a": "Qxh7 was the error; the engine wanted Rf1, meeting Kg8 with Bd3.",
         "b": "You blunder in 7.3% of endgame moves, across 654 of them."},
        PAYLOAD,
    )
    assert result["ok"], result


def test_an_invented_move_is_caught():
    result = validate.check({"a": "Nxe6 loses the knight and Qd4 would have won."}, PAYLOAD)
    assert result["invented_moves"] == ["Nxe6", "Qd4"]


def test_an_invented_statistic_is_caught():
    result = validate.check({"a": "You blunder in 23.8% of endgame moves."}, PAYLOAD)
    assert result["invented_numbers"] == [23.8]


def test_castling_is_recognised_as_a_move_not_prose():
    ok = validate.check({"a": "O-O was right."}, {"critical_moments": [{"best_san": "O-O"}]})
    assert ok["ok"]
    bad = validate.check({"a": "O-O-O was right."}, {"critical_moments": [{"best_san": "O-O"}]})
    assert bad["invented_moves"] == ["O-O-O"]


def test_rounding_and_ordinary_small_numbers_are_not_flagged():
    """A model saying "about 7%" of a 7.3% fact has invented nothing, and "one of
    three games" must not trip the checker."""
    result = validate.check(
        {"a": "About 7% of your endgame moves, in one of three games."}, PAYLOAD
    )
    assert result["ok"], result
