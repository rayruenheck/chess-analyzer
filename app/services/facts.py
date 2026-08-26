"""Flattens the report's aggregates into a numbered list of citable facts.

The model is handed a deep JSON payload and asked not to invent statistics. That
is a lot to ask of a nested structure: to cite a number it has to remember where
in the tree it came from, and the easiest way to be wrong is to half-remember a
figure from a neighbouring branch.

So the claimable numbers are lifted out into a flat menu. Each carries an id, a
plain statement of what it measures, its value, the sample it came from, and
whether that sample is big enough to say anything. The model cites `F12`; the
validator afterwards checks that every number in the prose belongs to some fact.

Sufficiency is decided here rather than asked of the model. "Do not generalise
from small samples" is an instruction a model can rationalise its way around;
`sufficient: false` is a property of the data that the prompt can refer to
mechanically.
"""

# A rate over fewer moves than this is noise dressed as a finding.
MIN_MOVES_FOR_A_RATE = 30
# A count-based claim (games, conversions) needs far fewer to be worth stating,
# but one occurrence is an anecdote in any sample size.
MIN_EVENTS_FOR_A_COUNT = 3


def _fact(facts: list, statement: str, value, *, n=None, unit=None, floor=None) -> None:
    if value is None:
        return
    sufficient = True if n is None else n >= (floor or MIN_MOVES_FOR_A_RATE)
    facts.append(
        {
            "id": f"F{len(facts) + 1}",
            "statement": statement,
            "value": value,
            **({"unit": unit} if unit else {}),
            **({"n": n} if n is not None else {}),
            "sufficient": sufficient,
        }
    )


def build(payload: dict) -> list[dict]:
    """Every number in the payload the model is allowed to quote, numbered."""
    facts: list[dict] = []

    overall = payload.get("overall") or {}
    moves = overall.get("moves_played")
    _fact(facts, "average centipawn loss, all positions",
          overall.get("average_centipawn_loss"), n=moves, unit="cp")
    _fact(facts, "average centipawn loss, competitive positions only",
          overall.get("average_centipawn_loss_competitive"),
          n=overall.get("moves_in_competitive_positions"), unit="cp")
    _fact(facts, "share of moves matching the engine's first choice",
          None if overall.get("engine_best_move_rate") is None
          else round(100 * overall["engine_best_move_rate"], 1), n=moves, unit="pct")
    plural = {"blunder": "blunders", "mistake": "mistakes", "inaccuracy": "inaccuracies"}
    for severity, count in (overall.get("severity_counts") or {}).items():
        if severity in plural:
            _fact(facts, f"total {plural[severity]} across the games reviewed", count,
                  n=moves, unit="count")

    rates = payload.get("rates") or {}
    _fact(facts, "overall blunder rate", rates.get("blunder_rate_pct"),
          n=rates.get("scored_moves"), unit="pct")

    for label, group in (
        ("game phase", rates.get("by_phase")),
        ("clock remaining", rates.get("by_clock_remaining")),
        ("position before the move", rates.get("by_position_state")),
    ):
        for key, entry in (group or {}).items():
            _fact(facts, f"blunder rate by {label}: {key}",
                  entry.get("blunder_rate_pct"), n=entry.get("moves"), unit="pct")

    for tag, entry in (rates.get("move_type_enrichment") or {}).items():
        readable = tag.replace("_", " ")
        _fact(facts,
              f"'{readable}' moves: {entry.get('share_of_blunders_pct')}% of blunders "
              f"vs {entry.get('share_of_all_moves_pct')}% of all moves, verdict "
              f"{entry.get('verdict')}",
              entry.get("enrichment"), n=rates.get("scored_moves"), unit="ratio")

    refutations = rates.get("refutations") or {}
    _fact(facts, "share of blunders punished by a check or capture, i.e. visible one ply ahead",
          refutations.get("punished_by_a_check_or_capture_pct"),
          n=refutations.get("blunders_with_a_known_refutation"), unit="pct")

    conversion = payload.get("conversion") or {}
    _fact(facts, "share of winning positions converted into wins",
          conversion.get("conversion_rate_pct"),
          n=conversion.get("winning_positions_reached"), unit="pct",
          floor=MIN_EVENTS_FOR_A_COUNT)

    missed = payload.get("missed_punishment") or {}
    _fact(facts, "times an opponent's error went unpunished within a move",
          missed.get("count"), n=missed.get("count"), unit="count",
          floor=MIN_EVENTS_FOR_A_COUNT)

    clock = payload.get("clock") or {}
    _fact(facts, "average seconds per move", clock.get("average_seconds_per_move"),
          n=clock.get("moves_with_timing"), unit="seconds")
    for key, label in (("snap_moves", "moves played in under 5 seconds"),
                       ("considered_moves", "moves given at least 5 seconds")):
        bucket = clock.get(key) or {}
        _fact(facts, f"average centipawn loss on {label}",
              bucket.get("average_centipawn_loss"), n=bucket.get("count"), unit="cp")

    openings = payload.get("openings") or {}
    for side in ("as_white", "as_black"):
        for row in openings.get(side) or []:
            _fact(facts,
                  f"score in {row.get('opening')} {side.replace('_', ' ')}",
                  row.get("score_pct"), n=row.get("games"), unit="pct",
                  floor=MIN_EVENTS_FOR_A_COUNT)

    exits = payload.get("book_exits") or {}
    if isinstance(exits, dict) and "unavailable" not in exits:
        _fact(facts, "median move number at which known theory is left",
              exits.get("median_exit_move"), unit="move number")

    sessions = payload.get("sessions") or {}
    _fact(facts, "average centipawn loss in the first two games of a sitting",
          sessions.get("average_centipawn_loss_first_two_games"),
          n=sessions.get("sessions_of_three_or_more"), unit="cp",
          floor=MIN_EVENTS_FOR_A_COUNT)
    _fact(facts, "average centipawn loss in later games of a sitting",
          sessions.get("average_centipawn_loss_later_games"),
          n=sessions.get("sessions_of_three_or_more"), unit="cp",
          floor=MIN_EVENTS_FOR_A_COUNT)

    return facts


def summary(facts: list[dict]) -> dict:
    """What the prompt needs to know about the menu it was given."""
    thin = [f["id"] for f in facts if not f["sufficient"]]
    return {
        "how_to_use": (
            "Every number you write must come from this list and carry its id, for "
            "example [F12]. Do not compute new numbers, combine them, or round them "
            "into different ones. A fact with sufficient=false comes from too small "
            "a sample to support a claim about a habit: you may mention it as a "
            "single observation, never as a pattern."
        ),
        "total": len(facts),
        "insufficient_sample": thin,
    }
