"""Cross-game pattern detection over engine and clock data.

Deterministic, like classify.py, and for the same reason: these are the claims the
coaching report will make out loud, so they are computed and counted here rather
than inferred by a model from raw moves. The model receives the finished numbers.

Each function answers one question a player actually asks -- what do I play, where
does my time go, do I convert winning positions, do I punish mistakes, do I fall
apart late in a session -- and returns None when the data cannot support an answer,
so the report can stay silent instead of guessing.
"""

from collections import defaultdict

# Evaluation, in centipawns from the player's own perspective, at which a position
# is considered winning enough that failing to win it is worth remarking on.
WINNING_CP = 200
# How much of an advantage must evaporate in one move to count as squandered.
COLLAPSE_CP = 150
# Opponent centipawn loss that counts as a gift.
GIFT_CP = 200
# A move faster than this in a non-bullet game is a reflex, not a decision.
SNAP_SECONDS = 5.0
# Fraction of the initial bank below which a player is treated as in time trouble.
TIME_TROUBLE_FRACTION = 0.2
# Gap between games beyond which they belong to different sittings.
SESSION_GAP_SECONDS = 30 * 60
# Time controls where a clock reading is calendar time, not thinking time. A daily
# game reports days per move, which would otherwise dominate every clock average.
UNTIMED_SPEEDS = {"daily", "correspondence"}


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _player_eval(move: dict, colour: str) -> int | None:
    """Evaluation after a move, flipped so positive always favours `colour`.

    Engine scores are stored from White's point of view; every comparison in this
    module is about whether the *player* stood better, so the sign has to follow
    the colour they had rather than the board.
    """
    value = move.get("eval_after_cp")
    if value is None:
        return None
    return value if colour == "white" else -value


# --------------------------------------------------------------------------- #
# Openings
# --------------------------------------------------------------------------- #

# Words that end an opening's family name; everything after is a specific variation.
FAMILY_TERMINATORS = ("defense", "defence", "opening", "game", "attack", "gambit",
                      "system", "counter")


def opening_family(name: str | None) -> str | None:
    """Collapses a specific variation to the opening it belongs to.

    "Caro-Kann Defense: Exchange Variation" and "Caro-Kann Defense: Advance, 4.Nf3"
    are one opening to coach, not two. Grouping by family also gives each bucket
    enough games to say something about -- specific variation names are so granular
    that a twenty-game sample yields twenty buckets of one.

    Lichess separates the variation with a colon; the names parsed out of Chess.com
    ECO URLs have no punctuation, so the family is taken up to the first structural
    word instead.
    """
    if not name:
        return None

    head = name.split(":")[0].strip()
    if head != name:
        return head

    words = head.split()
    for index, word in enumerate(words):
        if word.lower().strip(",") in FAMILY_TERMINATORS:
            return " ".join(words[: index + 1])
    return " ".join(words[:3]) or None


def opening_summary(games: list[dict], min_games: int = 2) -> dict:
    """What the player opens with, what they face, and how each performs.

    Repertoire lines are only reported once they have been seen `min_games` times:
    a single outing says nothing about an opening, and listing one-offs would let
    the report mistake a stray game for a repertoire choice.
    """
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"games": 0, "score": 0.0, "eco": None}
    )

    for game in games:
        name = opening_family(game.get("opening_name"))
        if not name:
            continue
        bucket = buckets[(game["user_color"], name)]
        bucket["games"] += 1
        bucket["eco"] = bucket["eco"] or game.get("eco")
        if game.get("user_score") is not None:
            bucket["score"] += game["user_score"]

    def rows(colour: str) -> list[dict]:
        out = [
            {
                "opening": name,
                "eco": data["eco"],
                "games": data["games"],
                "score_pct": round(100 * data["score"] / data["games"]),
            }
            for (col, name), data in buckets.items()
            if col == colour and data["games"] >= min_games
        ]
        return sorted(out, key=lambda r: -r["games"])

    repertoire = {"as_white": rows("white"), "as_black": rows("black")}
    covered = sum(r["games"] for side in repertoire.values() for r in side)

    return {
        **repertoire,
        "distinct_openings": len({name for _, name in buckets}),
        "games_in_recurring_lines": covered,
        "games_total": len(games),
    }


def book_exit_summary(exits: list[dict]) -> dict | None:
    """Aggregates where the player leaves known theory.

    `exits` come from explorer lookups done by the caller, one per game, each
    `{"game_id", "colour", "opening", "exit_ply", "played", "common"}`.
    """
    scored = [e for e in exits if e.get("exit_ply") is not None]
    if not scored:
        return None

    by_colour: dict[str, list[int]] = defaultdict(list)
    for exit_ in scored:
        by_colour[exit_["colour"]].append(exit_["exit_ply"])

    return {
        "median_exit_move": sorted(e["exit_ply"] for e in scored)[len(scored) // 2] // 2 + 1,
        "average_exit_move_by_colour": {
            colour: round(_mean(plies) / 2 + 0.5, 1) for colour, plies in by_colour.items()
        },
        "earliest_exits": sorted(scored, key=lambda e: e["exit_ply"])[:6],
    }


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #

def timing_summary(moves: list[dict], initial_seconds: int | None = None) -> dict | None:
    """How the player spends time, and whether spending more of it helps.

    The interesting quantity is not seconds but the relationship between seconds
    and accuracy: a player who blunders only when moving fast has a clock problem,
    and one who blunders after long thinks has a calculation problem. Those need
    opposite advice, which is why they are separated here rather than averaged.
    """
    timed = [
        m
        for m in moves
        if m.get("seconds_spent") is not None and m.get("speed") not in UNTIMED_SPEEDS
    ]
    if not timed:
        return None

    scored = [m for m in timed if m.get("cp_lost") is not None]
    snap = [m for m in scored if m["seconds_spent"] < SNAP_SECONDS]
    considered = [m for m in scored if m["seconds_spent"] >= SNAP_SECONDS]

    summary = {
        "moves_with_timing": len(timed),
        "average_seconds_per_move": _mean([m["seconds_spent"] for m in timed]),
        "average_seconds_by_severity": {
            severity: _mean(
                [m["seconds_spent"] for m in scored if m["severity"] == severity]
            )
            for severity in ("blunder", "mistake", "inaccuracy", "ok")
            if any(m["severity"] == severity for m in scored)
        },
        "snap_moves": {
            "threshold_seconds": SNAP_SECONDS,
            "count": len(snap),
            "average_centipawn_loss": _mean([m["cp_lost"] for m in snap]),
            "blunders": sum(1 for m in snap if m["severity"] == "blunder"),
        },
        "considered_moves": {
            "count": len(considered),
            "average_centipawn_loss": _mean([m["cp_lost"] for m in considered]),
            "blunders": sum(1 for m in considered if m["severity"] == "blunder"),
        },
        "slowest_moves": [
            {
                "game_id": m.get("game_id"),
                "move": f"{m['move_number']}{'.' if m['side_to_move'] == 'white' else '...'} {m['played_san']}",
                "seconds": m["seconds_spent"],
                "severity": m["severity"],
                "cp_lost": m.get("cp_lost"),
            }
            for m in sorted(timed, key=lambda m: -m["seconds_spent"])[:5]
        ],
    }

    if initial_seconds:
        threshold = initial_seconds * TIME_TROUBLE_FRACTION
        pressed = [
            m
            for m in scored
            if m.get("clock_after_seconds") is not None
            and m["clock_after_seconds"] <= threshold
        ]
        if pressed:
            summary["time_trouble"] = {
                "below_seconds": round(threshold),
                "moves": len(pressed),
                "average_centipawn_loss": _mean([m["cp_lost"] for m in pressed]),
                "blunders": sum(1 for m in pressed if m["severity"] == "blunder"),
            }

    return summary


# --------------------------------------------------------------------------- #
# Conversion and punishment
# --------------------------------------------------------------------------- #

def conversion_summary(per_game: list[dict]) -> dict | None:
    """Whether winning positions actually become wins.

    `per_game` entries need `game_id`, `colour`, `user_score`, and the annotated
    move list for the whole game (both sides), since the peak evaluation can occur
    on either player's move.
    """
    reached: list[dict] = []

    for game in per_game:
        evals = [
            value
            for move in game["moves"]
            if (value := _player_eval(move, game["colour"])) is not None
        ]
        if not evals:
            continue
        peak = max(evals)
        if peak >= WINNING_CP:
            reached.append(
                {
                    "game_id": game["game_id"],
                    "colour": game["colour"],
                    "peak_cp": peak,
                    "user_score": game["user_score"],
                    "converted": game["user_score"] == 1.0,
                }
            )

    if not reached:
        return None

    won = sum(1 for g in reached if g["converted"])
    return {
        "winning_positions_reached": len(reached),
        "converted_to_wins": won,
        "conversion_rate_pct": round(100 * won / len(reached)),
        "threshold_cp": WINNING_CP,
        "squandered": [g for g in reached if not g["converted"]][:6],
    }


def missed_punishment(moves: list[dict], colour: str) -> list[dict]:
    """Opponent errors the player failed to capitalize on, within one game.

    `moves` is every ply of the game in order, annotated, both sides. A gift is
    only counted as missed when the advantage it created actually evaporates on
    the player's very next move -- otherwise a slow squeeze that wins anyway would
    be scored as a failure.
    """
    missed: list[dict] = []

    for index, move in enumerate(moves[:-1]):
        opponent_moved = move["side_to_move"] != colour
        if not opponent_moved or (move.get("cp_lost") or 0) < GIFT_CP:
            continue

        opportunity = _player_eval(move, colour)
        reply = moves[index + 1]
        after_reply = _player_eval(reply, colour)
        if opportunity is None or after_reply is None or opportunity < WINNING_CP:
            continue

        if opportunity - after_reply >= COLLAPSE_CP:
            missed.append(
                {
                    "game_id": reply.get("game_id"),
                    "opponent_move": move["played_san"],
                    "advantage_offered_cp": opportunity,
                    "your_reply": f"{reply['move_number']} {reply['played_san']}",
                    "advantage_after_cp": after_reply,
                    "engine_wanted": reply.get("best_san"),
                }
            )

    return missed


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def session_summary(games: list[dict]) -> dict | None:
    """Whether play degrades across a sitting or after a loss.

    Games are grouped into sittings by the gap between them, then compared by
    position within the sitting. This is the one section most likely to be noise,
    so it reports nothing unless there are enough multi-game sittings to mean
    something.
    """
    from datetime import datetime

    dated = []
    for game in games:
        if not game.get("played_at") or game.get("average_centipawn_loss") is None:
            continue
        try:
            when = datetime.fromisoformat(game["played_at"])
        except ValueError:
            continue
        dated.append((when, game))

    if len(dated) < 4:
        return None

    dated.sort(key=lambda pair: pair[0])

    sessions: list[list[dict]] = [[]]
    previous = None
    for when, game in dated:
        if previous and (when - previous).total_seconds() > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(game)
        previous = when

    multi = [s for s in sessions if len(s) >= 3]
    if not multi:
        return None

    early = [g["average_centipawn_loss"] for s in multi for g in s[:2]]
    late = [g["average_centipawn_loss"] for s in multi for g in s[2:]]

    after_loss, after_other = [], []
    for session in sessions:
        for previous_game, game in zip(session, session[1:]):
            target = after_loss if previous_game.get("user_score") == 0 else after_other
            if game["average_centipawn_loss"] is not None:
                target.append(game["average_centipawn_loss"])

    result = {
        "sessions": len(sessions),
        "sessions_of_three_or_more": len(multi),
        "longest_session_games": max(len(s) for s in sessions),
        "average_centipawn_loss_first_two_games": _mean(early),
        "average_centipawn_loss_later_games": _mean(late),
    }
    if after_loss and after_other:
        result["average_centipawn_loss_after_a_loss"] = _mean(after_loss)
        result["average_centipawn_loss_otherwise"] = _mean(after_other)
    return result
