"""Assembles engine facts into coaching feedback.

Three tiers, all on-demand: a single move, one game's critical moments, and a
report across every game in a finished analysis job. Nothing here runs during the
analysis job itself, so you only pay for feedback on games you actually open.

The selection logic in this module is the real cost control. A 40-move game is 80
plies; sending all of them to the model would be expensive and would bury the
moves that mattered. Only the player's own moves are sent, chosen by how much
expected score they threw away -- the rest is compressed into aggregate counts.

Selection is also where the report's quality is decided, not just its price. The
worst dozen moves are all catastrophes, and a habit that leaks six points a game
across forty games never appears among them; since naming that habit is the whole
job, _report_moments samples deliberately rather than taking the tail.
"""

import json
import logging

import chess

from app.config import settings
from app.db import get_db
from app.schemas import GameFeedback, MoveFeedback, OpeningCoach, PlayerReport
from app.services import classify, evaluation, explorer, facts, insights, llm, position
from app.services import motifs, probes, tablebase
from app.services import validate
from app.services import jobs as jobs_service
from app.services.citations import GameIndex
from app.services.fen import normalize_fen

logger = logging.getLogger(__name__)

# How many of the player's worst moves get sent for explanation. Past roughly this
# many, a review stops being actionable and starts being a wall of text.
# A cap rather than a quota. A fixed count is the wrong instrument for one game:
# eight slots force eight criticisms out of a cleanly played game, where the
# eighth-worst move cost under two points of expected score and was simply fine.
CRITICAL_MOMENTS_PER_GAME = 10
# Below this much expected score given away, a move is not worth reviewing.
MOMENT_FLOOR_WIN_PCT = 5.0
# When nothing clears the floor the game was played well, and saying so is a real
# review -- but a couple of near misses still give the praise something to sit on.
MOMENTS_IN_A_CLEAN_GAME = 3
# How many worst-moments each time-control breakdown carries.
CRITICAL_MOMENTS_PER_SPEED = 4

PLY_FIELDS = [
    "ply",
    "game_id",
    "fen_before",
    "fen_after",
    "move_uci",
    "move_san",
    "mover",
    "clock_after_seconds",
    "seconds_spent",
    "eval_diff_cp",
    "best_move",
    "principal_variation",
    "score_cp_before",
    "mate_in_before",
    "score_cp_after",
    "mate_in_after",
    "refutation_uci",
]

_PLY_QUERY = """
SELECT p.ply, p.game_id, p.fen_before, p.fen_after, p.move_uci, p.move_san, p.mover,
       p.clock_after_seconds, p.seconds_spent,
       me.eval_diff_cp,
       before_eval.best_move, before_eval.principal_variation,
       before_eval.score_cp, before_eval.mate_in,
       after_eval.score_cp, after_eval.mate_in,
       after_eval.best_move
FROM game_plies p
LEFT JOIN move_evaluations me
    ON me.fen = p.fen_after AND me.previous_fen = p.fen_before AND me.depth = ?
LEFT JOIN evaluations before_eval
    ON before_eval.fen = p.fen_before AND before_eval.depth = ?
LEFT JOIN evaluations after_eval
    ON after_eval.fen = p.fen_after AND after_eval.depth = ?
WHERE p.game_id = ?
ORDER BY p.ply
"""

GAME_FIELDS = [
    "game_id",
    "job_id",
    "platform",
    "username",
    "user_color",
    "opponent",
    "user_rating",
    "opponent_rating",
    "result",
    "user_score",
    "speed",
    "url",
    "played_at",
    "eco",
    "opening_name",
    "opening_ply",
    "initial_seconds",
    "increment_seconds",
    "analyzed",
]



class ExplorerUnavailable(RuntimeError):
    """The Lichess Opening Explorer could not be reached.

    Kept distinct from "found no book exit" so the report can say the lookup failed
    instead of implying the player never left theory. Silently treating an outage as
    a clean result would put a false claim in the coaching text.
    """


class AnalysisIncomplete(RuntimeError):
    """Raised when feedback is requested before the underlying analysis has finished.

    This exists because the model call costs money and, worse, produces a confident
    answer about data that is not there yet. A report built from a partially walked
    job is not a cheap answer -- it is a wrong one at full price.
    """


async def get_game(game_id: str) -> dict | None:
    db = get_db()
    async with db.execute(
        f"SELECT {', '.join(GAME_FIELDS)} FROM games WHERE game_id = ?", (game_id,)
    ) as cursor:
        row = await cursor.fetchone()
    return dict(zip(GAME_FIELDS, row)) if row else None


async def list_games(job_id: str) -> list[dict]:
    """Every game in a job, newest first, numbered.

    The `number` is the one the player sees everywhere -- the sidebar row, the
    "Game 3" in the coaching prose, the citation link. It is assigned here, over
    the whole job and before any filtering, so a report narrowed to one time
    control still refers to games by the same numbers as the sidebar.
    """
    db = get_db()
    async with db.execute(
        f"SELECT {', '.join(GAME_FIELDS)} FROM games WHERE job_id = ? ORDER BY played_at DESC",
        (job_id,),
    ) as cursor:
        rows = await cursor.fetchall()

    games = []
    for number, row in enumerate(rows, start=1):
        game = dict(zip(GAME_FIELDS, row))
        game["number"] = number
        game["opening_family"] = insights.opening_family(game.get("opening_name"))
        games.append(game)
    return games


async def _load_plies(game_id: str, depth: int) -> list[dict]:
    db = get_db()
    async with db.execute(_PLY_QUERY, (depth, depth, depth, game_id)) as cursor:
        rows = await cursor.fetchall()
    return [dict(zip(PLY_FIELDS, row)) for row in rows]


def _annotate(ply: dict, game: dict | None = None) -> dict:
    """Turns one joined DB row into the fact bundle the model sees."""
    pv = ply["principal_variation"]
    refutation = _refutation(ply)
    facts = classify.describe_move(
        fen_before=ply["fen_before"],
        move_uci=ply["move_uci"],
        eval_diff_cp=ply["eval_diff_cp"],
        best_move_uci=ply["best_move"],
        score_cp_before=ply["score_cp_before"],
        mate_in_before=ply["mate_in_before"],
        score_cp_after=ply["score_cp_after"],
        mate_in_after=ply["mate_in_after"],
        best_pv_uci=json.loads(pv) if pv else [],
        move_number=(ply["ply"] + 1) // 2,
    )
    # move_san is stored by newer jobs; fall back to the SAN the classifier derives.
    facts["played_san"] = ply["move_san"] or facts["played_san"]
    facts["ply"] = ply["ply"]
    facts["move_number"] = (ply["ply"] + 1) // 2
    facts["game_id"] = ply["game_id"]
    facts["fen_before"] = ply["fen_before"]
    facts["fen_after"] = ply["fen_after"]
    facts["clock_after_seconds"] = ply["clock_after_seconds"]
    facts["seconds_spent"] = ply["seconds_spent"]

    # Seconds are not comparable across time controls -- 20 seconds left is calm in
    # a bullet game and desperate in a 30-minute one. The fraction of the starting
    # bank is, so that is what the model gets to reason about time pressure with.
    bank = (game or {}).get("initial_seconds")
    left = ply["clock_after_seconds"]
    if bank and left is not None and (game or {}).get("speed") not in insights.UNTIMED_SPEEDS:
        facts["clock_fraction_left"] = round(left / bank, 3)
        facts["time_pressure"] = left / bank <= insights.TIME_TROUBLE_FRACTION

    facts.update(refutation)
    return facts


def _refutation(ply: dict) -> dict:
    """The opponent's best reply, and whether it was a forcing one.

    This is the hope-chess signal. "You dropped 300 centipawns" does not say
    whether the punishment was a capture sitting in plain sight or a quiet squeeze
    twelve moves deep, and those are opposite lessons: the first is a missing habit
    of checking replies, the second is genuinely hard chess. A refutation that
    begins with a check or a capture is one the player could have seen by looking
    one move ahead, which is exactly what they failed to do.
    """
    uci = ply.get("refutation_uci")
    if not uci:
        return {}
    board = chess.Board(ply["fen_after"])
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return {}
    if move not in board.legal_moves:
        return {}

    capture, check = board.is_capture(move), board.gives_check(move)
    themes = motifs.tag(ply["fen_after"], uci)
    # The mover is whoever moved into this position, i.e. not the side to move here.
    mover = not board.turn
    before = classify.material_balance(chess.Board(ply["fen_before"]), mover)
    after_board = board.copy(stack=False)
    after_board.push(move)
    after = classify.material_balance(after_board, mover)

    return {
        "refutation_san": board.san(move),
        # One ply of looking would have revealed it.
        "refutation_is_forcing": capture or check,
        # Chess.com's blunder gate: did the move actually cost anything, or did
        # only the engine's number move? Counted across the move and its refutation.
        "lost_material": after < before,
        "material_swing_cp": after - before,
        # Named with Lichess's puzzle vocabulary, so a weakness found here can be
        # turned into a themed drill set the player can actually go and practise.
        "punished_by": themes,
    }


def _summarize(annotated: list[dict]) -> dict:
    """Aggregate stats over the player's moves -- the part that is NOT sent move by move."""
    scored = [m for m in annotated if m["cp_lost"] is not None]
    counts = {"blunder": 0, "mistake": 0, "inaccuracy": 0, "ok": 0, "unknown": 0}
    by_phase: dict[str, list[int]] = {}

    for move in annotated:
        counts[move["severity"]] = counts.get(move["severity"], 0) + 1
        if move["cp_lost"] is not None:
            by_phase.setdefault(move["phase"], []).append(move["cp_lost"])

    # Positions that were already decided inflate the headline average: a move
    # that gives up another 400cp when three pieces down is not the player being
    # inaccurate. The conventional number stays first so it remains comparable to
    # what Lichess and Chess.com report; the second is the honest one to coach on.
    competitive = [
        m
        for m in scored
        if m.get("eval_before_cp") is not None
        and abs(m["eval_before_cp"]) <= classify.DECIDED_CP
    ]

    return {
        "moves_played": len(annotated),
        "average_centipawn_loss": (
            round(sum(m["cp_lost"] for m in scored) / len(scored), 1) if scored else None
        ),
        "average_centipawn_loss_competitive": (
            round(sum(m["cp_lost"] for m in competitive) / len(competitive), 1)
            if competitive
            else None
        ),
        "moves_in_competitive_positions": len(competitive),
        "severity_counts": counts,
        "average_centipawn_loss_by_phase": {
            phase: round(sum(losses) / len(losses), 1) for phase, losses in by_phase.items()
        },
        "engine_best_move_rate": (
            round(sum("engine_best" in m["tags"] for m in annotated) / len(annotated), 3)
            if annotated
            else None
        ),
    }


def _critical(annotated: list[dict], limit: int) -> list[dict]:
    """The costliest moves, ranked by expected score thrown away, in board order.

    Ranked on win probability rather than centipawn loss, which does not survive
    contact with the extremes this function exclusively looks at. Centipawn loss is
    clamped at MAX_REPORTED_CP_LOSS, so every move that allows mate lands on the
    identical value and the top-N becomes an arbitrary slice of a large tie -- in a
    100-game job, 48 moves sat on the ceiling for 25 places, and no move that lost
    less than the cap could ever be selected however instructive it was.

    Removing the clamp would not fix it. Unclamped, those same moves score ~99000
    and fill every slot deterministically instead of arbitrarily, including the ones
    played from positions that were lost twenty moves earlier. The unit is the
    problem: centipawns measure the position, and selection has to measure what the
    mistake cost.
    """
    ranked = _by_cost(annotated)
    if not ranked:
        return []

    # A job analysed before pre-move evaluations were stored has no win probability
    # on any move, so the floor cannot be applied to it -- fall back to taking the
    # worst `limit` by centipawns rather than treating the whole game as clean.
    if not any(m.get("win_prob_lost") is not None for m in ranked):
        return sorted(ranked[:limit], key=lambda m: m["ply"])

    above = [m for m in ranked if (m.get("win_prob_lost") or 0) >= MOMENT_FLOOR_WIN_PCT]
    chosen = above[:limit] if above else ranked[:MOMENTS_IN_A_CLEAN_GAME]
    return sorted(chosen, key=lambda m: m["ply"])


def _by_cost(annotated: list[dict]) -> list[dict]:
    """Moves that cost something, worst first."""
    ranked = sorted(
        (
            m
            for m in annotated
            if m.get("win_prob_lost") is not None and m["win_prob_lost"] > 0
        ),
        key=lambda m: m["win_prob_lost"],
        reverse=True,
    )
    # Fall back to centipawns for jobs analysed before the pre-move evaluation was
    # stored, where win_prob_lost is null for every move.
    if not ranked:
        ranked = sorted(
            (m for m in annotated if m["cp_lost"]),
            key=lambda m: m["cp_lost"],
            reverse=True,
        )
    return ranked


# How the report's examples are apportioned. Selecting purely by cost returns the
# tail -- twelve catastrophes -- and a habit that shows up as a six-point leak in
# forty games can never be selected, however often it recurs. Since finding the
# recurring thing is the report's entire job, the sample is built to contain it.
REPORT_QUOTAS = (
    ("worst overall", 6),
    ("a won position thrown away", 3),
    ("a typical error, not a catastrophe", 4),
)
# The band a habitual leak lives in: costly enough to matter, ordinary enough to
# repeat. Catastrophes are already covered by the quota above.
TYPICAL_BAND = (5.0, 15.0)


def _report_moments(annotated: list[dict]) -> list[dict]:
    """A stratified sample of mistakes, not simply the worst ones.

    Each moment carries why it was chosen, so the model knows what job the example
    is doing rather than treating a routine slip and a thrown-away win alike.
    """
    ranked = _by_cost(annotated)
    chosen: dict[tuple, dict] = {}

    def take(reason: str, candidates: list[dict], quota: int) -> None:
        for move in candidates:
            if quota <= 0:
                return
            key = (move.get("game_id"), move.get("ply"))
            if key in chosen:
                continue
            chosen[key] = {**move, "selected_as": reason}
            quota -= 1

    take("worst overall", ranked, REPORT_QUOTAS[0][1])
    take(
        "a won position thrown away",
        [m for m in ranked if m.get("position_state") == "winning"],
        REPORT_QUOTAS[1][1],
    )
    low, high = TYPICAL_BAND
    take(
        "a typical error, not a catastrophe",
        [m for m in ranked if low <= (m.get("win_prob_lost") or 0) < high],
        REPORT_QUOTAS[2][1],
    )
    # Cover any phase the sample missed entirely, so the report cannot conclude
    # the player never errs in an opening it simply was not shown.
    for phase in ("opening", "middlegame", "endgame"):
        if not any(m.get("phase") == phase for m in chosen.values()):
            take(f"the only {phase} example", [m for m in ranked if m.get("phase") == phase], 1)

    return sorted(chosen.values(), key=lambda m: (str(m.get("game_id")), m.get("ply") or 0))


def _with_imbalances(moments: list[dict]) -> list[dict]:
    """Attaches the positional features to the moments actually being discussed.

    Only these, not every ply: the imbalances are what let the model coach a
    concept rather than a number, but a hundred games of them would be most of the
    payload and the model only ever writes about the selected moments.
    """
    out = []
    for moment in moments:
        enriched = dict(moment)
        fen, colour = moment.get("fen_before"), moment.get("side_to_move")
        if fen and colour:
            imbalances = position.features(fen, colour, moment.get("phase"))
            if imbalances:
                enriched["imbalances"] = imbalances
        out.append(enriched)
    return out


# How deep to look for the point where a player leaves known theory. Past about
# move 8 an "unusual" move is a real choice rather than a memory failure, and each
# ply costs an explorer lookup.
BOOK_SEARCH_PLIES = 16
# Number of games that must have reached a position for it to count as known
# theory. Deliberately an absolute count rather than a share of the parent position:
# share measures popularity, not theory. 1.c4 is played by under 5% of players and
# is the entirely mainline English Opening, so a share threshold flags it as "leaving
# book" on move one -- which is both wrong and actively misleading as coaching. What
# actually marks the end of theory is reaching a position hardly anyone has reached.
MIN_BOOK_GAMES = 1000


def _match_move(options: list[dict], ply: dict) -> dict | None:
    """Finds the played move among the explorer's continuations.

    Matched on UCI first, then SAN, because the two sources disagree about castling:
    python-chess writes the king's destination square (e1g1) while Lichess uses the
    king-takes-rook form (e1h1). On UCI alone every castling move silently scores
    zero games and gets reported as leaving theory.
    """
    for option in options:
        if option.get("uci") == ply["move_uci"]:
            return option
    for option in options:
        if ply["move_san"] and option.get("san") == ply["move_san"]:
            return option
    return None


def _game_count(entry: dict | None) -> int:
    """Games behind an explorer node -- a position total or a single move's line."""
    if not entry:
        return 0
    return entry.get("white", 0) + entry.get("draws", 0) + entry.get("black", 0)


async def find_book_exit(game: dict, plies: list[dict]) -> dict | None:
    """First move where the player departs from what their rating band plays.

    Uses the Lichess Opening Explorer, which is why this is worth more than the ECO
    name alone: "you play the Caro-Kann" is something the player already knows,
    while "you leave known paths on move 6, where 78% play Nc3" is not.

    Note the explorer is Lichess data. For a Chess.com player the rating band is an
    approximation -- the two scales are not directly comparable -- so this locates
    where theory ends, not how a Chess.com peer would rate the choice.
    """
    rating = game.get("user_rating") or 1500
    speed = game.get("speed") or "blitz"
    # Chess.com time classes mostly share names with Lichess speeds; daily has no
    # explorer equivalent and is skipped rather than mapped to something wrong.
    if speed not in {"bullet", "blitz", "rapid", "classical"}:
        return None

    for ply in plies:
        if ply["ply"] > BOOK_SEARCH_PLIES:
            break
        if ply["mover"] != game["user_color"]:
            continue

        try:
            stats = await explorer.get_lichess_stats(ply["fen_before"], rating, [speed])
        except Exception as exc:
            logger.warning("Opening explorer lookup failed: %s: %s", type(exc).__name__, exc)
            raise ExplorerUnavailable(str(exc)) from exc

        total = _game_count(stats)
        if total < MIN_BOOK_GAMES:
            # The position before this move is already off the map, so theory ended
            # earlier than this ply and there is nothing to attribute here.
            return None

        options = stats.get("moves") or []
        played = _match_move(options, ply)
        # A move's game count is the number of games reaching the position it creates.
        played_games = _game_count(played)

        if played_games < MIN_BOOK_GAMES:
            most_common = max(options, key=_game_count, default=None)
            # If even the most popular continuation is below the threshold, theory
            # ended here for everybody -- the player did not depart from a beaten
            # path, there was no longer a path. Reporting that as their deviation
            # would blame them for the database running out.
            if _game_count(most_common) < MIN_BOOK_GAMES:
                return None
            return {
                "game_id": game["game_id"],
                "colour": game["user_color"],
                "opening": game.get("opening_name"),
                "exit_ply": ply["ply"],
                "exit_move": f"{(ply['ply'] + 1) // 2}. {ply['move_san']}",
                "played": ply["move_san"],
                "played_games": played_games,
                "games_at_position": total,
                "common": most_common.get("san") if most_common else None,
                "common_games": _game_count(most_common) if most_common else None,
            }

    return None


async def explain_move(
    fen: str,
    move_uci: str,
    depth: int | None = None,
    rating: int | None = None,
    refresh: bool = False,
) -> dict:
    """Explains a single move, analyzing the position on demand if not yet cached."""
    depth = depth or settings.stockfish_default_depth
    fen_before = normalize_fen(fen)

    board = chess.Board(fen_before)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"{move_uci} is not legal in this position")
    board.push(move)
    fen_after = normalize_fen(board.fen())

    before_eval, _ = await evaluation.get_or_analyze(fen_before, depth)
    after_eval, _ = await evaluation.get_or_analyze(fen_after, depth)
    diff = await evaluation.get_or_compute_diff(
        fen_after, fen_before, after_eval["depth"], after_eval["score_cp"]
    )

    facts = classify.describe_move(
        fen_before=fen_before,
        move_uci=move_uci,
        eval_diff_cp=diff,
        best_move_uci=before_eval["best_move"],
        score_cp_before=before_eval["score_cp"],
        mate_in_before=before_eval["mate_in"],
        score_cp_after=after_eval["score_cp"],
        mate_in_after=after_eval["mate_in"],
        best_pv_uci=before_eval["principal_variation"],
    )

    payload = {
        "task": "Explain this single move to the player who played it.",
        "player_rating": rating,
        "position_fen": fen_before,
        "move": facts,
    }
    feedback = await llm.generate(
        kind="move",
        subject=f"{fen_before}|{move_uci}",
        payload=payload,
        schema=MoveFeedback,
        refresh=refresh,
    )
    return {"facts": facts, "feedback": feedback}


async def game_moves(game_id: str, depth: int | None = None) -> dict:
    """Every ply of a game with its evaluation, for stepping through a board.

    No model call -- this is pure engine data, so the UI can render the board, the
    eval bar and the move list immediately, and only spend tokens when the player
    asks for the written review.
    """
    depth = depth or settings.stockfish_default_depth

    game = await get_game(game_id)
    if game is None:
        raise LookupError(f"No analyzed game {game_id}.")

    plies = await _load_plies(game_id, depth)
    if not plies:
        raise LookupError(f"Game {game_id} has no analyzed positions at depth {depth}.")

    moves = []
    for ply in plies:
        facts = _annotate(ply, game)
        moves.append(
            {
                **facts,
                "fen_after": ply["fen_after"],
                "is_player_move": ply["mover"] == game["user_color"],
            }
        )

    own = [m for m in moves if m["is_player_move"]]
    return {"game": game, "statistics": _summarize(own), "moves": moves}


async def explain_game(game_id: str, depth: int | None = None, refresh: bool = False) -> dict:
    """Reviews one game: aggregate stats plus explanations of the worst moments."""
    depth = depth or settings.stockfish_default_depth

    game = await get_game(game_id)
    if game is None:
        raise LookupError(f"No analyzed game {game_id}. Run a job for this player first.")
    if not game["analyzed"]:
        raise AnalysisIncomplete(
            f"Game {game_id} is still being analyzed. Wait for it to finish."
        )

    plies = await _load_plies(game_id, depth)
    if not plies:
        raise LookupError(f"Game {game_id} has no analyzed positions at depth {depth}.")

    every_move = [_annotate(p, game) for p in plies]
    own = [m for m in every_move if m["side_to_move"] == game["user_color"]]
    stats = _summarize(own)
    moments = await probes.enrich(_with_imbalances(_critical(own, CRITICAL_MOMENTS_PER_GAME)), depth)

    # Numbered against the whole job so "Game 3" means the same game here as it
    # does in the sidebar and in the cross-game report.
    index = GameIndex(await list_games(game["job_id"]))
    index.record_plies(game_id, every_move)

    payload = {
        "task": (
            "Review this game for the player. Summarize how the game went, explain each "
            "critical moment, and end with one takeaway they can act on. This is a "
            "single game, so cite positions with the bare [#ply] form."
        ),
        "player": {
            "colour": game["user_color"],
            "rating": game["user_rating"],
            "opponent_rating": game["opponent_rating"],
            "result": game["result"],
            "time_control": game["speed"],
        },
        "statistics": stats,
        "critical_moments": index.swap_ids(moments),
    }
    feedback = await llm.generate(
        kind="game",
        subject=game_id,
        payload=payload,
        schema=GameFeedback,
        refresh=refresh,
    )
    feedback, cited = index.resolve(feedback, default_game_id=game_id)
    return {
        "game": game,
        "statistics": stats,
        "critical_moments": moments,
        "feedback": feedback,
        "citations": cited,
    }


def _by_time_control(
    games: list[dict],
    all_moves: list[dict],
    conversion_input: list[dict],
    per_game: list[dict],
) -> dict:
    """Splits every aggregate by time control.

    A player's bullet and rapid habits are close to two different players, and an
    average across both describes neither. Splitting here rather than making the
    caller run one report per speed keeps it to a single model call and, more
    usefully, lets the model contrast the speeds against each other -- "solid in
    rapid, falls apart in bullet" is a finding that no single-speed report can see.
    """
    breakdown: dict[str, dict] = {}

    for speed in sorted({g["speed"] for g in games if g["speed"]}):
        speed_games = [g for g in games if g["speed"] == speed]
        ids = {g["game_id"] for g in speed_games}
        moves = [m for m in all_moves if m["game_id"] in ids]
        if not moves:
            continue

        scores = [g["user_score"] for g in speed_games if g["user_score"] is not None]
        banks = {
            g["initial_seconds"]
            for g in speed_games
            if g["initial_seconds"] and speed not in insights.UNTIMED_SPEEDS
        }

        breakdown[speed] = {
            "games": len(speed_games),
            "score_pct": round(100 * sum(scores) / len(scores)) if scores else None,
            "accuracy": _summarize(moves),
            "rates": insights.pattern_rates(moves),
            "clock": insights.timing_summary(
                moves, banks.pop() if len(banks) == 1 else None
            ),
            "conversion": insights.conversion_summary(
                [c for c in conversion_input if c["game_id"] in ids]
            ),
            "openings": insights.opening_summary(speed_games),
            "sessions": insights.session_summary(
                [g for g in per_game if g["game_id"] in ids]
            ),
            "worst_moments": _critical(moves, CRITICAL_MOMENTS_PER_SPEED),
        }

    return breakdown


# How far into a game the "opening" coaching looks. Far enough to reach the
# middlegame the opening was aiming for, since whether the player actually got that
# middlegame is the whole question.
OPENING_COACH_PLIES = 30


async def list_openings(job_id: str, depth: int | None = None) -> list[dict]:
    """Opening families in a job, with how the player scored in each. No model call."""
    games = [g for g in await list_games(job_id) if g["analyzed"]]
    grouped: dict[tuple[str, str], dict] = {}

    for game in games:
        family = insights.opening_family(game.get("opening_name"))
        if not family:
            continue
        key = (family, game["user_color"])
        entry = grouped.setdefault(
            key,
            {
                "opening": family,
                "colour": game["user_color"],
                "eco": game.get("eco"),
                "games": 0,
                "score": 0.0,
                "game_ids": [],
                "game_numbers": [],
            },
        )
        entry["games"] += 1
        entry["game_ids"].append(game["game_id"])
        entry["game_numbers"].append(game["number"])
        if game["user_score"] is not None:
            entry["score"] += game["user_score"]

    out = []
    for entry in grouped.values():
        entry["score_pct"] = round(100 * entry["score"] / entry["games"])
        del entry["score"]
        out.append(entry)
    return sorted(out, key=lambda e: (-e["games"], e["opening"]))


async def coach_opening(
    job_id: str,
    opening: str,
    colour: str | None = None,
    depth: int | None = None,
    refresh: bool = False,
) -> dict:
    """Coaches one opening: what it aims for, and whether the player got there.

    This is the one place the model is allowed to draw on its own chess knowledge
    rather than only narrating engine output. The distinction is deliberate --
    evaluating a position is calculation, which it does badly, but "what middlegame
    does the King's Indian aim for" is encyclopedic and heavily documented. Every
    concrete claim about the player's actual moves still comes from Stockfish; only
    the description of the opening's intent comes from the model, and it is asked to
    say when a line is too obscure for it to be sure.
    """
    depth = depth or settings.stockfish_default_depth

    all_games = await list_games(job_id)
    index = GameIndex(all_games)
    games = [g for g in all_games if g["analyzed"]]
    target = opening.casefold()
    matching = [
        g
        for g in games
        if (insights.opening_family(g.get("opening_name")) or "").casefold() == target
        and (colour is None or g["user_color"] == colour)
    ]
    if not matching:
        raise LookupError(f"Job {job_id} has no analyzed games in {opening}.")

    played: list[dict] = []
    for game in matching:
        plies = await _load_plies(game["game_id"], depth)
        opening_plies = [p for p in plies if p["ply"] <= OPENING_COACH_PLIES]
        if not opening_plies:
            continue

        annotated = [_annotate(p, game) for p in opening_plies]
        index.record_plies(game["game_id"], annotated)
        played.append(
            {
                "game_id": game["game_id"],
                "colour": game["user_color"],
                "full_opening_name": game.get("opening_name"),
                "eco": game.get("eco"),
                "result": game["result"],
                "user_score": game["user_score"],
                "time_control": game["speed"],
                # Both sides, in order: the opening is a dialogue, and the player's
                # moves make no sense without what they were answering.
                "moves": [
                    {
                        "ply": m["ply"],
                        "move_number": m["move_number"],
                        "side": m["side_to_move"],
                        "san": m["played_san"],
                        "yours": m["side_to_move"] == game["user_color"],
                        "severity": m["severity"],
                        "cp_lost": m["cp_lost"],
                        "engine_preferred": m["best_san"],
                        "seconds_spent": m["seconds_spent"],
                    }
                    for m in annotated
                ],
                "eval_after_opening_cp": annotated[-1]["eval_after_cp"],
            }
        )

    if not played:
        raise LookupError(f"No analyzed positions for {opening} in job {job_id}.")

    payload = {
        "task": (
            "Coach this player on one opening. First explain what the opening is "
            "actually trying to achieve -- the structures it aims for, the middlegame "
            "it produces, where the pieces belong and which pawn breaks matter. Then "
            "compare what this player actually did against that idea, using the "
            "supplied moves and engine judgements. Name the specific points where "
            "their play diverged from the opening's intent, and say whether the "
            "divergence was a real error or just a different plan."
        ),
        "opening": opening,
        "player": {
            "rating": matching[0]["user_rating"],
            "games_in_this_opening": len(played),
        },
        "games_index": index.payload_index(matching),
        "games": played,
    }
    payload = index.swap_ids(payload)
    feedback = await llm.generate(
        kind="opening",
        subject=f"{job_id}:{opening}:{colour or 'both'}",
        payload=payload,
        schema=OpeningCoach,
        # Worth real reasoning: this asks the model to hold the opening's plan and
        # the player's actual moves side by side and find where they parted.
        effort="high",
        refresh=refresh,
    )
    feedback, cited = index.resolve(feedback)
    return {
        "opening": opening,
        "games": index.scope(matching),
        "feedback": feedback,
        "citations": cited,
    }


async def build_report(
    job_id: str,
    depth: int | None = None,
    refresh: bool = False,
    speed: str | None = None,
) -> dict:
    """The player-level report: patterns across the games in a job.

    This is the tier an engine cannot replace. A single game's blunder is noise;
    the same blunder in eight games out of twenty is a habit, and the aggregates
    below are what let the model say so.

    `speed` narrows the report to one time control. Worth doing rather than
    averaging: a player's bullet and rapid habits are close to two different
    players, and pooling them hides both.
    """
    depth = depth or settings.stockfish_default_depth

    job = await jobs_service.get_job(job_id)
    if job is None:
        raise LookupError(f"No such job {job_id}.")
    if job["status"] != "done":
        detail = f" ({job['error']})" if job["error"] else ""
        raise AnalysisIncomplete(
            f"Job {job_id} is {job['status']} -- {job['games_done']}/{job['games_total']} "
            f"games analyzed{detail}. The report reads patterns across whole games, so "
            "running it now would bill a full-effort request for an answer built on "
            "partial data."
        )

    all_games = await list_games(job_id)
    # Numbered over every game in the job, before the speed filter, so a filtered
    # report still uses the same G-numbers as the sidebar and the game reviews.
    index = GameIndex(all_games)
    games = [g for g in all_games if g["analyzed"]]
    if speed:
        games = [g for g in games if g["speed"] == speed]
        if not games:
            raise LookupError(f"Job {job_id} has no analyzed {speed} games.")
    if not games:
        raise LookupError(
            f"No analyzed games for job {job_id}. Jobs created before the games table "
            "existed have no colour information; re-run the job."
        )

    all_moves: list[dict] = []
    per_game: list[dict] = []
    conversion_input: list[dict] = []
    missed: list[dict] = []
    book_exits: list[dict] = []
    explorer_ok = True

    for game in games:
        plies = await _load_plies(game["game_id"], depth)
        if not plies:
            continue

        every_move = [_annotate(p, game) for p in plies]
        for move in every_move:
            move["colour"] = game["user_color"]
            move["speed"] = game["speed"]
        index.record_plies(game["game_id"], every_move)

        own = [m for m in every_move if m["side_to_move"] == game["user_color"]]
        if not own:
            continue
        all_moves.extend(own)

        # Conversion and missed punishment both need the opponent's moves too --
        # the peak evaluation and the gift that created it can occur on either side.
        conversion_input.append(
            {
                "game_id": game["game_id"],
                "colour": game["user_color"],
                "user_score": game["user_score"],
                "moves": every_move,
            }
        )
        missed.extend(insights.missed_punishment(every_move, game["user_color"]))

        if explorer_ok:
            try:
                exit_ = await find_book_exit(game, plies)
            except ExplorerUnavailable:
                # One outage is enough evidence; stop hammering a dead service for
                # every remaining game in the job.
                explorer_ok = False
            else:
                if exit_:
                    book_exits.append(exit_)

        stats = _summarize(own)
        per_game.append(
            {
                "game_id": game["game_id"],
                "colour": game["user_color"],
                "result": game["result"],
                "user_score": game["user_score"],
                "time_control": game["speed"],
                "opponent_rating": game["opponent_rating"],
                "opening": game["opening_name"],
                "played_at": game["played_at"],
                "average_centipawn_loss": stats["average_centipawn_loss"],
                "severity_counts": stats["severity_counts"],
            }
        )

    if not all_moves:
        raise LookupError(f"Job {job_id} produced no evaluated moves at depth {depth}.")

    as_white = [m for m in all_moves if m["colour"] == "white"]
    as_black = [m for m in all_moves if m["colour"] == "black"]
    speeds_present = sorted({g["speed"] for g in games if g["speed"]})
    # Time-trouble thresholds only mean anything against a single clock, so they are
    # computed only when every game in scope shares one.
    banks = {
        g["initial_seconds"]
        for g in games
        if g["initial_seconds"] and g["speed"] not in insights.UNTIMED_SPEEDS
    }

    endgame_flips: list[dict] = []
    for game in games:
        own_moves = [m for m in all_moves if m["game_id"] == game["game_id"]]
        endgame_flips.extend(await tablebase.review(own_moves, game["user_color"]))

    report_moments = await probes.enrich(_with_imbalances(_report_moments(all_moves)), depth)

    payload = {
        "task": (
            "Identify this player's recurring patterns across these games. Name real "
            "habits visible in the data, not one-off mistakes. Every claim about how "
            "OFTEN something happens must come from the rates section, never from "
            "counting the critical moments -- those are a dozen selected examples, "
            "not a sample to generalise from. Every theme must cite "
            "specific moves or numbers from the supplied sections as evidence. Cover "
            "their openings, their clock use, and whether they convert winning "
            "positions, but only where the data supports a claim -- say nothing about "
            "a section that is null or thin. Fill by_time_control with exactly one "
            "entry per time control appearing in the by_time_control data, using that "
            "time control's own numbers and contrasting it against the others. Finish "
            "with drills targeting the weaknesses you actually found."
        ),
        "player": {
            "username": games[0]["username"],
            "platform": games[0]["platform"],
            "rating": games[0]["user_rating"],
            "games_reviewed": len(per_game),
            "time_controls": speeds_present,
            "filtered_to_speed": speed,
        },
        "games_index": index.payload_index(games),
        "overall": _summarize(all_moves),
        # The denominators. Without these the model is naming habits from a dozen
        # examples with no idea how often anything happens.
        "rates": insights.pattern_rates(all_moves),
        "as_white": _summarize(as_white) if as_white else None,
        "as_black": _summarize(as_black) if as_black else None,
        "by_time_control": _by_time_control(
            games, all_moves, conversion_input, per_game
        ),
        "openings": insights.opening_summary(games),
        "book_exits": (
            insights.book_exit_summary(book_exits)
            if explorer_ok
            else {"unavailable": "The Lichess Opening Explorer could not be reached, so "
                  "there is no theory comparison for these games. Say nothing about "
                  "where this player leaves book."}
        ),
        "clock": insights.timing_summary(
            all_moves, banks.pop() if len(banks) == 1 else None
        ),
        "conversion": insights.conversion_summary(conversion_input),
        "missed_punishment": {
            "count": len(missed),
            "examples": missed[:8],
        }
        if missed
        else None,
        "sessions": insights.session_summary(per_game),
        # Ground truth rather than an estimate, and the only section that can say
        # a result actually changed rather than an evaluation moving.
        "endgame_technique": tablebase.summary(endgame_flips),
        "per_game": per_game,
        "critical_moments": report_moments,
    }
    payload = index.swap_ids(payload)
    # Flattened last: it reads the finished aggregates and turns them into the
    # numbered menu the model must cite from.
    payload["facts"] = facts.build(payload)
    payload["using_facts"] = facts.summary(payload["facts"])
    feedback = await llm.generate(
        kind="report",
        subject=f"{job_id}:{speed}" if speed else job_id,
        payload=payload,
        schema=PlayerReport,
        # The report is the one call worth spending real reasoning on: finding a
        # pattern across 20 games is the whole point, and it runs once per job.
        effort="high",
        refresh=refresh,
    )
    feedback, cited = index.resolve(feedback)
    # Every move and number in the prose has to exist in what the model was given.
    grounding = validate.check(feedback, payload)
    if not grounding["ok"]:
        logger.warning(
            "Report for %s contains unsupported content: moves=%s numbers=%s",
            job_id, grounding["invented_moves"], grounding["invented_numbers"],
        )
    return {
        "statistics": payload["overall"],
        "facts": payload["facts"],
        "grounding": grounding,
        "by_time_control": payload["by_time_control"],
        "openings": payload["openings"],
        "book_exits": payload["book_exits"],
        "clock": payload["clock"],
        "conversion": payload["conversion"],
        "missed_punishment": payload["missed_punishment"],
        "sessions": payload["sessions"],
        "endgame_technique": payload["endgame_technique"],
        "rates": payload["rates"],
        "per_game": payload["per_game"],
        "games": index.scope(games),
        "feedback": feedback,
        "citations": cited,
    }
