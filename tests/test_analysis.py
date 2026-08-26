"""Unit tests for the deterministic layers.

These are the modules that decide what is *true* about a game before any model sees
it, so a silent regression here becomes a confidently wrong sentence in the coaching
output. Several of these tests exist because the behaviour they pin down was wrong
at some point during development and produced exactly that.
"""

import chess
import pytest

from app.services import classify, insights
from app.services.feedback import _match_move
from app.services.games import _opening_from_eco_url, _parse_chesscom_time_control
from app.services.jobs import _user_perspective, _walk_plies
from app.schemas import NormalizedGame
from app.services.explorer import nearest_rating_buckets
from app.services.fen import normalize_fen


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "diff, severity",
    [
        (50, "ok"),
        (0, "ok"),
        (-49, "ok"),
        (-50, "inaccuracy"),
        (-99, "inaccuracy"),
        (-100, "mistake"),
        (-299, "mistake"),
        (-300, "blunder"),
        (None, "unknown"),
    ],
)
def test_severity_bands(diff, severity):
    assert classify.classify_severity(diff) == severity


def test_mate_scores_are_clamped_out_of_the_average():
    """Stockfish reports mate as +/-100000. Left unclamped, a single mate-in-N
    would swamp every other move in an average centipawn loss."""
    assert classify.cp_lost(-99_000) == classify.MAX_REPORTED_CP_LOSS
    assert classify.cp_lost(-250) == 250
    assert classify.cp_lost(75) == 0
    assert classify.cp_lost(None) is None


def test_engine_agreement_suppresses_the_severity_label():
    """A move cannot be an inaccuracy if it is the engine's own first choice; the
    swing between two searches carries noise that the label must not inherit."""
    facts = classify.describe_move(
        fen_before="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        move_uci="g1f3",
        eval_diff_cp=-140,
        best_move_uci="g1f3",
        score_cp_after=30,
        mate_in_after=None,
    )
    assert "engine_best" in facts["tags"]
    assert facts["severity"] == "ok"
    assert facts["cp_lost"] == 140  # still reported, just not blamed


def test_describe_move_reads_san_and_engine_line():
    facts = classify.describe_move(
        fen_before="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        move_uci="g1f3",
        eval_diff_cp=-140,
        best_move_uci="f1c4",
        score_cp_after=30,
        mate_in_after=None,
        best_pv_uci=["f1c4", "g8f6", "d2d3"],
    )
    assert facts["played_san"] == "Nf3"
    assert facts["best_san"] == "Bc4"
    assert facts["best_line_san"] == ["Bc4", "Nf6", "d3"]
    assert facts["severity"] == "mistake"
    assert facts["side_to_move"] == "white"


def test_forced_mate_is_attributed_to_the_right_side():
    """Scholar's mate: Black delivers it, so the mate belongs to the mover even
    though the raw score is negative from White's point of view."""
    facts = classify.describe_move(
        fen_before="rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq -",
        move_uci="d8h4",
        eval_diff_cp=99_000,
        best_move_uci="d8h4",
        score_cp_after=-100_000,
        mate_in_after=-1,
    )
    assert facts["played_san"] == "Qh4#"
    assert "has_forced_mate" in facts["tags"]
    assert facts["eval_after_cp"] is None  # a mate score is not a centipawn count


def test_hanging_piece_only_flags_the_unambiguous_case():
    # Bishop to a square attacked by a pawn and defended by nothing.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
    facts = classify.describe_move(
        fen_before=board.fen(), move_uci="d1h5", eval_diff_cp=-30,
        best_move_uci="e2e4", score_cp_after=0, mate_in_after=None,
    )
    assert "hangs_piece" not in facts["tags"]  # h5 is not attacked here


def test_phase_is_driven_by_material():
    assert classify.phase(chess.Board()) == "opening"
    assert classify.phase(chess.Board("4k3/8/8/8/8/8/8/4K3 w - -")) == "endgame"
    # A double-rook ending is an endgame, not a middlegame.
    assert classify.phase(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -")) == "endgame"
    # Queen and rook apiece still has a middlegame in it.
    assert classify.phase(chess.Board("r2qk3/8/8/8/8/8/8/R2QK3 w Qq -")) == "middlegame"


def test_a_closed_position_stops_being_the_opening_eventually():
    """Material alone keeps calling a locked King's Indian the opening at move 30,
    because nothing has been traded. The move number caps that."""
    full = chess.Board()
    assert classify.phase(full, move_number=6) == "opening"
    assert classify.phase(full, move_number=30) == "middlegame"
    # Unknown move number (a position analysed out of context) falls back to material.
    assert classify.phase(full, move_number=None) == "opening"


# --------------------------------------------------------------------------- #
# insights
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name, family",
    [
        ("Caro-Kann Defense: Exchange Variation", "Caro-Kann Defense"),
        ("Alekhine Defense: Sämisch Attack", "Alekhine Defense"),
        ("Kings Pawn Opening St George Defense", "Kings Pawn Opening"),
        ("Giuoco Piano Game", "Giuoco Piano Game"),
        ("Nimzowitsch Larsen Attack Modern Variation", "Nimzowitsch Larsen Attack"),
        (None, None),
    ],
)
def test_opening_family_collapses_variations(name, family):
    assert insights.opening_family(name) == family


def test_repertoire_excludes_one_off_openings():
    games = [
        {"user_color": "white", "opening_name": "Caro-Kann Defense: Exchange",
         "eco": "B13", "user_score": 1.0},
        {"user_color": "white", "opening_name": "Caro-Kann Defense: Advance",
         "eco": "B12", "user_score": 0.0},
        {"user_color": "white", "opening_name": "Vienna Game", "eco": "C25",
         "user_score": 1.0},
    ]
    summary = insights.opening_summary(games)

    # The two Caro-Kann variations are one opening; the lone Vienna is not a repertoire.
    assert [r["opening"] for r in summary["as_white"]] == ["Caro-Kann Defense"]
    assert summary["as_white"][0]["games"] == 2
    assert summary["as_white"][0]["score_pct"] == 50


def _timed(seconds, cp, severity, speed="blitz"):
    return {
        "seconds_spent": seconds, "cp_lost": cp, "severity": severity, "speed": speed,
        "move_number": 1, "side_to_move": "white", "played_san": "e4",
        "clock_after_seconds": 100.0,
    }


def test_timing_separates_snap_moves_from_considered_ones():
    summary = insights.timing_summary(
        [_timed(1.0, 300, "blunder"), _timed(2.0, 40, "ok"),
         _timed(30.0, 10, "ok"), _timed(40.0, 20, "ok")]
    )
    assert summary["snap_moves"]["count"] == 2
    assert summary["snap_moves"]["blunders"] == 1
    assert summary["considered_moves"]["count"] == 2
    assert summary["considered_moves"]["blunders"] == 0


def test_correspondence_games_are_excluded_from_clock_stats():
    """A daily game reports days per move; averaging it in destroys every number."""
    moves = [_timed(2.0, 10, "ok"), _timed(259_191.0, 0, "ok", speed="daily")]
    summary = insights.timing_summary(moves)

    assert summary["moves_with_timing"] == 1
    assert summary["average_seconds_per_move"] == 2.0


def test_timing_returns_none_when_no_clock_data():
    assert insights.timing_summary([{"seconds_spent": None, "cp_lost": 10}]) is None


def _move(cp_after, side="white", cp_lost=0, san="e4"):
    return {"eval_after_cp": cp_after, "side_to_move": side, "cp_lost": cp_lost,
            "played_san": san, "move_number": 1, "best_san": "Qd5", "game_id": "g"}


def test_conversion_counts_winning_positions_that_were_not_won():
    summary = insights.conversion_summary([
        {"game_id": "a", "colour": "white", "user_score": 1.0,
         "moves": [_move(500)]},
        {"game_id": "b", "colour": "white", "user_score": 0.0,
         "moves": [_move(400)]},
        {"game_id": "c", "colour": "white", "user_score": 1.0,
         "moves": [_move(50)]},   # never winning: not counted either way
    ])
    assert summary["winning_positions_reached"] == 2
    assert summary["converted_to_wins"] == 1
    assert summary["conversion_rate_pct"] == 50


def test_conversion_flips_evaluation_for_black():
    """Scores are stored from White's point of view; a -500 eval is Black winning."""
    summary = insights.conversion_summary([
        {"game_id": "a", "colour": "black", "user_score": 0.0, "moves": [_move(-500)]},
    ])
    assert summary["winning_positions_reached"] == 1
    assert summary["converted_to_wins"] == 0


def test_missed_punishment_needs_the_advantage_to_actually_evaporate():
    gift = _move(400, side="black", cp_lost=350, san="Qb6??")
    squandered = _move(20, side="white", cp_lost=380, san="Kh1")
    held = _move(390, side="white", cp_lost=10, san="Qxb6")

    assert len(insights.missed_punishment([gift, squandered], "white")) == 1
    assert insights.missed_punishment([gift, held], "white") == []


def test_session_summary_stays_silent_on_thin_data():
    assert insights.session_summary([]) is None
    assert insights.session_summary(
        [{"played_at": "2026-01-01T10:00:00+00:00", "average_centipawn_loss": 30}]
    ) is None


# --------------------------------------------------------------------------- #
# Platform parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("600", (600, 0)),
        ("600+5", (600, 5)),
        ("180+2", (180, 2)),
        ("1/86400", (86400, None)),   # daily: a per-move allowance, not a bank
        (None, (None, None)),
        ("nonsense", (None, None)),
    ],
)
def test_chesscom_time_control_parsing(raw, expected):
    assert _parse_chesscom_time_control(raw) == expected


@pytest.mark.parametrize(
    "url, name",
    [
        ("https://www.chess.com/openings/Caro-Kann-Defense-Exchange-Variation",
         "Caro Kann Defense Exchange Variation"),
        ("https://www.chess.com/openings/Kings-Pawn-Opening-St-George-Defense-2.d4-b5",
         "Kings Pawn Opening St George Defense"),
        # Move notation welded onto the last name token.
        ("https://www.chess.com/openings/Nimzowitsch-Larsen-Attack-Modern-Variation...3.e3-d5",
         "Nimzowitsch Larsen Attack Modern Variation"),
        (None, None),
    ],
)
def test_opening_name_from_eco_url(url, name):
    assert _opening_from_eco_url(url) == name


def test_user_perspective_finds_the_players_colour_and_score():
    game = NormalizedGame(
        platform="lichess", game_id="x", url="u", result="0-1",
        white_username="Rival", black_username="TeStEr",
    )
    # Platform usernames are case-insensitive.
    assert _user_perspective(game, "tester") == ("black", 1.0)
    assert _user_perspective(game, "Rival") == ("white", 0.0)


def test_normalize_fen_strips_the_move_counters():
    full = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert normalize_fen(full) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def test_rating_buckets_bracket_the_players_rating():
    assert nearest_rating_buckets(1500) == [1400, 1600]
    assert nearest_rating_buckets(1000) == [1000, 1200]
    assert nearest_rating_buckets(3000) == [2500]


# --------------------------------------------------------------------------- #
# Clock extraction
# --------------------------------------------------------------------------- #

PGN_WITH_CLOCKS = """[Event "Test"]
[TimeControl "600"]

1. e4 {[%clk 0:09:58]} e5 {[%clk 0:09:57]} 2. Nf3 {[%clk 0:09:50]} Nc6 {[%clk 0:09:40]} 1-0
"""


def test_walk_plies_derives_time_spent_from_clock_readings():
    plies = list(_walk_plies(PGN_WITH_CLOCKS, increment=0, initial=600))

    assert [p["move_san"] for p in plies] == ["e4", "e5", "Nf3", "Nc6"]
    # Each player is diffed against their own previous reading, not the opponent's.
    assert plies[0]["seconds_spent"] == 2.0    # 600 -> 598
    assert plies[1]["seconds_spent"] == 3.0    # 600 -> 597
    assert plies[2]["seconds_spent"] == 8.0    # 598 -> 590
    assert plies[3]["seconds_spent"] == 17.0   # 597 -> 580


def test_walk_plies_credits_the_increment():
    """A clock reading already includes the increment, so time actually burned is
    previous + increment - current."""
    plies = list(_walk_plies(PGN_WITH_CLOCKS, increment=5, initial=600))
    assert plies[0]["seconds_spent"] == 7.0


def test_walk_plies_survives_a_pgn_with_no_clocks():
    pgn = '[Event "x"]\n\n1. e4 e5 1-0\n'
    plies = list(_walk_plies(pgn, increment=0, initial=600))
    assert len(plies) == 2
    assert all(p["seconds_spent"] is None for p in plies)
    assert all(p["move_san"] for p in plies)


# --------------------------------------------------------------------------- #
# Explorer move matching
# --------------------------------------------------------------------------- #

def test_castling_matches_despite_differing_uci_conventions():
    """Lichess writes castling king-takes-rook (e1h1); python-chess writes the
    king's destination (e1g1). Matching on UCI alone scored every castling move as
    zero games and reported it as leaving theory."""
    options = [
        {"uci": "e1h1", "san": "O-O", "white": 100, "draws": 10, "black": 90},
        {"uci": "d2d4", "san": "d4", "white": 5, "draws": 1, "black": 4},
    ]
    assert _match_move(options, {"move_uci": "e1g1", "move_san": "O-O"})["san"] == "O-O"
    assert _match_move(options, {"move_uci": "d2d4", "move_san": "d4"})["san"] == "d4"
    assert _match_move(options, {"move_uci": "h2h4", "move_san": "h4"}) is None


# --------------------------------------------------------------------------- #
# Win probability and how critical moments get chosen
# --------------------------------------------------------------------------- #

def test_win_probability_folds_in_mate_and_survives_mate_scores():
    assert classify.win_probability(0) == pytest.approx(0.5)
    assert classify.win_probability(None) is None
    # One scale unit either side of level is the logistic's own half-way point,
    # so this pins the curve's shape without hardcoding the constant it is tuned to.
    scale = classify.WIN_PROB_SCALE
    assert classify.win_probability(scale) == pytest.approx(0.731, abs=0.001)
    assert classify.win_probability(-scale) == pytest.approx(0.269, abs=0.001)
    # Symmetric about equality, and monotonic.
    assert classify.win_probability(100) + classify.win_probability(-100) == pytest.approx(1.0)
    assert classify.win_probability(50) < classify.win_probability(150)
    # Mate wins outright regardless of the centipawn field.
    assert classify.win_probability(None, mate_in=3) == 1.0
    assert classify.win_probability(None, mate_in=-3) == 0.0
    # Mate is reported as +/-100000 cp, which overflows exp() if not clamped.
    assert classify.win_probability(100_000) == pytest.approx(1.0)
    assert classify.win_probability(-100_000) == pytest.approx(0.0)


def test_position_state_is_read_from_the_movers_side():
    assert classify.position_state(900, None, mover_is_white=True) == "winning"
    assert classify.position_state(900, None, mover_is_white=False) == "losing"
    assert classify.position_state(0, None, mover_is_white=True) == "competitive"
    assert classify.position_state(None, 2, mover_is_white=True) == "winning"
    assert classify.position_state(None, None, mover_is_white=True) is None


def test_a_losing_move_costs_less_expected_score_the_more_lost_you_already_were():
    """The whole reason selection moved off centipawns."""
    def cost(before, after):
        return classify.describe_move(
            fen_before=chess.STARTING_FEN, move_uci="e2e4", eval_diff_cp=after - before,
            best_move_uci="d2d4", score_cp_before=before, mate_in_before=None,
            score_cp_after=after, mate_in_after=None,
        )["win_prob_lost"]

    from_even = cost(0, -400)
    from_lost = cost(-2000, -2400)
    assert from_even > 25          # gave away most of a playable game
    assert from_lost < 2           # gave away almost nothing that was left
    # Identical centipawn swings, wildly different real cost.
    assert from_even > from_lost * 10


def _moment(ply, cp, wp, state="competitive"):
    return {"ply": ply, "cp_lost": cp, "win_prob_lost": wp, "position_state": state,
            "severity": "blunder", "tags": [], "phase": "middlegame"}


def test_critical_moments_are_not_an_arbitrary_slice_of_a_clamped_tie():
    """The bug this replaced: every mate-allowing move lands on MAX_REPORTED_CP_LOSS,
    so ranking by centipawns picked N of a large tie and could never surface a
    genuinely instructive move that lost slightly less."""
    from app.services.feedback import _critical

    tied = [_moment(p, classify.MAX_REPORTED_CP_LOSS, 1.0, "losing") for p in range(1, 21)]
    real = _moment(99, 600, 55.0)
    picked = _critical(tied + [real], 5)

    assert real in picked, "a real blunder must outrank throwing away a lost position"
    assert picked == sorted(picked, key=lambda m: m["ply"]), "returned in board order"


def test_critical_moments_fall_back_to_centipawns_for_older_jobs():
    """Jobs analysed before the pre-move evaluation was stored have no win
    probability on any move, and must still produce a review rather than nothing."""
    from app.services.feedback import _critical

    old = [{"ply": p, "cp_lost": p * 100, "win_prob_lost": None} for p in (1, 2, 3)]
    assert [m["ply"] for m in _critical(old, 2)] == [2, 3]


# --------------------------------------------------------------------------- #
# Pattern rates -- the denominators
# --------------------------------------------------------------------------- #

def _rated(severity, cp, tags=(), phase="middlegame", state="competitive", clock=None):
    m = {"severity": severity, "cp_lost": cp, "tags": list(tags), "phase": phase,
         "position_state": state}
    if clock is not None:
        m["clock_fraction_left"] = clock
    return m


def test_pattern_rates_reports_a_common_move_type_as_not_a_weakness():
    """Captures were 21% of one player's blunders and 24% of all their moves, which
    reads damning without the denominator. Enrichment at or below 1.0 is what stops
    the report inventing a weakness out of a base rate."""
    moves = ([_rated("blunder", 400, ["capture"])] * 2
             + [_rated("blunder", 400)] * 8
             + [_rated("ok", 10, ["capture"])] * 30
             + [_rated("ok", 10)] * 60)
    rates = insights.pattern_rates(moves)

    capture = rates["move_type_enrichment"]["capture"]
    assert capture["share_of_blunders_pct"] == 20.0
    assert capture["share_of_all_moves_pct"] == 32.0
    assert capture["enrichment"] < 1.0


def test_pattern_rates_separates_blundering_a_won_game_from_blundering_a_close_one():
    moves = ([_rated("blunder", 400, state="winning")] * 5
             + [_rated("ok", 10, state="winning")] * 5
             + [_rated("blunder", 400, state="competitive")] * 5
             + [_rated("ok", 10, state="competitive")] * 95)
    rates = insights.pattern_rates(moves)

    assert rates["by_position_state"]["winning"]["blunder_rate_pct"] == 50.0
    assert rates["by_position_state"]["competitive"]["blunder_rate_pct"] == 5.0


def test_pattern_rates_stays_silent_on_a_sample_too_small_to_mean_anything():
    assert insights.pattern_rates([_rated("blunder", 400)] * 10) is None


def test_a_marginal_skew_is_called_average_rather_than_a_weakness():
    """A hard 1.0 cut turns rounding into a finding. One real player's
    "was in check" sat at 1.18x -- a 6.2% blunder rate against 5.2% -- which is
    noise, and naming it a weakness is how a report fills up with invented patterns."""
    # 11% of blunders, 10% of all moves: a 1.1x skew, i.e. nothing.
    moves = ([_rated("blunder", 400, ["was_in_check"])] * 11
             + [_rated("blunder", 400)] * 89
             + [_rated("ok", 10, ["was_in_check"])] * 100
             + [_rated("ok", 10)] * 900)
    entry = insights.pattern_rates(moves)["move_type_enrichment"]["was_in_check"]

    assert entry["enrichment"] == pytest.approx(1.1, abs=0.02)
    assert entry["verdict"] == "about average"


def test_a_real_skew_and_a_real_strength_both_get_named():
    moves = ([_rated("blunder", 400, ["hangs_piece"])] * 40
             + [_rated("blunder", 400, ["capture"])] * 2
             + [_rated("blunder", 400)] * 58
             + [_rated("ok", 10, ["hangs_piece"])] * 60
             + [_rated("ok", 10, ["capture"])] * 300
             + [_rated("ok", 10)] * 540)
    tags = insights.pattern_rates(moves)["move_type_enrichment"]

    assert tags["hangs_piece"]["verdict"] == "over-represented"
    assert tags["capture"]["verdict"] == "not a weakness"


def test_severity_thresholds_are_on_the_win_percent_scale_not_winning_chances():
    """Lichess's Advice.scala reads .3 / .2 / .1, but those are deltas of
    `winningChances`, which scalachess defines on [-1, +1]; WinPercent is
    `50 + 50 * winningChances` on [0, 100]. Copying .3 across as 30 points grades
    twice as leniently as Lichess and roughly halves the blunder count."""
    assert classify.BLUNDER_WIN_PCT == 15.0
    assert classify.MISTAKE_WIN_PCT == 10.0
    assert classify.INACCURACY_WIN_PCT == 5.0

    # Equivalent statement: a .3 drop in winning chances IS a blunder.
    before, after = classify.win_probability(200), classify.win_probability(-200)
    winning_chances_drop = (2 * before - 1) - (2 * after - 1)
    assert winning_chances_drop >= 0.3
    assert classify.classify_by_win_probability(100 * (before - after)) == "blunder"


def test_a_swing_in_a_decided_position_is_not_a_blunder_on_this_scale():
    """The false positive centipawn bands produced: a large eval move where the
    game's outcome barely changes."""
    before, after = classify.win_probability(1400), classify.win_probability(900)
    assert (before - after) * 100 < classify.INACCURACY_WIN_PCT
    assert classify.classify_by_win_probability(100 * (before - after)) == "ok"
    # ...while the same 500cp from equality is a blunder.
    lively = classify.win_probability(250) - classify.win_probability(-250)
    assert classify.classify_by_win_probability(100 * lively) == "blunder"
