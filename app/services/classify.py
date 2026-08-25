"""Deterministic move classification.

Everything here is computed from Stockfish output and python-chess board state --
no model is involved. This is deliberately the layer that decides *what is true*
about a move, so the LLM layer only ever has to decide *how to say it*. Tags are
kept to facts that are certain (mate scores, engine agreement, board geometry)
rather than judgement calls, because a wrong tag becomes a confidently wrong
sentence in the coaching output.
"""

import math

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Centipawns lost by the mover, as bands. Chosen to line up with what players are
# used to seeing on Lichess/Chess.com so the labels don't need explaining.
BLUNDER = 300
MISTAKE = 100
INACCURACY = 50

# stockfish.py reports mate as +/-100000 via mate_score. Anything near that is a
# forced mate rather than a real centipawn count, and must not be averaged in.
MATE_THRESHOLD = 90000

# Phase boundaries, in non-pawn material summed over both sides (3200 each at the
# start). The endgame cut sits above a double-rook ending (2000) so that R+R vs R+R
# is not called a middlegame, and below queen-and-rook apiece (2800), where there is
# still a real middlegame to play.
ENDGAME_MATERIAL = 2400
OPENING_MATERIAL = 5000
# Past this move a position is no longer the opening however little has been traded.
OPENING_MOVES = 12

# A move that walks into mate produces a ~99000 centipawn "loss", which would
# single-handedly dominate any average taken over a game. The severity label and
# the allows_forced_mate tag already carry that the move was catastrophic, so the
# number itself gets capped to keep aggregate stats meaningful.
#
# The cap is right for averaging and wrong for ranking, which is what win_prob_lost
# below exists for: every mate-allowing move lands on exactly this value, so sorting
# by centipawn loss collapses them into one indistinguishable tie at the ceiling.
MAX_REPORTED_CP_LOSS = 1000

# Centipawns beyond which a position is treated as already decided. Loss taken from
# such a position is mostly not a lesson -- walking into mate when already down a
# queen is a formality, not the error that lost the game.
DECIDED_CP = 500

# Logistic scale converting a centipawn evaluation into an expected score. Shared
# with the eval bar in the UI so the two never disagree.
#
# Lichess publishes a scale of 1/271.6, fitted against games between 2300-rated
# players. Deliberately not used here: checked against this app's own games, that
# curve badly under-rates losing positions at club level -- it calls a -5 position
# a 3.5% expected score where the real games score far higher, because below master
# level a lost position is still very much alive. A flatter curve is the safer error
# for coaching, since the alternative is telling a player that throwing away a bad
# position cost them nothing.
WIN_PROB_SCALE = 320


def win_probability(score_cp: int | None, mate_in: int | None = None) -> float | None:
    """Expected score for White, 0..1, from an engine evaluation.

    Expected score rather than literal win probability: draws are folded in, the
    same way Lichess's published "Win%" does it.
    """
    if mate_in is not None:
        return 1.0 if mate_in > 0 else 0.0
    if score_cp is None:
        return None
    # Mate is reported as +/-100000 centipawns, which overflows exp() unclamped.
    exponent = max(-40.0, min(40.0, score_cp / WIN_PROB_SCALE))
    return 1 / (1 + math.exp(-exponent))


def classify_severity(eval_diff_cp: int | None) -> str:
    """`eval_diff_cp` is from the mover's perspective: negative means they lost ground.

    Callers should suppress this to "ok" when the move played was the engine's own
    first choice -- see describe_move. The swing between two evaluations is not a
    pure measure of move quality: both sides of the diff carry search noise, and a
    forced recapture can post a triple-digit "loss" at shallow depth. Labelling the
    best available move an inaccuracy would put a flatly false claim in front of
    the player, so agreement with the engine wins over the arithmetic.
    """
    if eval_diff_cp is None:
        return "unknown"

    loss = -eval_diff_cp
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return "ok"


def cp_lost(eval_diff_cp: int | None) -> int | None:
    """Centipawns the mover gave up, clamped to MAX_REPORTED_CP_LOSS.

    Clamping matters because these values get averaged across a game and across a
    whole report; a single mate-in-N would otherwise swamp every other move.
    """
    if eval_diff_cp is None:
        return None
    return min(max(0, -eval_diff_cp), MAX_REPORTED_CP_LOSS)


def is_mate_score(score_cp: int | None) -> bool:
    return score_cp is not None and abs(score_cp) >= MATE_THRESHOLD


def phase(board: chess.Board, move_number: int | None = None) -> str:
    """Opening / middlegame / endgame.

    Material is the primary signal, because move number lies in both directions: a
    queen trade on move 12 is an endgame, and a 30-move theoretical line is still
    the opening. Both sides start with 3200 of non-pawn material.

    Move number is still needed to close one gap material cannot see. A closed
    position -- a King's Indian with the centre locked, say -- can reach move 30
    with every piece still on the board, and material alone keeps calling that the
    opening long after theory ended. When the move number is known it caps how long
    a position can be called an opening; when it is not (a single position analysed
    out of context) the material test stands alone.
    """
    non_pawn = sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.piece_type not in (chess.PAWN, chess.KING)
    )
    if non_pawn <= ENDGAME_MATERIAL:
        return "endgame"
    if non_pawn >= OPENING_MATERIAL and (move_number is None or move_number <= OPENING_MOVES):
        return "opening"
    return "middlegame"


def _hangs_piece(board_after: chess.Board, to_square: int) -> bool:
    """True when the just-moved piece sits undefended under attack.

    Restricted to the unambiguous case (attacked, zero defenders) rather than a
    full static exchange evaluation -- a half-right exchange heuristic would put
    claims in the coaching text that the engine line then contradicts.
    """
    piece = board_after.piece_at(to_square)
    if piece is None or piece.piece_type == chess.KING:
        return False

    attackers = board_after.attackers(board_after.turn, to_square)
    if not attackers:
        return False

    defenders = board_after.attackers(not board_after.turn, to_square)
    if defenders:
        return False

    return PIECE_VALUES[piece.piece_type] >= PIECE_VALUES[chess.PAWN]


def pv_to_san(fen: str, pv_uci: list[str], limit: int = 6) -> list[str]:
    """Renders an engine principal variation as SAN from `fen`.

    The PV in SAN is the single most useful thing the model receives -- "Rxe6 Bxe6
    Qxd5+" is explainable, "e1e6 c8e6 d1d5" is not.
    """
    board = chess.Board(fen)
    san_moves: list[str] = []
    for uci in pv_uci[:limit]:
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            san_moves.append(board.san(move))
            board.push(move)
        except ValueError:
            break
    return san_moves


def describe_move(
    fen_before: str,
    move_uci: str,
    eval_diff_cp: int | None,
    best_move_uci: str | None,
    score_cp_after: int | None,
    mate_in_after: int | None,
    best_pv_uci: list[str] | None = None,
    move_number: int | None = None,
    score_cp_before: int | None = None,
    mate_in_before: int | None = None,
) -> dict:
    """Facts about one move, ready to hand to the LLM layer verbatim.

    `fen_before` is a normalized FEN (no halfmove/fullmove counters), so
    fullmove_number is always 1 here and move numbering comes from the ply index
    the caller already has.
    """
    board = chess.Board(fen_before)
    move = chess.Move.from_uci(move_uci)
    legal = move in board.legal_moves

    played_san = board.san(move) if legal else move_uci
    best_san = None
    if best_move_uci:
        try:
            best = chess.Move.from_uci(best_move_uci)
            if best in board.legal_moves:
                best_san = board.san(best)
        except ValueError:
            pass

    tags: list[str] = []
    if legal and board.is_capture(move):
        tags.append("capture")
    if legal and board.gives_check(move):
        tags.append("gives_check")
    if board.is_check():
        tags.append("was_in_check")

    board_after = board.copy()
    if legal:
        board_after.push(move)
        if _hangs_piece(board_after, move.to_square):
            tags.append("hangs_piece")

    played_best = best_move_uci is not None and move_uci == best_move_uci
    if played_best:
        tags.append("engine_best")

    # Mate facts come straight off the engine score, so they are certain.
    if mate_in_after is not None:
        mover_is_white = board.turn == chess.WHITE
        mate_favours_mover = (mate_in_after > 0) == mover_is_white
        tags.append("has_forced_mate" if mate_favours_mover else "allows_forced_mate")

    # How much of the game the move actually threw away, as expected score. This
    # is what "how bad was it" has to mean for ranking: -800 to -1200 is a rounding
    # error and +100 to -300 is most of the game, and centipawns rate them alike.
    mover_is_white = board.turn == chess.WHITE

    def for_mover(score: int | None, mate: int | None) -> float | None:
        chance = win_probability(score, mate)
        return None if chance is None else (chance if mover_is_white else 1 - chance)

    before_chance = for_mover(score_cp_before, mate_in_before)
    after_chance = for_mover(score_cp_after, mate_in_after)
    if before_chance is None or after_chance is None:
        win_prob_lost = None
    elif played_best:
        # Same reasoning as the severity label: the engine's own first choice is
        # never charged for the swing, which carries search noise at both ends.
        win_prob_lost = 0.0
    else:
        win_prob_lost = round(100 * (before_chance - after_chance), 1)

    return {
        "played_san": played_san,
        "played_uci": move_uci,
        "best_san": best_san,
        # UCI as well as SAN because the board draws the engine's choice as an
        # arrow, which needs the two squares rather than the algebraic name.
        "best_uci": best_move_uci if best_san else None,
        "best_line_san": pv_to_san(fen_before, best_pv_uci or []),
        "severity": "ok" if played_best else classify_severity(eval_diff_cp),
        "cp_lost": cp_lost(eval_diff_cp),
        "eval_before_cp": None if is_mate_score(score_cp_before) else score_cp_before,
        "eval_after_cp": None if is_mate_score(score_cp_after) else score_cp_after,
        # Percentage points of expected score, from the mover's own perspective.
        "win_prob_lost": win_prob_lost,
        "position_state": position_state(score_cp_before, mate_in_before, mover_is_white),
        "mate_in": mate_in_after,
        "phase": phase(board, move_number),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "tags": tags,
    }


def position_state(
    score_cp: int | None, mate_in: int | None, mover_is_white: bool
) -> str | None:
    """Whether the mover was already winning, already lost, or still in a game.

    Sent with every critical moment because it decides whether a mistake is worth
    coaching -- and because blundering a won position is a different habit from
    blundering an equal one, and needs different advice.
    """
    if mate_in is not None:
        return "winning" if (mate_in > 0) == mover_is_white else "losing"
    if score_cp is None:
        return None
    from_mover = score_cp if mover_is_white else -score_cp
    if from_mover > DECIDED_CP:
        return "winning"
    if from_mover < -DECIDED_CP:
        return "losing"
    return "competitive"
