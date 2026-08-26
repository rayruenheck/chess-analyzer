"""Positional facts about a board, computed rather than inferred.

The coaching prompt asks the model to teach in terms of imbalances -- superior
minor piece, pawn structure, space, files and squares, development, king safety --
which it cannot do from centipawn numbers alone. Handed only a severity and a
clock reading it will either fall back to reciting statistics or invent positional
claims to fit a concept, and the second is worse than the first.

So the imbalances are derived here, from the board, deterministically. This module
is the same kind of thing as classify.py: it decides what is *true* about a
position so the model only ever has to decide how to say it. Nothing here is a
judgement call -- every value is counted off the FEN.

Only notable features are returned. A position where neither side holds an
imbalance produces an empty dict rather than a wall of zeroes, both to keep the
payload small and because "nothing to say here" is the honest answer.
"""

from collections import Counter

import chess

# A bishop with at least this many of its own pawns fixed on its own colour is
# what Silman calls a bad bishop -- hemmed in by the structure behind it. Four is
# the starting position for every bishop, so the bar sits above that.
BAD_BISHOP_PAWNS = 5
# Space and development are reported only when one side is meaningfully ahead;
# a one-square edge is noise.
SPACE_EDGE = 4
DEVELOPMENT_EDGE = 2


def _pawn_files(board: chess.Board, colour: bool) -> Counter:
    return Counter(chess.square_file(s) for s in board.pieces(chess.PAWN, colour))


def _is_passed(board: chess.Board, square: int, colour: bool) -> bool:
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    for enemy in board.pieces(chess.PAWN, not colour):
        if abs(chess.square_file(enemy) - file_) > 1:
            continue
        enemy_rank = chess.square_rank(enemy)
        ahead = enemy_rank > rank if colour == chess.WHITE else enemy_rank < rank
        if ahead:
            return False
    return True


def pawn_structure(board: chess.Board, colour: bool) -> dict:
    """Isolated, doubled and passed pawns, plus how many islands they form."""
    counts = _pawn_files(board, colour)

    isolated = sum(
        n for f, n in counts.items() if (f - 1) not in counts and (f + 1) not in counts
    )
    doubled = sum(n - 1 for n in counts.values() if n > 1)

    occupied = sorted(counts)
    islands = sum(
        1 for i, f in enumerate(occupied) if i == 0 or f != occupied[i - 1] + 1
    )
    passed = sum(
        1 for s in board.pieces(chess.PAWN, colour) if _is_passed(board, s, colour)
    )

    return {"isolated": isolated, "doubled": doubled, "passed": passed, "islands": islands}


def bad_bishops(board: chess.Board, colour: bool) -> list[dict]:
    """Each bishop with the count of own pawns sharing its colour complex."""
    out = []
    for square in board.pieces(chess.BISHOP, colour):
        light = bool((chess.square_rank(square) + chess.square_file(square)) % 2)
        blocked = sum(
            1
            for p in board.pieces(chess.PAWN, colour)
            if bool((chess.square_rank(p) + chess.square_file(p)) % 2) == light
        )
        out.append({"square": chess.square_name(square), "own_pawns_on_its_colour": blocked})
    return out


def outposts(board: chess.Board, colour: bool) -> list[str]:
    """Knights in enemy territory, pawn-defended, that no enemy pawn can evict."""
    found = []
    for square in board.pieces(chess.KNIGHT, colour):
        rank = chess.square_rank(square)
        advanced = rank >= 4 if colour == chess.WHITE else rank <= 3
        if not advanced:
            continue

        defenders = board.attackers(colour, square)
        if not any(board.piece_type_at(d) == chess.PAWN for d in defenders):
            continue

        file_ = chess.square_file(square)
        evictable = False
        for enemy in board.pieces(chess.PAWN, not colour):
            if abs(chess.square_file(enemy) - file_) != 1:
                continue
            enemy_rank = chess.square_rank(enemy)
            behind = enemy_rank > rank if colour == chess.WHITE else enemy_rank < rank
            if behind:
                evictable = True
                break
        if not evictable:
            found.append(chess.square_name(square))
    return found


def file_control(board: chess.Board, colour: bool) -> dict:
    """Open files, and whether this side's rooks are actually using them."""
    own = {chess.square_file(s) for s in board.pieces(chess.PAWN, colour)}
    theirs = {chess.square_file(s) for s in board.pieces(chess.PAWN, not colour)}

    open_files = [chess.FILE_NAMES[f] for f in range(8) if f not in own and f not in theirs]
    semi_open = [chess.FILE_NAMES[f] for f in range(8) if f not in own and f in theirs]

    seventh = 6 if colour == chess.WHITE else 1
    rooks = list(board.pieces(chess.ROOK, colour))
    return {
        "open_files": open_files,
        "your_semi_open_files": semi_open,
        "rooks_on_open_files": sum(
            1 for r in rooks if chess.FILE_NAMES[chess.square_file(r)] in open_files
        ),
        "rooks_on_seventh": sum(1 for r in rooks if chess.square_rank(r) == seventh),
    }


def king_safety(board: chess.Board, colour: bool) -> dict | None:
    """Where the king sits, how much pawn cover it has, and whether it is exposed."""
    king = board.king(colour)
    if king is None:
        return None

    king_file = chess.square_file(king)
    home = 0 if colour == chess.WHITE else 7
    shield = 0
    for pawn in board.pieces(chess.PAWN, colour):
        if abs(chess.square_file(pawn) - king_file) > 1:
            continue
        distance = (
            chess.square_rank(pawn) - home
            if colour == chess.WHITE
            else home - chess.square_rank(pawn)
        )
        if 0 < distance <= 2:
            shield += 1

    own_files = {chess.square_file(s) for s in board.pieces(chess.PAWN, colour)}
    return {
        "square": chess.square_name(king),
        "pawn_shield": shield,
        "on_open_file": king_file not in own_files,
    }


def space(board: chess.Board, colour: bool) -> int:
    """Squares controlled in the enemy half -- the usual proxy for space."""
    half = range(32, 64) if colour == chess.WHITE else range(0, 32)
    return sum(1 for s in half if board.is_attacked_by(colour, s))


def developed_minors(board: chess.Board, colour: bool) -> int:
    home = 0 if colour == chess.WHITE else 7
    return sum(
        1
        for piece_type in (chess.KNIGHT, chess.BISHOP)
        for s in board.pieces(piece_type, colour)
        if chess.square_rank(s) != home
    )


def features(fen: str, colour: str, phase: str | None = None) -> dict:
    """The imbalances in a position, from `colour`'s point of view.

    Returns only what is actually notable, so an unremarkable position yields an
    empty dict and the model is not handed a page of zeroes to read meaning into.
    """
    board = chess.Board(fen)
    me = chess.WHITE if colour == "white" else chess.BLACK
    them = not me

    out: dict = {}

    mine = pawn_structure(board, me)
    theirs = pawn_structure(board, them)
    my_weak = {k: v for k, v in mine.items() if k != "islands" and v}
    their_weak = {k: v for k, v in theirs.items() if k != "islands" and v}
    if my_weak or their_weak or mine["islands"] != theirs["islands"]:
        out["pawn_structure"] = {
            "yours": {**my_weak, "islands": mine["islands"]},
            "theirs": {**their_weak, "islands": theirs["islands"]},
        }

    my_bishops = len(board.pieces(chess.BISHOP, me))
    their_bishops = len(board.pieces(chess.BISHOP, them))
    if my_bishops >= 2 and their_bishops < 2:
        out["bishop_pair"] = "you"
    elif their_bishops >= 2 and my_bishops < 2:
        out["bishop_pair"] = "opponent"

    bad = [
        b for b in bad_bishops(board, me)
        if b["own_pawns_on_its_colour"] >= BAD_BISHOP_PAWNS
    ]
    if bad:
        out["your_bad_bishops"] = bad

    my_outposts = outposts(board, me)
    their_outposts = outposts(board, them)
    if my_outposts or their_outposts:
        out["outposts"] = {"yours": my_outposts, "theirs": their_outposts}

    files = file_control(board, me)
    if files["open_files"] or files["rooks_on_seventh"]:
        out["files"] = files

    my_king = king_safety(board, me)
    their_king = king_safety(board, them)
    if my_king and (my_king["pawn_shield"] <= 1 or my_king["on_open_file"]):
        out["your_king"] = my_king
    if their_king and (their_king["pawn_shield"] <= 1 or their_king["on_open_file"]):
        out["their_king"] = their_king

    my_space, their_space = space(board, me), space(board, them)
    if abs(my_space - their_space) >= SPACE_EDGE:
        out["space"] = {"yours": my_space, "theirs": their_space}

    if phase == "opening":
        my_dev, their_dev = developed_minors(board, me), developed_minors(board, them)
        if abs(my_dev - their_dev) >= DEVELOPMENT_EDGE:
            out["developed_minor_pieces"] = {"yours": my_dev, "theirs": their_dev}

    return out
