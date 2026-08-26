"""Names the tactic that punished a move.

"You lost 40 points of expected score" tells a player what it cost. "You walked
into a fork" tells them what to practise, and those are different sentences. The
gap between them is this module: it looks at the refutation the engine found and
names the pattern, using the same vocabulary Lichess tags its puzzles with, so a
weakness found in a player's own games can be turned straight into a drill set
filtered by theme.

Everything here is computed from board geometry, so a tag is a fact rather than a
guess. Only the motifs that can be identified without ambiguity are detected --
deflection and overloading need to know what a piece was doing before it moved,
which is a judgement, and inventing those would put a confident wrong noun in
front of the player.
"""

import chess

# Detected motifs, named as Lichess names them in its puzzle database so the
# themes line up with a puzzle set the player can actually be sent to.
FORK = "fork"
PIN = "pin"
SKEWER = "skewer"
DISCOVERED_ATTACK = "discoveredAttack"
HANGING_PIECE = "hangingPiece"
DOUBLE_CHECK = "doubleCheck"
BACK_RANK_MATE = "backRankMate"
PROMOTION = "promotion"
MATE = "mate"

_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10_000,
}
_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def _worth_forking(board: chess.Board, square: int, attacker_value: int, defender: bool) -> bool:
    """A target worth counting: more valuable than the attacker, or undefended."""
    piece = board.piece_at(square)
    if piece is None:
        return False
    if piece.piece_type == chess.KING:
        return True
    if _VALUE[piece.piece_type] > attacker_value:
        return True
    return not board.attackers(defender, square)


def _line_targets(board: chess.Board, origin: int, victim_colour: bool) -> list[tuple[int, int]]:
    """Pairs of enemy pieces standing one behind the other on a line from `origin`."""
    pairs = []
    piece = board.piece_at(origin)
    if piece is None or piece.piece_type not in _SLIDERS:
        return pairs

    for direction in _ray_directions(piece.piece_type):
        found: list[int] = []
        square = origin
        while True:
            square = _step(square, direction)
            if square is None:
                break
            occupant = board.piece_at(square)
            if occupant is None:
                continue
            if occupant.color != victim_colour:
                break
            found.append(square)
            if len(found) == 2:
                pairs.append((found[0], found[1]))
                break
    return pairs


def _ray_directions(piece_type: int) -> list[tuple[int, int]]:
    diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    straight = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if piece_type == chess.BISHOP:
        return diagonal
    if piece_type == chess.ROOK:
        return straight
    return diagonal + straight


def _step(square: int, direction: tuple[int, int]) -> int | None:
    file_ = chess.square_file(square) + direction[0]
    rank = chess.square_rank(square) + direction[1]
    if 0 <= file_ <= 7 and 0 <= rank <= 7:
        return chess.square(file_, rank)
    return None


def tag(fen: str, move_uci: str) -> list[str]:
    """The tactical motifs in `move_uci`, played from `fen`.

    `fen` is the position the punishing move is played from, so for a blunder that
    is the position after the blunder, and the move is the opponent's refutation.
    """
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return []
    if move not in board.legal_moves:
        return []

    mover = board.turn
    victim = not mover
    found: set[str] = set()

    # Captured something that nothing was defending.
    if board.is_capture(move) and not board.is_en_passant(move):
        target = board.piece_at(move.to_square)
        if target and not board.attackers(victim, move.to_square):
            found.add(HANGING_PIECE)

    if move.promotion:
        found.add(PROMOTION)

    after = board.copy(stack=False)
    after.push(move)

    if after.is_checkmate():
        found.add(MATE)
        king = after.king(victim)
        back_rank = 0 if victim == chess.WHITE else 7
        giver = after.piece_at(move.to_square)
        if (
            king is not None
            and chess.square_rank(king) == back_rank
            and giver is not None
            and giver.piece_type in (chess.ROOK, chess.QUEEN)
        ):
            found.add(BACK_RANK_MATE)

    if after.is_check():
        king = after.king(victim)
        if king is not None and len(after.attackers(mover, king)) >= 2:
            found.add(DOUBLE_CHECK)

    # Fork: the piece that just moved now hits two or more things worth taking.
    landed = after.piece_at(move.to_square)
    if landed:
        attacker_value = _VALUE[landed.piece_type]
        targets = [
            square
            for square in after.attacks(move.to_square)
            if _worth_forking(after, square, attacker_value, victim)
            and after.piece_at(square)
            and after.piece_at(square).color == victim
        ]
        if len(targets) >= 2:
            found.add(FORK)

    # Pin and skewer, from whichever slider is now lined up on two enemy pieces.
    for origin in list(after.pieces(chess.QUEEN, mover)) + list(
        after.pieces(chess.ROOK, mover)
    ) + list(after.pieces(chess.BISHOP, mover)):
        for front, behind in _line_targets(after, origin, victim):
            front_value = _VALUE[after.piece_at(front).piece_type]
            behind_value = _VALUE[after.piece_at(behind).piece_type]
            if behind_value > front_value:
                found.add(PIN)
            elif front_value > behind_value:
                found.add(SKEWER)

    # Discovered attack: a different friendly slider gained a valuable target by
    # the mover stepping out of its way.
    for origin in list(after.pieces(chess.QUEEN, mover)) + list(
        after.pieces(chess.ROOK, mover)
    ) + list(after.pieces(chess.BISHOP, mover)):
        if origin == move.to_square:
            continue
        gained = after.attacks(origin) & after.occupied_co[victim]
        was = board.attacks(origin) & board.occupied_co[victim] if board.piece_at(origin) else chess.SquareSet()
        for square in gained & ~chess.SquareSet(was):
            piece = after.piece_at(square)
            if piece and _VALUE[piece.piece_type] >= _VALUE[chess.KNIGHT]:
                found.add(DISCOVERED_ATTACK)
                break

    return sorted(found)
