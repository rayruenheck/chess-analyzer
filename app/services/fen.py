def normalize_fen(fen: str) -> str:
    """Strips the halfmove clock and fullmove number.

    Two FENs that differ only in those counters are the same position for
    evaluation and cache purposes; keeping them in the key would prevent
    cache hits across transpositions and repeated positions in other games.
    """
    board, turn, castling, en_passant, *_ = fen.split()
    return f"{board} {turn} {castling} {en_passant}"
