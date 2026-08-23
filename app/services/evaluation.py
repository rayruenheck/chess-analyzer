import chess

from app.config import settings
from app.services import cache
from app.services.stockfish import stockfish_engine


async def get_or_analyze(
    fen: str, depth: int | None = None, time_limit: float | None = None
) -> tuple[dict, bool]:
    """Returns (evaluation, was_cached). `fen` must already be normalized."""
    if time_limit is not None:
        # Achieved depth is unknown ahead of time, so there's nothing to key
        # a cache lookup on; the engine still runs and its result gets cached
        # under whatever depth it happens to reach.
        evaluation = await stockfish_engine.analyze_fen(fen, time_limit=time_limit)
        await cache.save_evaluation(evaluation)
        return evaluation, False

    depth = depth or settings.stockfish_default_depth
    cached = await cache.get_evaluation(fen, depth)
    if cached is not None:
        return cached, True

    evaluation = await stockfish_engine.analyze_fen(fen, depth=depth)
    await cache.save_evaluation(evaluation)
    return evaluation, False


async def get_or_compute_diff(fen: str, previous_fen: str, depth: int, score_cp: int) -> int:
    """Returns the evaluation change caused by the move from previous_fen to
    fen, from the perspective of whoever made it (positive = good move)."""
    existing = await cache.get_move_diff(fen, previous_fen, depth)
    if existing is not None:
        return existing

    previous_evaluation, _ = await get_or_analyze(previous_fen, depth)

    # The side to move in previous_fen is whoever played the move into fen.
    mover_is_white = chess.Board(previous_fen).turn == chess.WHITE
    raw_diff = score_cp - previous_evaluation["score_cp"]
    diff_cp = raw_diff if mover_is_white else -raw_diff

    await cache.save_move_diff(fen, previous_fen, depth, diff_cp)
    return diff_cp
