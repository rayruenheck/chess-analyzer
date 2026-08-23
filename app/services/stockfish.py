import asyncio

import chess
import chess.engine

from app.config import settings


class StockfishEngine:
    """Wraps a single persistent Stockfish process behind an asyncio lock.

    python-chess's engine protocol is synchronous-transport but exposes an
    asyncio-friendly popen_uci, so all calls still need to go through the
    engine's own event loop transport methods.
    """

    def __init__(self) -> None:
        self._transport: chess.engine.UciProtocol | None = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> chess.engine.UciProtocol:
        if self._transport is None:
            _, engine = await chess.engine.popen_uci(settings.stockfish_path)
            self._transport = engine
        return self._transport

    async def analyze_fen(
        self, fen: str, depth: int | None = None, time_limit: float | None = None
    ) -> dict:
        board = chess.Board(fen)
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)
        async with self._lock:
            engine = await self._ensure_started()
            info = await engine.analyse(board, limit)

        score = info["score"].white()
        return {
            "fen": fen,
            "depth": info.get("depth"),
            "best_move": info["pv"][0].uci() if info.get("pv") else None,
            "principal_variation": [m.uci() for m in info.get("pv", [])],
            "score_cp": score.score(mate_score=100000),
            "mate_in": score.mate(),
        }

    async def best_move(
        self, fen: str, depth: int | None = None, time_limit: float | None = None
    ) -> dict:
        board = chess.Board(fen)
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)
        async with self._lock:
            engine = await self._ensure_started()
            result = await engine.play(board, limit)

        return {
            "fen": fen,
            "best_move": result.move.uci() if result.move else None,
            "ponder": result.ponder.uci() if result.ponder else None,
        }

    async def close(self) -> None:
        if self._transport is not None:
            await self._transport.quit()
            self._transport = None


stockfish_engine = StockfishEngine()
