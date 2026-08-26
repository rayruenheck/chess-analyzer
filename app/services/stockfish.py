"""A pool of Stockfish processes.

The workload is thousands of independent positions at a fixed depth, so what
matters is throughput, not how fast any one search finishes. That points at
several engines running side by side rather than one engine given several
threads: parallel search scales sublinearly and a pool scales close to linearly
on independent work.

Determinism matters as much as speed here, because every cached evaluation is
keyed by `(fen, depth)` on the assumption that the pair has exactly one answer.
Two things are needed for that to hold, and only one of them is obvious.

Each engine is pinned to a single thread: a multi-threaded search explores the
tree in an order that depends on thread timing, so the same position at the same
depth can come back with a different move.

Less obviously, each search starts from a clean transposition table. A table
carried over from the previous position changes what this search finds, so the
answer depends on which engine happened to take the job and what it looked at
before. Measured on 44 positions at depth 18, six engines with a warm table
disagreed with the serial run on nine best moves and by up to 30 centipawns; with
a clean table they agree exactly. The cost is about 5% -- far less than the
correctness is worth, and the same run is still nearly four times faster than one
engine doing the work alone.

Engines start lazily and are reused. A crashed engine is dropped rather than
returned to the pool, so one bad position cannot poison every later request.
"""

import asyncio
import contextlib
import logging

import chess
import chess.engine

from app.config import settings

logger = logging.getLogger(__name__)


class StockfishPool:
    """Several persistent Stockfish processes, handed out one at a time."""

    def __init__(self) -> None:
        self._idle: asyncio.Queue[chess.engine.UciProtocol] = asyncio.Queue()
        self._started = 0
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return max(1, settings.stockfish_workers)

    async def _spawn(self) -> chess.engine.UciProtocol:
        _, engine = await chess.engine.popen_uci(settings.stockfish_path)
        # Threads=1 is what keeps a fixed-depth search reproducible, which the
        # (fen, depth) cache depends on. Hash is per process, so the pool's total
        # memory is this figure times the worker count.
        await engine.configure(
            {"Threads": 1, "Hash": max(16, settings.stockfish_hash_mb)}
        )
        return engine

    @contextlib.asynccontextmanager
    async def _engine(self):
        """Checks out an engine, growing the pool up to `size` on demand."""
        engine: chess.engine.UciProtocol | None = None
        if self._idle.empty():
            async with self._lock:
                if self._idle.empty() and self._started < self.size:
                    engine = await self._spawn()
                    self._started += 1
        if engine is None:
            engine = await self._idle.get()

        try:
            yield engine
        except Exception:
            # A search that raised may have left the process unusable. Drop it and
            # let the pool grow a fresh one rather than handing the wreck onward.
            async with self._lock:
                self._started -= 1
            with contextlib.suppress(Exception):
                await engine.quit()
            raise
        else:
            self._idle.put_nowait(engine)

    async def analyze_fen(
        self, fen: str, depth: int | None = None, time_limit: float | None = None
    ) -> dict:
        board = chess.Board(fen)
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)

        async with self._engine() as engine:
            info = await engine.analyse(board, limit, game=object())

        score = info["score"].white()
        return {
            "fen": fen,
            "depth": info.get("depth"),
            "best_move": info["pv"][0].uci() if info.get("pv") else None,
            "principal_variation": [m.uci() for m in info.get("pv", [])],
            "score_cp": score.score(mate_score=100000),
            "mate_in": score.mate(),
        }

    async def analyse_alternatives(self, fen: str, depth: int | None = None) -> list[dict]:
        """The engine's top two moves and their scores, best first.

        The gap between them is how critical the position is. One move keeping the
        evaluation while every alternative throws it away is a moment the player had
        to find; a position where six moves are equal is one they could not really
        get wrong. Grading both the same way is what makes accuracy statistics feel
        unfair, and this is what lets a report separate them.
        """
        board = chess.Board(fen)
        limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)
        async with self._engine() as engine:
            infos = await engine.analyse(board, limit, multipv=2, game=object())

        out = []
        for info in infos:
            if not info.get("pv"):
                continue
            score = info["score"].white()
            out.append(
                {
                    "move": info["pv"][0].uci(),
                    "score_cp": score.score(mate_score=100000),
                    "mate_in": score.mate(),
                }
            )
        return out

    async def threat_after_null_move(self, fen: str, depth: int | None = None) -> dict | None:
        """What the opponent would do if handed a free move in this position.

        This is how a threat that was already there gets told apart from one the
        played move created. If the punishment is the same either way, the player
        did not blunder into it -- they were looking at their own plan while the
        opponent's was already on the board, which is a different error with a
        different fix (prophylaxis) from simply hanging something.

        Returns None when there was no free move to give, which in practice means
        the side to move is in check.
        """
        board = chess.Board(fen)
        if board.is_check():
            return None

        board.push(chess.Move.null())
        # Rebuilt from the resulting FEN rather than passed with its history: a
        # null move cannot be sent over UCI, so python-chess would fall back to the
        # FEN anyway and warn about it once per probe. This is the same position
        # with the en passant square already cleared by the null push.
        passed = chess.Board(board.fen())

        limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)
        async with self._engine() as engine:
            info = await engine.analyse(passed, limit, game=object())

        if not info.get("pv"):
            return None
        return {"move": info["pv"][0].uci(), "san": passed.san(info["pv"][0])}

    async def best_move(
        self, fen: str, depth: int | None = None, time_limit: float | None = None
    ) -> dict:
        board = chess.Board(fen)
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth or settings.stockfish_default_depth)

        async with self._engine() as engine:
            result = await engine.play(board, limit, game=object())

        return {
            "fen": fen,
            "best_move": result.move.uci() if result.move else None,
            "ponder": result.ponder.uci() if result.ponder else None,
        }

    async def close(self) -> None:
        async with self._lock:
            while not self._idle.empty():
                engine = self._idle.get_nowait()
                with contextlib.suppress(Exception):
                    await engine.quit()
            self._started = 0


stockfish_engine = StockfishPool()
