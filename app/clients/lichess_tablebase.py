"""Syzygy tablebase lookups against tablebase.lichess.ovh.

Different in kind from everything else the app asks an engine or an API for: a
tablebase answer is not an evaluation, it is the result with perfect play. There
is no depth to argue with and no noise to filter. So when a position was a draw
and the next one is a loss, that is not a judgement that the move was bad -- the
game was thrown away, and the coaching can say so flatly.

Complete to 7 pieces, which is the ceiling on when this is worth asking.
"""

import asyncio
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# The most pieces Syzygy covers completely. Above this the endpoint has nothing.
MAX_PIECES = 7


def piece_count(fen: str) -> int:
    return sum(1 for character in fen.split()[0] if character.isalpha())


def in_range(fen: str) -> bool:
    return piece_count(fen) <= MAX_PIECES


def complete(fen: str) -> str:
    """Restores the counters the app strips from FENs before caching them.

    Evaluations are keyed on a normalized FEN with the halfmove and fullmove
    counters removed, since they do not change how a position evaluates. The
    tablebase rejects that outright, and it genuinely needs the halfmove clock:
    the fifty-move rule is what separates a win from a "cursed win".

    Zero is the honest default here. The counter is not stored, and asking with a
    fresh clock gives the theoretical result ignoring the fifty-move rule, which
    is the result worth coaching -- a technique error is a technique error whether
    or not the rule would have rescued it.
    """
    fields = fen.split()
    if len(fields) >= 6:
        return fen
    return " ".join(fields + ["0"] * (5 - len(fields)) + ["1"])


class LichessTablebaseClient:
    """Serial, throttled client. Lichess publishes no limit here; be polite anyway."""

    MIN_REQUEST_INTERVAL = 0.3
    DEFAULT_RETRY_AFTER = 5.0
    MAX_RETRIES = 2

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.lichess_tablebase_base_url,
            headers={
                "Accept": "application/json",
                "User-Agent": settings.lichess_user_agent,
            },
            timeout=20.0,
        )
        self._throttle = asyncio.Lock()
        self._last_request = 0.0

    async def _wait_turn(self) -> None:
        async with self._throttle:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.MIN_REQUEST_INTERVAL:
                await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            self._last_request = time.monotonic()

    async def lookup(self, fen: str) -> dict | None:
        """The theoretical result for `fen`, or None if it is out of range."""
        if not in_range(fen):
            return None

        for attempt in range(self.MAX_RETRIES + 1):
            await self._wait_turn()
            response = await self._client.get("/standard", params={"fen": complete(fen)})
            if response.status_code == 404:
                return None
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            if attempt == self.MAX_RETRIES:
                response.raise_for_status()
            await asyncio.sleep(self.DEFAULT_RETRY_AFTER)
        return None

    async def close(self) -> None:
        await self._client.aclose()


lichess_tablebase_client = LichessTablebaseClient()
