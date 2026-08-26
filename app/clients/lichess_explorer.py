import asyncio
import json
import time

import httpx

from app.config import settings


class LichessExplorerClient:
    """Client for the Lichess Opening Explorer (masters/lichess/player DBs).

    This is a separate service from the main lichess.org API, hosted at
    explorer.lichess.org, with its own endpoints and rate limits. It answers 401 to
    anonymous requests, so settings.lichess_token must be set; any token works and
    no OAuth scopes are required.
    """

    # Seconds between requests. Lichess asks API clients to keep concurrency at one
    # and back off; this is the proactive half of that bargain. It is a starting
    # point rather than a fixed rate -- see _interval, which grows when the server
    # actually pushes back.
    MIN_REQUEST_INTERVAL = 0.7
    # A 429 means the opening rate was too high for this session, so slow down for
    # the rest of it rather than rediscovering the limit on every later request. A
    # report walks hundreds of positions; without this it re-hits the wall for each.
    BACKOFF_FACTOR = 1.6
    MAX_REQUEST_INTERVAL = 6.0
    # Lichess sends no Retry-After on explorer 429s, so this is the assumed cooldown.
    DEFAULT_RETRY_AFTER = 8.0
    MAX_RETRIES = 3

    def __init__(self) -> None:
        headers = {
            "Accept": "application/json",
            "User-Agent": settings.lichess_user_agent,
        }
        # The explorer now answers 401 to anonymous requests -- a bogus path on the
        # same host still returns 404, so nginx is gating the route rather than
        # blocking the caller, and its CORS preflight advertises Authorization.
        # The main lichess.org API remains usable without a token; this host is not.
        if settings.lichess_token:
            headers["Authorization"] = f"Bearer {settings.lichess_token}"

        self._client = httpx.AsyncClient(
            base_url=settings.lichess_explorer_base_url,
            headers=headers,
            timeout=30.0,
        )
        # The explorer rate-limits, and a report walking twenty games issues dozens
        # of lookups back to back. Serialising them behind a minimum interval keeps
        # the whole feature working instead of getting a 429 partway through and
        # losing every remaining game's analysis.
        self._throttle = asyncio.Lock()
        self._last_request = 0.0
        self._interval = self.MIN_REQUEST_INTERVAL

    async def _wait_turn(self) -> None:
        async with self._throttle:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_request = time.monotonic()

    def _slow_down(self) -> None:
        self._interval = min(self._interval * self.BACKOFF_FACTOR, self.MAX_REQUEST_INTERVAL)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    async def _get(self, path: str, params: dict) -> dict:
        """One throttled GET, retrying while the server is rate-limiting us.

        Without this a single 429 partway through a report killed the opening
        analysis for every remaining game -- the caller cannot retry usefully,
        because by then it has already given up on the whole feature. Backing off
        here keeps a slow answer, which is worth far more than no answer.
        """
        for attempt in range(self.MAX_RETRIES + 1):
            await self._wait_turn()
            resp = await self._client.get(path, params=params)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()

            self._slow_down()
            if attempt == self.MAX_RETRIES:
                resp.raise_for_status()
            await asyncio.sleep(self._retry_after(resp) or self.DEFAULT_RETRY_AFTER)

        raise RuntimeError("unreachable")

    async def get_masters(
        self, fen: str, moves: int = 12, top_games: int = 4
    ) -> dict:
        return await self._get(
            "/masters", {"fen": fen, "moves": moves, "topGames": top_games}
        )

    async def get_lichess(
        self,
        fen: str,
        ratings: list[int] | None = None,
        speeds: list[str] | None = None,
        moves: int = 12,
        top_games: int = 4,
    ) -> dict:
        params: dict = {"fen": fen, "moves": moves, "topGames": top_games}
        if ratings:
            params["ratings"] = ",".join(str(r) for r in ratings)
        if speeds:
            params["speeds"] = ",".join(speeds)
        return await self._get("/lichess", params)

    async def get_player(
        self,
        fen: str,
        player: str,
        color: str,
        speeds: list[str] | None = None,
        moves: int = 12,
        recent_games: int = 4,
    ) -> dict:
        params: dict = {
            "fen": fen,
            "player": player,
            "color": color,
            "moves": moves,
            "recentGames": recent_games,
        }
        if speeds:
            params["speeds"] = ",".join(speeds)

        # Streamed as ndjson: an initial partial result while lichess indexes
        # the player's games, followed by updates until indexing completes.
        # The last line is the most complete result.
        await self._wait_turn()
        async with self._client.stream("GET", "/player", params=params) as resp:
            resp.raise_for_status()
            last_line: str | None = None
            async for line in resp.aiter_lines():
                if line.strip():
                    last_line = line

        return json.loads(last_line) if last_line else {}

    async def close(self) -> None:
        await self._client.aclose()


lichess_explorer_client = LichessExplorerClient()
