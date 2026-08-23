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
    # and back off; this is the proactive half of that bargain.
    MIN_REQUEST_INTERVAL = 0.7

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

    async def _wait_turn(self) -> None:
        async with self._throttle:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.MIN_REQUEST_INTERVAL:
                await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            self._last_request = time.monotonic()

    async def get_masters(
        self, fen: str, moves: int = 12, top_games: int = 4
    ) -> dict:
        await self._wait_turn()
        resp = await self._client.get(
            "/masters",
            params={"fen": fen, "moves": moves, "topGames": top_games},
        )
        resp.raise_for_status()
        return resp.json()

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
        await self._wait_turn()
        resp = await self._client.get("/lichess", params=params)
        resp.raise_for_status()
        return resp.json()

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
