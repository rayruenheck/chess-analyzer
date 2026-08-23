import json

import httpx

from app.config import settings


class LichessExplorerClient:
    """Client for the Lichess Opening Explorer (masters/lichess/player DBs).

    This is a separate service from the main lichess.org API, hosted at
    explorer.lichess.org, with its own endpoints and rate limits.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.lichess_explorer_base_url,
            headers={"Accept": "application/json", "User-Agent": settings.lichess_user_agent},
            timeout=30.0,
        )

    async def get_masters(
        self, fen: str, moves: int = 12, top_games: int = 4
    ) -> dict:
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
