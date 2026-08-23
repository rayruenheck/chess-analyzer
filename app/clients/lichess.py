import json

import httpx

from app.config import settings


class LichessClient:
    def __init__(self) -> None:
        headers = {"Accept": "application/json", "User-Agent": settings.lichess_user_agent}
        if settings.lichess_token:
            headers["Authorization"] = f"Bearer {settings.lichess_token}"
        self._client = httpx.AsyncClient(
            base_url=settings.lichess_base_url,
            headers=headers,
            timeout=15.0,
        )

    async def get_user(self, username: str) -> dict:
        resp = await self._client.get(f"/api/user/{username}")
        resp.raise_for_status()
        return resp.json()

    async def get_user_games(self, username: str, max_games: int = 20) -> list[dict]:
        resp = await self._client.get(
            f"/api/games/user/{username}",
            params={"max": max_games, "pgnInJson": "true"},
            headers={"Accept": "application/x-ndjson"},
        )
        resp.raise_for_status()
        return [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    async def get_cloud_eval(self, fen: str, multi_pv: int = 1) -> dict:
        resp = await self._client.get(
            "/api/cloud-eval",
            params={"fen": fen, "multiPv": multi_pv},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()


lichess_client = LichessClient()
