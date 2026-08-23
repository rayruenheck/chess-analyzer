import httpx

from app.config import settings


class ChessComClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.chesscom_base_url,
            headers={"User-Agent": settings.chesscom_user_agent},
            timeout=15.0,
        )

    async def get_player(self, username: str) -> dict:
        resp = await self._client.get(f"/player/{username}")
        resp.raise_for_status()
        return resp.json()

    async def get_player_stats(self, username: str) -> dict:
        resp = await self._client.get(f"/player/{username}/stats")
        resp.raise_for_status()
        return resp.json()

    async def get_player_current_games(self, username: str) -> dict:
        resp = await self._client.get(f"/player/{username}/games")
        resp.raise_for_status()
        return resp.json()

    async def get_player_games_by_month(self, username: str, year: int, month: int) -> dict:
        resp = await self._client.get(f"/player/{username}/games/{year}/{month:02d}")
        resp.raise_for_status()
        return resp.json()

    async def get_player_game_archives(self, username: str) -> list[str]:
        resp = await self._client.get(f"/player/{username}/games/archives")
        resp.raise_for_status()
        return resp.json().get("archives", [])

    async def close(self) -> None:
        await self._client.aclose()


chesscom_client = ChessComClient()
