from enum import Enum

from fastapi import APIRouter, HTTPException
from httpx import HTTPStatusError

from app.schemas import NormalizedGame
from app.services.games import get_game_history

router = APIRouter(prefix="/games", tags=["games"])


class Platform(str, Enum):
    lichess = "lichess"
    chesscom = "chesscom"


@router.get("/{platform}/{username}", response_model=list[NormalizedGame])
async def get_games(platform: Platform, username: str, max_games: int = 20):
    try:
        return await get_game_history(platform.value, username, max_games)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
