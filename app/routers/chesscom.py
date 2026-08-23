from fastapi import APIRouter, HTTPException
from httpx import HTTPStatusError

from app.clients.chesscom import chesscom_client

router = APIRouter(prefix="/chesscom", tags=["chess.com"])


@router.get("/player/{username}")
async def get_player(username: str):
    try:
        return await chesscom_client.get_player(username)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/player/{username}/stats")
async def get_player_stats(username: str):
    try:
        return await chesscom_client.get_player_stats(username)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/player/{username}/games/current")
async def get_current_games(username: str):
    try:
        return await chesscom_client.get_player_current_games(username)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/player/{username}/games/{year}/{month}")
async def get_games_by_month(username: str, year: int, month: int):
    try:
        return await chesscom_client.get_player_games_by_month(username, year, month)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
