from fastapi import APIRouter, HTTPException
from httpx import HTTPStatusError

from app.clients.lichess import lichess_client

router = APIRouter(prefix="/lichess", tags=["lichess"])


@router.get("/user/{username}")
async def get_user(username: str):
    try:
        return await lichess_client.get_user(username)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/user/{username}/games")
async def get_user_games(username: str, max_games: int = 20):
    try:
        return await lichess_client.get_user_games(username, max_games)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/cloud-eval")
async def cloud_eval(fen: str, multi_pv: int = 1):
    try:
        return await lichess_client.get_cloud_eval(fen, multi_pv)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
