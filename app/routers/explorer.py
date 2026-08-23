from fastapi import APIRouter, HTTPException, Query
from httpx import HTTPStatusError

from app.services import explorer

router = APIRouter(prefix="/explorer", tags=["explorer"])


@router.get("/masters")
async def masters(fen: str):
    try:
        return await explorer.get_masters(fen)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/lichess")
async def lichess_stats(
    fen: str,
    rating: int = 1500,
    speeds: list[str] = Query(default=["blitz", "rapid"]),
):
    try:
        return await explorer.get_lichess_stats(fen, rating, speeds)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc


@router.get("/player")
async def player_stats(
    fen: str,
    username: str,
    color: str,
    speeds: list[str] = Query(default=["blitz", "rapid"]),
):
    try:
        return await explorer.get_player_stats(fen, username, color, speeds)
    except HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
