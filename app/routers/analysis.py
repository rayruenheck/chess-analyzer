from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import evaluation
from app.services.fen import normalize_fen
from app.services.stockfish import stockfish_engine

router = APIRouter(prefix="/analysis", tags=["analysis"])


class FenRequest(BaseModel):
    fen: str
    previous_fen: str | None = None
    depth: int | None = None
    time_limit: float | None = None


@router.post("/evaluate")
async def evaluate(request: FenRequest):
    try:
        fen = normalize_fen(request.fen)
        result, cached = await evaluation.get_or_analyze(fen, request.depth, request.time_limit)

        response = {**result, "cached": cached}

        if request.previous_fen:
            previous_fen = normalize_fen(request.previous_fen)
            response["eval_diff_cp"] = await evaluation.get_or_compute_diff(
                fen, previous_fen, result["depth"], result["score_cp"]
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

    return response


@router.post("/best-move")
async def best_move(request: FenRequest):
    try:
        fen = normalize_fen(request.fen)
        return await stockfish_engine.best_move(fen, request.depth, request.time_limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc
