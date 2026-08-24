from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import feedback
from app.services.feedback import AnalysisIncomplete
from app.services.llm import LLMUnavailable

router = APIRouter(prefix="/feedback", tags=["feedback"])


class MoveFeedbackRequest(BaseModel):
    fen: str
    move: str
    depth: int | None = None
    rating: int | None = None
    refresh: bool = False


def _unavailable(exc: LLMUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def _upstream(exc: RuntimeError) -> HTTPException:
    """502 for a model-side failure (refusal, truncation, unparseable output).

    These carry an actionable message, so they are surfaced to the caller rather
    than left to become an opaque 500 with a traceback in the server log.
    """
    return HTTPException(status_code=502, detail=str(exc))


@router.get("/games/{job_id}")
async def list_analyzed_games(job_id: str):
    """The games a job analyzed, with the colour the player had. Drives the UI list."""
    return await feedback.list_games(job_id)


@router.get("/game/{game_id}/moves")
async def game_moves(game_id: str, depth: int | None = None):
    """Full move list with evaluations. Pure engine data -- costs nothing to call."""
    try:
        return await feedback.game_moves(game_id, depth)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/move")
async def explain_move(request: MoveFeedbackRequest):
    """Explains one move. Analyzes the position with Stockfish first if needed."""
    try:
        return await feedback.explain_move(
            request.fen, request.move, request.depth, request.rating, request.refresh
        )
    except LLMUnavailable as exc:
        raise _unavailable(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _upstream(exc) from exc


@router.get("/game/{game_id}")
async def explain_game(game_id: str, depth: int | None = None, refresh: bool = False):
    """Reviews one already-analyzed game."""
    try:
        return await feedback.explain_game(game_id, depth, refresh)
    except LLMUnavailable as exc:
        raise _unavailable(exc) from exc
    except AnalysisIncomplete as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _upstream(exc) from exc


@router.get("/openings/{job_id}")
async def list_openings(job_id: str):
    """Opening families in a job with the player's score in each. No model call."""
    return await feedback.list_openings(job_id)


@router.get("/opening/{job_id}")
async def coach_opening(
    job_id: str,
    name: str,
    colour: str | None = None,
    depth: int | None = None,
    refresh: bool = False,
):
    """Coaches one opening: what it aims for, and whether the player got there.

    `name` is an opening family as returned by /feedback/openings/{job_id}.
    """
    try:
        return await feedback.coach_opening(job_id, name, colour, depth, refresh)
    except LLMUnavailable as exc:
        raise _unavailable(exc) from exc
    except AnalysisIncomplete as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _upstream(exc) from exc


@router.get("/report/{job_id}")
async def player_report(
    job_id: str,
    depth: int | None = None,
    refresh: bool = False,
    speed: str | None = None,
):
    """The cross-game coaching report for a finished analysis job.

    `speed` narrows it to one time control (bullet/blitz/rapid/classical/daily).
    """
    try:
        return await feedback.build_report(job_id, depth, refresh, speed)
    except LLMUnavailable as exc:
        raise _unavailable(exc) from exc
    except AnalysisIncomplete as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise _upstream(exc) from exc
