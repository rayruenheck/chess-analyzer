from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.clients.chesscom import chesscom_client
from app.clients.lichess import lichess_client
from app.clients.lichess_explorer import lichess_explorer_client
from app.clients.lichess_tablebase import lichess_tablebase_client
from app.db import close_db, init_db
from app.routers import analysis, chesscom, explorer, feedback, games, jobs, lichess
from app.services import llm
from app.services.stockfish import stockfish_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await lichess_client.close()
    await chesscom_client.close()
    await lichess_explorer_client.close()
    await lichess_tablebase_client.close()
    await stockfish_engine.close()
    await llm.close()
    await close_db()


app = FastAPI(title="Chess Analyzer", lifespan=lifespan)

app.include_router(lichess.router)
app.include_router(chesscom.router)
app.include_router(explorer.router)
app.include_router(games.router)
app.include_router(jobs.router)
app.include_router(analysis.router)
app.include_router(feedback.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")
