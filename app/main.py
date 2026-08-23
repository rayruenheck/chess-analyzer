from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.chesscom import chesscom_client
from app.clients.lichess import lichess_client
from app.clients.lichess_explorer import lichess_explorer_client
from app.db import close_db, init_db
from app.routers import analysis, chesscom, explorer, games, jobs, lichess
from app.services.stockfish import stockfish_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await lichess_client.close()
    await chesscom_client.close()
    await lichess_explorer_client.close()
    await stockfish_engine.close()
    await close_db()


app = FastAPI(title="Chess Analyzer", lifespan=lifespan)

app.include_router(lichess.router)
app.include_router(chesscom.router)
app.include_router(explorer.router)
app.include_router(games.router)
app.include_router(jobs.router)
app.include_router(analysis.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
