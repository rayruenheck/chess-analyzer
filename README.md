# chess-analyzer

FastAPI backend that pulls game history from Lichess and Chess.com, evaluates
positions with Stockfish, and caches everything in SQLite. Also integrates the
Lichess Opening Explorer (masters DB, community DB by rating band, per-player DB)
to compare a player's moves against common theory/practice at their level.

## Setup

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # already pre-filled with your local Stockfish path
```

Stockfish was installed via `winget install Stockfish.Stockfish` and is referenced
directly by path in `.env` (no PATH restart needed).

## Run

```powershell
venv\Scripts\python -m uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.

## Endpoints

**Raw platform passthroughs**
- `GET /lichess/user/{username}`
- `GET /lichess/user/{username}/games?max_games=20`
- `GET /lichess/cloud-eval?fen=...`
- `GET /chesscom/player/{username}`
- `GET /chesscom/player/{username}/stats`
- `GET /chesscom/player/{username}/games/current`
- `GET /chesscom/player/{username}/games/{year}/{month}`

**Normalized game history** — the entry point for the analyzer flow
- `GET /games/{platform}/{username}?max_games=20` — `platform` is `lichess` or
  `chesscom`. Returns one common shape (`platform`, `game_id`, `pgn`,
  white/black username+rating, `result`, `rated`, `speed`, `played_at`)
  regardless of source.

**Stockfish analysis**
- `POST /analysis/evaluate` — `{"fen": "...", "previous_fen": "...", "depth": 18}`
  (`previous_fen` is optional; when given, returns `eval_diff_cp`, the evaluation
  change caused by the move, from the perspective of whoever made it — positive
  is good for them, negative is a blunder). Analysis is depth-limited by default
  (`STOCKFISH_DEFAULT_DEPTH`, 18) rather than time-limited, because a fixed depth
  is what makes an evaluation deterministic and safely cacheable — pass
  `time_limit` instead only if you specifically want a time-boxed search (those
  results are still cached, but can't benefit from a cache *lookup*, since the
  depth they'll land on isn't known ahead of time).
- `POST /analysis/best-move` — same request shape.

**Bulk analysis jobs** — Stockfish at depth 18 over hundreds of positions takes
minutes, so this doesn't run inline in a request. `POST /jobs/analyze` enqueues
a `BackgroundTasks` job that fetches a player's game history, walks every game
ply by ply, and evaluates+diffs+caches each position.
- `POST /jobs/analyze` — `{"platform": "chesscom", "username": "hikaru", "max_games": 20}`
  → `{"job_id": "..."}`
- `GET /jobs/{job_id}` — `{"status": "queued"|"running"|"done"|"error", "games_total", "games_done", "error"}`

**Lichess Opening Explorer** — what's "normal" at a given level, to compare
against what a player actually did
- `GET /explorer/masters?fen=...` — grandmaster game database
- `GET /explorer/lichess?fen=...&rating=1500&speeds=blitz,rapid` — Lichess's
  community database, filtered to the rating band bracketing `rating` (the
  explorer only accepts fixed buckets — 0/1000/1200/1400/1600/1800/2000/2200/2500 —
  so 1500 resolves to `ratings=1400,1600`). This is Lichess-only data; a
  Chess.com rating of 1500 is not directly comparable and isn't converted.
- `GET /explorer/player?fen=...&username=...&color=white&speeds=blitz` — a
  specific Lichess player's own historical move stats at a position

## Storage model

SQLite at `data/chess_analyzer.db` (auto-created on startup, gitignored). Two
kinds of tables, kept deliberately separate:

- **Position cache** (`evaluations`, `move_evaluations`, `explorer_cache`) — keyed
  by FEN (+ depth, for evaluations; + rating/speed/player filters, for explorer
  data). Stable forever, reusable across every game that transposes into the
  same position. FENs are normalized (halfmove/fullmove counters stripped)
  before being used as a cache key, and evaluations are keyed by `(fen, depth)`
  since evals at different depths aren't comparable and shouldn't be mixed.
- **Per-game facts** (`game_plies`, `jobs`) — one row per ply per game
  (`game_id`, `ply`, `fen_before`, `fen_after`, `move_uci`, `mover`), tied to
  the job that produced it. The centipawn-loss for a given ply is *not*
  duplicated here — it's obtained by joining `game_plies` against
  `move_evaluations` on `(fen_after, fen_before)`.
