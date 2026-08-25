# chess-analyzer

FastAPI app that pulls game history from Lichess and Chess.com, evaluates positions
with Stockfish, and turns the result into written coaching feedback from Claude.
Everything is cached in SQLite. Also integrates the Lichess Opening Explorer
(masters DB, community DB by rating band, per-player DB) to compare a player's moves
against common theory/practice at their level.

Open `http://127.0.0.1:8000/` for the UI: enter a username, watch the analysis run,
then step through any game with an eval bar, a colour-coded move list, and a written
review of what went wrong.

## Setup

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # already pre-filled with your local Stockfish path
```

`.env` needs two keys filled in to unlock everything: `ANTHROPIC_API_KEY` for the
coaching feedback, and `LICHESS_TOKEN` (any token, no scopes) for the opening
explorer. The app runs without either — analysis, board, evals and move list all
work — and the affected routes say why they are unavailable.

Stockfish was installed via `winget install Stockfish.Stockfish` and is referenced
directly by path in `.env` (no PATH restart needed).

## Run

```powershell
venv\Scripts\python -m uvicorn app.main:app
```

Drop `--reload` for real analysis runs. Jobs run as FastAPI `BackgroundTasks`, so
they die with the process: any reload cancels an in-flight job, which the job then
records as a cancellation rather than a crash.

Then open http://127.0.0.1:8000/ for the app, or /docs for interactive Swagger UI.

Coaching feedback needs `ANTHROPIC_API_KEY` in `.env`. Without it everything else
still works -- analysis, board, evals, move list -- and `/feedback/*` returns 503.

## Tests

```powershell
pip install -r requirements-dev.txt
venv\Scripts\python -m pytest
```

117 tests, ~7 seconds, and safe to run at any time: **no test reaches the network or
touches `data/chess_analyzer.db`.** Every model call is stubbed (they cost money),
every platform and explorer call is stubbed (rate limits and flakiness), Stockfish is
never launched, and each test gets a fresh SQLite file under `tmp_path`. The `client`
fixture deliberately runs with no API key configured, so an endpoint that reaches the
model without being stubbed fails as a recognisable 503 rather than quietly billing.

| File | Covers |
| --- | --- |
| `tests/conftest.py` | Fixtures: temp DB, app client, stubbed model/explorer, and `seed()`, which writes a finished job straight into SQLite |
| `tests/test_endpoints.py` | Every route: shapes, status codes, the guards, and what each endpoint actually sends the model |
| `tests/test_analysis.py` | The deterministic layers — severity, phase, timing, openings, conversion, platform parsing |
| `tests/test_llm.py` | Caching, cache-key invalidation, and turning model failures into actionable errors |
| `tests/test_citations.py` | Game numbering, and resolving the model's `[G3#47]` tokens into links that cannot point at the wrong position |

`seed()` writes rows rather than running the pipeline, which keeps the endpoint tests
off Stockfish and lets them assert on states a successful run never produces — a
half-analyzed game, a job that errored mid-way.

Several tests exist because the behaviour they pin down was wrong at some point and
produced confidently wrong coaching. Those are worth keeping honest:

- a move matching the engine's own choice is never graded a mistake
- mate scores are clamped so one mate-in-N cannot swamp an average
- correspondence games stay out of clock statistics
- castling matches the explorer despite `e1g1` vs `e1h1`
- a cached answer is never bought twice, and bumping `PROMPT_VERSION` retires text
  written under the old rubric
- no endpoint shows the model a raw game id, and a citation it invents degrades to
  plain text instead of becoming a link to the wrong board position

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

**Coaching feedback** -- Claude explains what the engine found

Stockfish is the source of truth and Claude is the narrator: the model is never
asked to evaluate a position or pick a move, only to explain engine output it is
handed. `app/services/classify.py` decides what is true about a move (severity band,
phase, engine agreement, mate facts, the principal variation in SAN) and
`app/services/llm.py` decides how it gets said. That split is what keeps the prose
from inventing chess.

All three tiers are on demand -- the analysis job itself never calls the model, so
you only pay for feedback on games you actually open. Every response is cached in
`llm_feedback` keyed by a hash of the exact request, so re-reads are free. Pass
`refresh=true` to regenerate.

*What the model is sent, and why.* Two rules shape the payload, both learned from
getting them wrong on real data.

**Selection is ranked by expected score, not centipawns.** `MAX_REPORTED_CP_LOSS`
clamps every mate-allowing move to the same 1000, which is correct for averaging and
ruinous for sorting: on a 100-game job, 48 moves sat on the ceiling competing for 25
places, so the critical moments were an arbitrary slice of one tie and no move losing
less than the cap could ever be selected, however instructive. Removing the clamp
would only make the failure deterministic instead of random — unclamped, those moves
score ~99000 and still take every slot. Centipawns measure the *position*; selection
has to measure what the mistake *cost*, so `_critical` ranks on `win_prob_lost`.
Eight of the old top 25 came from positions already lost by 500+ centipawns (one from
−8032); the new selection contains none. The severity bands and the headline ACPL
still use centipawns, which is the right unit for both and keeps ACPL comparable to
other sites — `average_centipawn_loss_competitive` is reported alongside it.

The win-probability curve is a logistic on centipawns, the same one the eval bar
draws, at `WIN_PROB_SCALE = 320`. Lichess publishes 1/271.6, fitted on 2300-rated
games; checked against this app's own games that curve badly under-rates losing
positions at club level, where a −5 is still very much alive. A flatter curve is the
safer error, since the alternative is telling a player that throwing away a bad
position cost them nothing.

**Frequency claims come from rates, never from examples.** `insights.pattern_rates`
sends blunder rates by phase, by clock remaining and by whether the position was
already won or lost, plus, per move type, its share of blunders against its share of
all moves. Without that denominator a model asked to name recurring habits will find
them whether they exist or not: captures were 21% of one player's blunders, which
reads damning until you know they are 24% of every move they play. Each move type
carries a `verdict` decided server-side, so the coaching text and the UI cannot
disagree about what counts as a weakness — and the band is deliberately not a hard
1.0 cut, since "was in check" at 1.18× is a 6.2% blunder rate against 5.2% and
naming that is how a report fills with patterns that are not there. Critical moments
were cut from 25 to 12 in the same change: a dozen examples plus real rates beats a
long tail of ties, and costs fewer tokens.

The report tab renders all of this — both accuracy figures, and the blunder-rate
breakdowns as bars on one shared scale with a tick at the player's overall rate, so
each bucket reads as above or below their own average.

*Citations.* The model never sees a database game id. `app/services/citations.py`
hands it games numbered `G1..Gn` -- the same numbers the sidebar shows -- and asks it
to cite claims inline as `[G3#47]`: game three, ply forty-seven. Coming back, those
tokens are resolved server-side against the move list into `[[c0]]` markers plus a
`citations` array carrying the game id, the ply and the move in SAN, which the UI
renders as links that open that game at that position. Resolution is server-side
because only the server has the move list: the token says ply 47, but the reader
needs to see "24.Qxh7". A citation naming a game that does not exist, or a ply that
was never analyzed, degrades to plain text or to a game-level link rather than
becoming a link to the wrong position. Numbering is assigned over the whole job
before any filtering, so a report narrowed to one time control still agrees with the
sidebar.

- `GET /feedback/games/{job_id}` -- games a job analyzed, with the colour the player
  had, its `number`, and its opening family. No model call.
- `GET /feedback/game/{game_id}/moves` -- every ply with its evaluation, severity and
  engine line, for stepping through a board. No model call, so the UI renders before
  spending anything.
- `POST /feedback/move` -- `{"fen": "...", "move": "g1f3", "rating": 1500}`. Explains
  one move, analyzing the position with Stockfish first if it is not already cached.
- `GET /feedback/game/{game_id}` -- reviews one game: aggregate stats plus an
  explanation of each of its worst 8 moments.
- `GET /feedback/openings/{job_id}` -- opening families in a job with the player's
  score in each. No model call. Games are grouped by *family* rather than by the exact
  variation name: "Caro-Kann Defense: Exchange" and "Caro-Kann Defense: Advance" are
  one opening to coach, and variation names are granular enough that a twenty-game
  sample would otherwise yield twenty buckets of one.
- `GET /feedback/opening/{job_id}?name=English+Opening&colour=white` -- coaches one
  opening: what it aims for, what middlegame it produces, and whether the player got
  there. Runs at `high` effort.

  **This is the one place the model draws on its own chess knowledge**, and the
  exception is deliberate. Evaluating a position is calculation, which language models
  do badly and which is why every other tier restricts them to narrating Stockfish.
  "What middlegame does the King's Indian aim for" is encyclopedic -- pawn structures,
  standard breaks, where the pieces belong -- and heavily documented, which they handle
  reliably. There is also nowhere to pull it from: ECO classification is names and move
  orders only, and no free structured database of middlegame plans exists.

  The boundary is enforced in the prompt: plans and structures may come from the model,
  every concrete claim about whether a move was good still comes from the engine, and
  `theory_confidence` makes it state whether the line is mainline or an obscure sideline
  it should not recite theory for. It is also told to separate a mistake from a choice --
  a move that steers somewhere else is a different plan, not automatically a fault.
- `GET /feedback/report/{job_id}?speed=blitz` -- the cross-game report, and the reason
  this app exists. A single game's blunder is noise; the same blunder in eight games
  out of twenty is a habit, and only this tier can see that. Runs at `high` effort.
  `speed` narrows it to one time control, which is worth doing rather than averaging:
  a player's bullet and rapid habits are close to two different players, and pooling
  them hides both.

The report is assembled from deterministic sections computed in
`app/services/insights.py` -- the model is handed finished numbers, never asked to
find the pattern itself. Every section returns null when the data cannot support a
claim, and the prompt treats a null section as one to stay silent about:

- **Openings** -- repertoire by colour with score, from the ECO code and name the
  platforms already send, grouped by opening family. Lines seen fewer than twice are
  excluded; one outing is not a repertoire.
- **By time control** -- every aggregate above, split per speed. Split rather than
  averaged because bullet and rapid habits belong to what are nearly two different
  players; kept in one request rather than one report per speed so the model can
  contrast them, which is usually the most useful finding available.
- **Book exit** -- the first move where the player leaves what their rating band
  actually plays, via the Lichess Opening Explorer. "You play the Caro-Kann" is
  something they know; "you leave known paths on move 6, where most play Nc3" is not.
  Measured by the **absolute number of games** reaching the position a move creates,
  not by that move's share of the parent position. Share measures popularity rather
  than theory: 1.c4 is chosen by under 5% of players and is the entirely mainline
  English Opening, so a share threshold reports "left book on move 1", which is both
  wrong and useless as coaching. Two cases are excluded deliberately -- a position
  whose *most popular* continuation is also below the threshold means theory ran out
  for everybody rather than the player departing from it, and moves are matched on
  SAN as well as UCI because Lichess encodes castling as king-takes-rook (`e1h1`)
  while python-chess writes the king's destination (`e1g1`).
- **Clock** -- time against accuracy, not raw seconds. Fast blunders and slow blunders
  are different problems with opposite fixes, so snap moves (<5s) and considered moves
  are reported separately, along with time-trouble accuracy.
- **Conversion** -- how often a winning position became a win. The biggest leak below
  2000, and computable entirely from evaluations already stored.
- **Missed punishment** -- opponent errors that handed over a winning position which
  then evaporated on the very next move.
- **Sessions** -- accuracy across a sitting and after a loss, from `played_at`.

Cost control is in the selection, not the model: a 40-move game is 80 plies, and
sending all of them would be both expensive and unreadable. Only the player's own
moves ranked by centipawn loss are sent (8 per game, 25 per report); everything else
is compressed into aggregate counts. A report over 20 games is roughly 6K input
tokens.

**Lichess Opening Explorer** — what's "normal" at a given level, to compare
against what a player actually did
> **The explorer requires `LICHESS_TOKEN`.** It answers 401 to anonymous requests
> (a bogus path on the same host still returns 404, so nginx gates the route rather
> than blocking the caller). Any token works and **no OAuth scopes are needed** —
> create one at https://lichess.org/account/oauth/token. Only public data is read, so
> granting scopes would add risk for no benefit. Without the token these routes fail
> and the report's book-exit section reports itself unavailable rather than claiming
> the player never leaves theory. Requests are throttled to one every 0.7s; a report
> over twenty games issues dozens of lookups and will otherwise hit a 429 partway
> through.

- `GET /explorer/masters?fen=...` — grandmaster game database
- `GET /explorer/lichess?fen=...&rating=1500&speeds=blitz,rapid` — Lichess's
  community database, filtered to the rating band bracketing `rating` (the
  explorer only accepts fixed buckets — 0/1000/1200/1400/1600/1800/2000/2200/2500 —
  so 1500 resolves to `ratings=1400,1600`). This is Lichess-only data; a
  Chess.com rating of 1500 is not directly comparable and isn't converted.
- `GET /explorer/player?fen=...&username=...&color=white&speeds=blitz` — a
  specific Lichess player's own historical move stats at a position

## Storage model

SQLite at `data/chess_analyzer.db` (auto-created on startup, gitignored). Three
kinds of tables, kept deliberately separate:

- **Position cache** (`evaluations`, `move_evaluations`, `explorer_cache`) — keyed
  by FEN (+ depth, for evaluations; + rating/speed/player filters, for explorer
  data). Stable forever, reusable across every game that transposes into the
  same position. FENs are normalized (halfmove/fullmove counters stripped)
  before being used as a cache key, and evaluations are keyed by `(fen, depth)`
  since evals at different depths aren't comparable and shouldn't be mixed.
- **Coaching cache** (`llm_feedback`) -- keyed by a SHA-256 of the model, prompt
  version and exact request payload. Bumping `PROMPT_VERSION` in `llm.py` retires
  every cached response generated under the old system prompt.
- **Per-game facts** (`games`, `game_plies`, `jobs`) — one row per ply per game
  (`game_id`, `ply`, `fen_before`, `fen_after`, `move_uci`, `move_san`, `mover`),
  tied to the job that produced it, plus one `games` row per game recording which
  colour the analyzed player had. That last part is load-bearing: `mover` records
  that *a* side moved, not whether it was the player being coached, so without
  `games.user_color` no feedback above the single-move tier can tell their mistakes
  from their opponent's. The centipawn-loss for a given ply is *not*
  duplicated here — it's obtained by joining `game_plies` against
  `move_evaluations` on `(fen_after, fen_before)`. Each ply also stores
  `clock_after_seconds` and the `seconds_spent` derived from it: a clock reading is
  what the mover had left *after* moving with the increment already credited, so time
  burned is their previous reading plus the increment earned minus what they have now.
  Correspondence games are stored but excluded from clock analysis, since their
  readings are calendar time rather than thinking time.

## The board

`app/static/index.html` is the whole UI: one file, no CDN, no build step, no
dependencies. The board is drawn from the FEN with CSS grid, so it works with the
network down.

The pieces are the one imported asset. They are the Cburnett/Wikimedia SVG set --
the same artwork Wikipedia and Lichess use -- as a single sprite in
`app/static/pieces.svg`, taken from the `cm-chessboard` package. They replaced
unicode glyphs, which had no white pieces of their own: `♟` recoloured white with a
text-shadow, which rendered thin and differed by system font. The sprite is injected
into the document once and each square is a `<use>` reference; an external
`<use href="file.svg#id">` would have been tidier but is subject to CORS and fails
from a `file://` origin, and the unicode glyphs remain as a fallback if the fetch
fails.

> Piece artwork © [Cburnett](https://en.wikipedia.org/wiki/User:Cburnett) and
> [Rfc1394](https://en.wikipedia.org/wiki/User:Rfc1394), from
> [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces/Standard),
> licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), modified
> by shaack for cm-chessboard. Only this artwork is imported -- no board library.

Deliberately *not* imported: `chessground` (Lichess's board) is GPL-3.0 and would
relicense the project, and a full board library would mean re-plumbing the eval bar,
seats and citation links to buy a drag-and-drop move input this replay viewer has no
use for.

The board also draws the engine's choice **for the position currently shown** as an
arrow -- which is the *next* ply's best move, since that ply was searched from this
exact position -- and blooms the king's square on check. The eval bar's fill is
always White's share; which end it grows from follows the board's orientation.
