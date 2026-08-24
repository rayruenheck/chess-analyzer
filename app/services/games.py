import re
from datetime import datetime, timezone

from app.clients.chesscom import chesscom_client
from app.clients.lichess import lichess_client
from app.schemas import NormalizedGame

# Chess.com states the time control as a raw string: "600" (10 minutes, no
# increment), "600+5" (with increment), or "1/86400" (daily, seconds per move).
DAILY_TIME_CONTROL = re.compile(r"^\d+/(\d+)$")


def _parse_chesscom_time_control(raw: str | None) -> tuple[int | None, int | None]:
    if not raw:
        return None, None

    daily = DAILY_TIME_CONTROL.match(raw)
    if daily:
        # Daily games are per-move allowances, not a depleting bank, so there is no
        # meaningful "increment" and per-move time analysis does not apply.
        return int(daily.group(1)), None

    base, _, increment = raw.partition("+")
    try:
        return int(base), int(increment) if increment else 0
    except ValueError:
        return None, None


def _opening_from_eco_url(url: str | None) -> str | None:
    """Recovers a readable opening name from a chess.com ECOUrl.

    Chess.com sends no opening name field -- only a URL whose last segment encodes
    the name followed by the moves that define the line, e.g.
    ".../Kings-Pawn-Opening-St-George-Defense-2.d4-b5". Everything from the first
    move-number token onward is notation, not name, so it gets cut.
    """
    if not url:
        return None

    slug = url.rstrip("/").rsplit("/", 1)[-1]
    words = []
    for word in slug.split("-"):
        # Move notation can be its own token ("3.e3", "2") or be welded onto the end
        # of the last name token ("Variation...3.e3"), so match a move number
        # anywhere in the word rather than only at the start.
        move_number = re.search(r"\d+\.", word)
        if move_number:
            head = word[: move_number.start()].rstrip(".")
            if head:
                words.append(head)
            break
        if word.isdigit():
            break
        words.append(word)
    return " ".join(words).strip() or None


def _normalize_lichess_game(raw: dict) -> NormalizedGame:
    players = raw.get("players", {})
    white = players.get("white", {})
    black = players.get("black", {})

    winner = raw.get("winner")
    if winner == "white":
        result = "1-0"
    elif winner == "black":
        result = "0-1"
    else:
        result = "1/2-1/2"

    created_at = raw.get("createdAt")
    played_at = (
        datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat()
        if created_at
        else None
    )

    opening = raw.get("opening") or {}
    clock = raw.get("clock") or {}

    return NormalizedGame(
        platform="lichess",
        game_id=raw["id"],
        url=f"https://lichess.org/{raw['id']}",
        pgn=raw.get("pgn"),
        white_username=(white.get("user") or {}).get("name"),
        white_rating=white.get("rating"),
        black_username=(black.get("user") or {}).get("name"),
        black_rating=black.get("rating"),
        result=result,
        rated=raw.get("rated"),
        speed=raw.get("speed"),
        played_at=played_at,
        eco=opening.get("eco"),
        opening_name=opening.get("name"),
        opening_ply=opening.get("ply"),
        initial_seconds=clock.get("initial"),
        increment_seconds=clock.get("increment"),
    )


def _normalize_chesscom_game(raw: dict) -> NormalizedGame:
    white = raw.get("white", {})
    black = raw.get("black", {})

    if white.get("result") == "win":
        result = "1-0"
    elif black.get("result") == "win":
        result = "0-1"
    else:
        result = "1/2-1/2"

    end_time = raw.get("end_time")
    played_at = (
        datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()
        if end_time
        else None
    )

    url = raw.get("url", "")
    eco_url = raw.get("eco")
    initial_seconds, increment_seconds = _parse_chesscom_time_control(raw.get("time_control"))

    return NormalizedGame(
        platform="chesscom",
        game_id=raw.get("uuid") or url.rsplit("/", 1)[-1],
        url=url,
        pgn=raw.get("pgn"),
        white_username=white.get("username"),
        white_rating=white.get("rating"),
        black_username=black.get("username"),
        black_rating=black.get("rating"),
        result=result,
        rated=raw.get("rated"),
        speed=raw.get("time_class"),
        played_at=played_at,
        eco=(re.findall(r'\[ECO "([^"]+)"\]', raw.get("pgn") or "") or [None])[0],
        opening_name=_opening_from_eco_url(eco_url),
        opening_ply=None,
        initial_seconds=initial_seconds,
        increment_seconds=increment_seconds,
    )


async def _fetch_chesscom_raw_games(username: str, max_games: int) -> list[dict]:
    archives = await chesscom_client.get_player_game_archives(username)

    games: list[dict] = []
    for archive_url in reversed(archives):
        year_str, month_str = archive_url.rstrip("/").split("/")[-2:]
        month_data = await chesscom_client.get_player_games_by_month(
            username, int(year_str), int(month_str)
        )
        games.extend(reversed(month_data.get("games", [])))
        if len(games) >= max_games:
            break

    return games[:max_games]


async def get_game_history(platform: str, username: str, max_games: int = 20) -> list[NormalizedGame]:
    if platform == "lichess":
        raw_games = await lichess_client.get_user_games(username, max_games)
        return [_normalize_lichess_game(g) for g in raw_games]

    if platform == "chesscom":
        raw_games = await _fetch_chesscom_raw_games(username, max_games)
        return [_normalize_chesscom_game(g) for g in raw_games]

    raise ValueError(f"Unsupported platform: {platform}")
