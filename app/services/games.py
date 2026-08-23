from datetime import datetime, timezone

from app.clients.chesscom import chesscom_client
from app.clients.lichess import lichess_client
from app.schemas import NormalizedGame


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
