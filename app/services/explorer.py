from app.clients.lichess_explorer import lichess_explorer_client
from app.services import explorer_cache
from app.services.fen import normalize_fen

# The explorer only accepts these fixed rating buckets; each covers from its
# own value up to the next one (e.g. 1400 covers [1400, 1600)).
RATING_BUCKETS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]


def nearest_rating_buckets(rating: int) -> list[int]:
    """Picks the bucket pair bracketing `rating`, e.g. 1500 -> [1400, 1600]."""
    lower = RATING_BUCKETS[0]
    for bucket in RATING_BUCKETS:
        if bucket <= rating:
            lower = bucket
        else:
            break

    idx = RATING_BUCKETS.index(lower)
    if idx + 1 < len(RATING_BUCKETS):
        return [lower, RATING_BUCKETS[idx + 1]]
    return [lower]


async def get_masters(fen: str) -> dict:
    fen = normalize_fen(fen)

    cached = await explorer_cache.get_response(fen, source="masters")
    if cached is not None:
        return cached

    response = await lichess_explorer_client.get_masters(fen)
    await explorer_cache.save_response(fen, source="masters", response=response)
    return response


async def get_lichess_stats(fen: str, rating: int, speeds: list[str]) -> dict:
    fen = normalize_fen(fen)
    ratings = nearest_rating_buckets(rating)
    ratings_key = ",".join(str(r) for r in ratings)
    speeds_key = ",".join(speeds)

    cached = await explorer_cache.get_response(
        fen, source="lichess", ratings=ratings_key, speeds=speeds_key
    )
    if cached is not None:
        return cached

    response = await lichess_explorer_client.get_lichess(fen, ratings=ratings, speeds=speeds)
    await explorer_cache.save_response(
        fen, source="lichess", response=response, ratings=ratings_key, speeds=speeds_key
    )
    return response


async def get_player_stats(fen: str, player: str, color: str, speeds: list[str]) -> dict:
    fen = normalize_fen(fen)
    speeds_key = ",".join(speeds)

    cached = await explorer_cache.get_response(
        fen, source="player", player=player, color=color, speeds=speeds_key
    )
    if cached is not None:
        return cached

    response = await lichess_explorer_client.get_player(
        fen, player=player, color=color, speeds=speeds
    )
    await explorer_cache.save_response(
        fen, source="player", response=response, player=player, color=color, speeds=speeds_key
    )
    return response
