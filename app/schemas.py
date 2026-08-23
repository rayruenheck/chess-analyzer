from pydantic import BaseModel


class NormalizedGame(BaseModel):
    platform: str
    game_id: str
    url: str
    pgn: str | None = None
    white_username: str | None = None
    white_rating: int | None = None
    black_username: str | None = None
    black_rating: int | None = None
    result: str
    rated: bool | None = None
    speed: str | None = None
    played_at: str | None = None
