from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    stockfish_path: str = "stockfish"
    stockfish_default_depth: int = 18

    db_path: str = "data/chess_analyzer.db"

    lichess_token: str = ""
    lichess_base_url: str = "https://lichess.org"
    lichess_explorer_base_url: str = "https://explorer.lichess.org"
    lichess_user_agent: str = "chess-analyzer-app (contact: rayruenheck@gmail.com)"

    chesscom_base_url: str = "https://api.chess.com/pub"
    chesscom_user_agent: str = "chess-analyzer-app"


settings = Settings()
