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

    # Claude writes the coaching prose. It never evaluates positions -- Stockfish is
    # the source of truth, and the model only explains numbers it is handed.
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5"
    # Per-move blurbs run at low effort; the multi-game report overrides to high.
    llm_effort: str = "low"
    # Generous on purpose. Thinking tokens are drawn from this same budget, so a
    # report at high effort can spend most of it reasoning before it starts writing;
    # too low a ceiling truncates the JSON mid-string and the whole call is wasted.
    llm_max_tokens: int = 32000


settings = Settings()
