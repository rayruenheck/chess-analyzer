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
    eco: str | None = None
    opening_name: str | None = None
    opening_ply: int | None = None
    initial_seconds: int | None = None
    increment_seconds: int | None = None


# --- Coaching feedback ---------------------------------------------------------
# These double as the JSON schema handed to the model via output_format, so field
# names and docstrings are part of the prompt: they are what tells the model what
# belongs in each slot. Keep them descriptive.


class MoveFeedback(BaseModel):
    headline: str
    explanation: str
    better_plan: str
    concept: str


class CriticalMoment(BaseModel):
    ply: int
    move_san: str
    what_happened: str
    better: str


class GameFeedback(BaseModel):
    summary: str
    critical_moments: list[CriticalMoment]
    takeaway: str


class ReportTheme(BaseModel):
    title: str
    detail: str
    evidence: list[str]


class TimeControlAdvice(BaseModel):
    time_control: str
    games: int
    # How this time control compares to the player's others -- the contrast is the
    # point, so this is filled even when a single speed dominates the sample.
    verdict: str
    advice: str


class OpeningDivergence(BaseModel):
    # The G-number of the game this happened in, e.g. "G3" -- never a database id.
    game_ref: str
    move: str
    what_you_played: str
    what_the_plan_wants: str
    engine_note: str


class OpeningCoach(BaseModel):
    opening: str
    # How confident the model is that it knows this opening's theory. Mainline
    # openings are well-documented; obscure sidelines are not, and saying so is
    # better than inventing plans for a line nobody has written about.
    theory_confidence: str
    the_idea: str
    typical_middlegame: str
    pawn_structure: str
    your_version: str
    divergences: list[OpeningDivergence]
    focus: str


class ReportClaim(BaseModel):
    """One thing that is true about this player, with the receipts for it.

    Filled before any prose, because fields are generated in the order they are
    declared: deciding what is true and what supports it, and only then writing,
    is what stops a good sentence dragging an unsupported claim along behind it.
    """

    # The chess idea, named -- "trades into endgames without checking the pawn
    # race", not "endgame weakness" and not "7.3% blunder rate".
    concept: str
    # Ids from the facts list, e.g. ["F12", "F31"]. A claim about frequency with
    # no fact id behind it is one you must not make.
    evidence_ids: list[str]
    # Citations of specific positions, e.g. ["[G3#47]"]. May be empty for a claim
    # that rests on aggregates alone.
    moments: list[str]
    # Why it costs points, in one sentence.
    why_it_matters: str
    # The exercise that fixes it. Specific enough to actually perform.
    drill: str


class PlayerReport(BaseModel):
    # Declared first on purpose -- see ReportClaim.
    claims: list[ReportClaim]
    headline: str
    strengths: list[ReportTheme]
    weaknesses: list[ReportTheme]
    # Narrative sections rather than themes: these answer "what do I play" and
    # "where does my time go", which read as prose, not as a scored list. Each must
    # say plainly when the data is too thin to support a claim -- an honest blank
    # beats an invented pattern.
    openings: str
    clock: str
    # One entry per time control present in the data, so a player who plays both
    # bullet and rapid gets advice for each rather than an average of two different
    # sets of habits.
    by_time_control: list[TimeControlAdvice]
    drills: list[str]
