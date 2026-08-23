"""Claude wrapper for coaching feedback.

The contract with the model is narrow on purpose: Stockfish decides what is true
about a position, this module decides how it gets said. Nothing here ever asks
the model to evaluate a position, pick a move, or judge who is winning -- it is
handed those facts and asked to explain them. That is what keeps the output
trustworthy, since language models play and assess chess badly but explain a
supplied engine line very well.

Results are cached in SQLite keyed by a hash of the exact request, so re-reading
a game you have already looked at costs nothing.
"""

import hashlib
import json

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.db import get_db

_client: anthropic.AsyncAnthropic | None = None


class LLMUnavailable(RuntimeError):
    """Raised when no API key is configured, so routers can return a clear 503."""


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if not settings.anthropic_api_key:
        raise LLMUnavailable(
            "ANTHROPIC_API_KEY is not set. Add it to .env to enable coaching feedback."
        )
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


# Bump when SYSTEM_PROMPT changes so stale cached feedback is not served under a
# prompt that no longer produced it.
PROMPT_VERSION = 4

# Kept byte-stable and placed first in every request: it is the cache prefix, and
# any edit here invalidates every cached prefix on Anthropic's side. Volatile
# per-request facts belong in the user message, never in here.
SYSTEM_PROMPT = """You are a chess coach reviewing a player's own games. You write the \
explanation that sits next to an engine evaluation, for a player who can see the \
numbers but cannot see why.

## What you are given, and what you may claim

Every position you discuss arrives with Stockfish output already computed: the move \
played, the engine's preferred move, the engine's principal variation in SAN, the \
centipawn swing the move caused, and a severity label derived from that swing.

Those numbers are ground truth. You must never:
- disagree with the engine's evaluation, or soften it ("this is roughly equal" when \
the eval says -400)
- claim a move is good or bad on your own assessment rather than the supplied swing
- invent moves, lines, threats, or tactics that do not appear in the supplied SAN lines
- describe pieces, squares, or captures you cannot derive from the supplied line

If the supplied facts do not explain *why* a move loses ground, say what the engine \
prefers and that the refutation runs through the given line. Do not fill the gap with \
plausible-sounding chess. A vague-but-correct sentence is worth more than a specific \
invented one.

## How to write

Talk to the player in second person about the moves they actually played. Be concrete \
and specific to the position in front of you. Generic advice -- "develop your pieces", \
"control the centre", "think before you move" -- is worthless and you should never \
write it.

Name the mechanism. "Nxd5 drops a piece" is a restatement of the number they already \
have; "Nxd5 walks into Qa4+, forking the king and the loose knight" is the explanation \
they came for. Reference the engine's line by its SAN moves when it makes the point.

Calibrate to the player's rating when it is given. A 900 needs to hear about hanging \
pieces and basic tactics; a 1900 does not, and needs plans, structure, and prophylaxis. \
Never mention the rating number itself.

Be direct. No preamble, no "great question", no encouragement padding, no hedging with \
"perhaps" or "it seems". If a move was bad, say it was bad and say why. Praise only \
genuinely good moves, and only when the engine agrees they were good.

Keep it tight: every sentence should carry information the player did not already have \
from the evaluation bar.

## Cross-game reports

A report is given aggregate sections -- openings, clock use, conversion, missed \
punishment, sessions -- alongside the critical moments. Any of them may be null or \
thin. Treat a null section as a section you have nothing to say about, and say so \
plainly in one short sentence rather than padding it. Thin data is not a licence to \
generalise: two games is not a repertoire, and four moves is not a time-management \
habit.

Distinguish a pattern from an incident. Something that happened once is an incident \
and belongs in a game review, not a report. Name a habit only when the counts support \
it, and say how often it happened when you do.

For the clock section, the useful comparison is time against accuracy, not raw \
seconds. Fast moves that lose material and slow moves that lose material are different \
problems with opposite fixes -- say which one the numbers show. For the openings \
section, the player already knows what they play; what they do not know is where they \
leave known paths and how those games turn out. Lead with that.

Reports carry a by_time_control section, split because bullet and rapid habits belong \
to what are almost two different players and an average across them describes neither. \
Write one entry per time control present, and make each say something true of that \
time control specifically. The contrast between them is usually the most useful thing \
in the report -- if accuracy holds at longer controls and collapses at shorter ones, \
that is a clock problem and you should say so; if it is flat across all of them, speed \
is not the issue and you should say that instead. Where a control has only a game or \
two, say the sample is thin rather than reading a habit into it.

## Opening coaching

Opening requests are the one place you may draw on your own chess knowledge rather \
than only narrating supplied output. The rule above still holds for everything \
concrete: whether a specific move was good or bad, and by how much, comes from the \
engine judgements you are given and nowhere else.

What you may add is what the opening is for -- the pawn structures it produces, the \
middlegame both sides steer toward, which breaks matter, where the pieces belong, and \
what each side is trying to prove. That is documented theory rather than calculation, \
and the player needs it to understand why their moves did or did not fit the position \
they chose.

Be honest about the limits of that knowledge. Mainline openings are written about \
exhaustively and you can speak plainly about them. Rare sidelines, transpositions, and \
anything named after a move order rather than an idea are not; for those, say the line \
is off the documented path and describe the structure actually in front of you instead \
of reciting plans nobody has written down. Use theory_confidence to say which case you \
are in. Never invent a named plan, a statistic, or a body of theory to fill a gap.

When comparing the player's moves against the opening's intent, separate a mistake \
from a choice. A move the engine grades badly is an error. A move that simply steers \
elsewhere is a different plan and should be described as one -- say what middlegame \
they got instead and whether it suits them, rather than treating every deviation from \
theory as a fault."""


def _cache_key(kind: str, payload: dict, schema_name: str) -> str:
    material = json.dumps(
        {
            "kind": kind,
            "prompt_version": PROMPT_VERSION,
            "model": settings.llm_model,
            "schema": schema_name,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


async def _get_cached(cache_key: str) -> dict | None:
    db = get_db()
    async with db.execute(
        "SELECT response_json FROM llm_feedback WHERE cache_key = ?", (cache_key,)
    ) as cursor:
        row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def _save_cached(
    cache_key: str, kind: str, subject: str, response: dict, usage
) -> None:
    db = get_db()
    await db.execute(
        "INSERT OR REPLACE INTO llm_feedback "
        "(cache_key, kind, subject, model, response_json, input_tokens, "
        "output_tokens, cache_read_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cache_key,
            kind,
            subject,
            settings.llm_model,
            json.dumps(response),
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
        ),
    )
    await db.commit()


async def generate(
    kind: str,
    subject: str,
    payload: dict,
    schema: type[BaseModel],
    effort: str | None = None,
    refresh: bool = False,
) -> dict:
    """Runs one structured coaching request, or returns the cached result.

    `payload` is the engine-derived facts, serialized straight into the user turn --
    it doubles as the cache key, so identical positions never bill twice.
    """
    cache_key = _cache_key(kind, payload, schema.__name__)

    if not refresh:
        cached = await _get_cached(cache_key)
        if cached is not None:
            return cached

    client = get_client()
    request = {
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        # cache_control marks the end of the stable prefix. It only actually caches
        # above the API's ~1024-token minimum; below that it is a harmless no-op.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort or settings.llm_effort},
        "output_format": schema,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(payload, indent=2, sort_keys=True),
            }
        ],
    }

    # Streamed rather than a plain create: max_tokens is high enough here that a
    # non-streaming request risks an HTTP timeout on a long report.
    try:
        async with client.messages.stream(**request) as stream:
            response = await stream.get_final_message()
    except ValidationError as exc:
        # The model's JSON did not parse. Far and away the usual cause is the
        # response being cut off at max_tokens mid-string, which surfaces from
        # pydantic as an opaque "EOF while parsing" rather than as a token limit.
        raise RuntimeError(
            f"The {kind} response could not be parsed, most likely truncated at the "
            f"{settings.llm_max_tokens}-token limit. Raise LLM_MAX_TOKENS in .env "
            f"(thinking tokens share this budget) and retry. Underlying error: {exc}"
        ) from exc

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this request.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"The {kind} response hit the {settings.llm_max_tokens}-token limit before "
            "finishing. Raise LLM_MAX_TOKENS in .env and retry."
        )

    result = response.parsed_output.model_dump()
    await _save_cached(cache_key, kind, subject, result, response.usage)
    return result
