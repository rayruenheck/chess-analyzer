"""Checks the model's prose against the facts it was given.

Prompting reduces invention; it does not end it. Chess is the rare domain where
an exact checker is possible, because every move and every number the coaching is
allowed to mention was supplied in the payload. So after generation the text is
read back and every chess move and every figure in it is looked up. Anything that
does not appear in the payload was invented.

This is deliberately a detector rather than a gate. Regenerating on a violation
doubles the bill for a report, so the default is to surface what failed and let
the caller decide; a single stray number is worth flagging, not paying twice for.
"""

import re

# SAN, loosely: piece letter, optional disambiguation, optional capture, square,
# optional promotion, optional check/mate. Castling handled separately.
_SAN = re.compile(
    r"\b(?:[KQRBN][a-h1-8]?x?[a-h][1-8](?:=[QRBN])?|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?|"
    r"[a-h][1-8](?:=[QRBN])?)[+#]?(?=[^\w]|$)"
)
_CASTLE = re.compile(r"\bO-O(?:-O)?[+#]?\b")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%?")

# Small integers are ordinary prose ("one of three games", "the 4th rank") and
# checking them produces noise rather than signal.
_TRIVIAL_BELOW = 11
# Rounding: a model quoting 7.3 as "about 7" is not inventing anything.
_TOLERANCE = 0.55


def _moves_in(payload) -> set[str]:
    """Every SAN token anywhere in the payload, however deeply nested."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, str):
            for token in _SAN.findall(value) + _CASTLE.findall(value):
                found.add(token.rstrip("+#"))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _numbers_in(payload) -> set[float]:
    found: set[float] = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
        elif isinstance(value, str):
            for raw in _NUMBER.findall(value):
                found.add(float(raw))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _prose(feedback) -> str:
    """Every string the player will actually read."""
    parts: list[str] = []

    def walk(value):
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(feedback)
    return "\n".join(parts)


def check(feedback, payload) -> dict:
    """Finds moves and numbers in `feedback` that are not in `payload`."""
    text = _prose(feedback)
    known_moves = _moves_in(payload)
    known_numbers = _numbers_in(payload)

    invented_moves = sorted(
        {
            token.rstrip("+#")
            for token in _SAN.findall(text) + _CASTLE.findall(text)
            if token.rstrip("+#") not in known_moves
        }
    )

    invented_numbers = []
    for raw in _NUMBER.findall(text):
        value = float(raw)
        if value < _TRIVIAL_BELOW and value == int(value):
            continue
        if not any(abs(value - known) <= _TOLERANCE for known in known_numbers):
            invented_numbers.append(value)

    return {
        "ok": not invented_moves and not invented_numbers,
        "invented_moves": invented_moves,
        "invented_numbers": sorted(set(invented_numbers)),
    }
