"""Numbered game references, and the inline citations the model writes with them.

The model is never shown a database game id. A 32-character platform id is
meaningless to the player reading the report and impossible to turn into a link,
so games are handed over as G1, G2, ... -- numbered by their position in the same
list the sidebar renders, which is what makes "Game 3" in the prose and "#3" in
the sidebar the same game.

The model cites them inline as `[G3#47]`: game three, ply forty-seven. This module
owns both ends of that contract -- building the index the model is given, and
resolving the tokens it writes back into a `{game_id, ply, san}` the UI can turn
into a link that opens the game at that position.

Resolution happens on the server rather than in the browser because only the
server has the move list. The token says ply 47; the reader needs to see
"24.Qxh7", and the browser would have to fetch every cited game to find out.
"""

import re

# `[G3#47]`, `[G3]`, or `[#47]` -- the last meaning "this game" when a single game
# is under review. Whitespace is tolerated because the model sometimes adds it;
# anything else inside the brackets fails to match and is left as literal text
# rather than becoming a broken link.
CITATION_RE = re.compile(r"\[\s*(?:G(\d+))?\s*(?:#\s*(\d+))?\s*\]")

# Cleans up after a dropped citation: the space that preceded it is left stranded,
# sometimes in front of the sentence's full stop.
_SPACE_BEFORE_PUNCTUATION = re.compile(r" +(?=[.,;:!?)])")
_REPEATED_SPACE = re.compile(r"  +")

_OUTCOME = {1.0: "won", 0.0: "lost", 0.5: "drew"}


def _describe(game: dict) -> str:
    """One line identifying a game to the model -- and to a reader on hover."""
    parts = [f"{_OUTCOME.get(game.get('user_score'), 'unfinished')} as {game.get('user_color') or '?'}"]
    if game.get("opponent"):
        rating = game.get("opponent_rating")
        parts.append(f"vs {game['opponent']}" + (f" ({rating})" if rating else ""))
    if game.get("speed"):
        parts.append(game["speed"])
    if game.get("opening_name"):
        parts.append(game["opening_name"])
    if game.get("played_at"):
        parts.append(str(game["played_at"])[:10])
    return " · ".join(parts)


def _move_text(info: dict) -> str:
    return f"{info['move_number']}{'.' if info['side'] == 'white' else '...'}{info['san']}"


class GameIndex:
    """Maps game ids to G-numbers, and resolves the citations written with them."""

    def __init__(self, games: list[dict]) -> None:
        self._by_id: dict[str, dict] = {}
        self._by_number: dict[int, dict] = {}
        # {game_id: {ply: {"san", "move_number", "side"}}}, filled by record_plies.
        self._plies: dict[str, dict[int, dict]] = {}

        for position, game in enumerate(games, start=1):
            # list_games already numbers the job; trust it so the sidebar row and
            # the citation cannot disagree about which game is "Game 3".
            number = game.get("number", position)
            entry = {
                "ref": f"G{number}",
                "number": number,
                "game_id": game["game_id"],
                "label": f"Game {number}",
                "description": _describe(game),
                "url": game.get("url"),
            }
            self._by_id[game["game_id"]] = entry
            self._by_number[number] = entry

    def record_plies(self, game_id: str, annotated: list[dict]) -> None:
        """Registers a game's moves so `[G3#47]` can name the move at ply 47.

        Both sides are recorded, not just the player's: the model cites opponent
        blunders too when explaining what it failed to punish.
        """
        self._plies[game_id] = {
            move["ply"]: {
                "san": move["played_san"],
                "move_number": move["move_number"],
                "side": move["side_to_move"],
            }
            for move in annotated
            if move.get("ply") is not None
        }

    def scope(self, games: list[dict]) -> list[dict]:
        """The index entries for a subset of games, in the order given."""
        return [self._by_id[g["game_id"]] for g in games if g["game_id"] in self._by_id]

    def payload_index(self, games: list[dict]) -> list[dict]:
        """What the model is shown: a ref and a description, never an id."""
        return [
            {"ref": entry["ref"], "game": entry["description"]}
            for entry in self.scope(games)
        ]

    def swap_ids(self, value):
        """Recursively rewrites every `game_id` key to the `game_ref` it maps to.

        Applied to the whole payload rather than at each site that produces one,
        because game ids surface from a dozen different aggregates -- squandered
        wins, slowest moves, book exits, missed punishment -- and one of them
        leaking a raw id is enough for the model to quote it back.
        """
        if isinstance(value, dict):
            swapped = {}
            for key, item in value.items():
                if key == "game_id" and isinstance(item, str):
                    entry = self._by_id.get(item)
                    swapped["game_ref"] = entry["ref"] if entry else item
                else:
                    swapped[key] = self.swap_ids(item)
            return swapped
        if isinstance(value, list):
            return [self.swap_ids(item) for item in value]
        return value

    def resolve(self, feedback, default_game_id: str | None = None):
        """Rewrites `[G3#47]` tokens to `[[c0]]` markers and returns the citations.

        `default_game_id` supplies the game for the bare `[#47]` form used when a
        single game is under review.

        A token naming a game that does not exist, or a ply that was never
        analyzed, degrades rather than failing: an unknown ply cites the game
        alone, and an unknown game becomes plain text. The model occasionally
        miscounts, and a wrong link is worse than no link.
        """
        citations: list[dict] = []
        seen: dict[tuple, str] = {}

        def cite(number: int | None, ply: int | None) -> str | None:
            entry = self._by_number.get(number) if number else self._by_id.get(default_game_id)
            if entry is None:
                return None

            info = self._plies.get(entry["game_id"], {}).get(ply) if ply else None
            if info is None:
                ply = None

            # A bare [#47] is written inside a single game's review, where naming
            # the game again in every citation would be noise. If its ply did not
            # resolve there is nothing left to link to -- the reader is already on
            # that game -- so it degrades to no citation at all.
            bare = number is None
            if bare and ply is None:
                return None

            key = (entry["game_id"], ply, bare)
            if key in seen:
                return seen[key]

            move = _move_text(info) if info else None
            if bare and move:
                text = move
            elif move:
                text = f"{entry['label']} · {move}"
            else:
                text = entry["label"]

            identifier = f"c{len(citations)}"
            citations.append(
                {
                    "id": identifier,
                    "text": text,
                    "game_id": entry["game_id"],
                    "game_number": entry["number"],
                    "game_label": entry["label"],
                    "description": entry["description"],
                    "url": entry["url"],
                    "ply": ply,
                    "move_number": info["move_number"] if info else None,
                    "san": info["san"] if info else None,
                }
            )
            seen[key] = identifier
            return identifier

        def replace(match: re.Match) -> str:
            raw_game, raw_ply = match.group(1), match.group(2)
            if raw_game is None and raw_ply is None:
                return match.group(0)

            number = int(raw_game) if raw_game else None
            identifier = cite(number, int(raw_ply) if raw_ply else None)
            if identifier:
                return f"[[{identifier}]]"
            # Unresolvable: drop the brackets so the sentence still reads.
            return f"Game {number}" if number else ""

        def walk(value):
            if isinstance(value, str):
                rewritten = CITATION_RE.sub(replace, value)
                if rewritten == value:
                    return value
                rewritten = _SPACE_BEFORE_PUNCTUATION.sub("", rewritten)
                return _REPEATED_SPACE.sub(" ", rewritten).strip()
            if isinstance(value, dict):
                return {key: walk(item) for key, item in value.items()}
            if isinstance(value, list):
                return [walk(item) for item in value]
            return value

        return walk(feedback), citations
