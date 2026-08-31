"""The narrator.

The LLM's only job is to explain a number it did not produce. The prompt gives
it the projections and the retrieved snippets and tells it, explicitly, that
inventing or adjusting a number is out of bounds -- and the response is checked
for the projected values so a drifting narration is caught rather than shipped.

Falls back to a deterministic template when ANTHROPIC_API_KEY is unset, so the
service runs end-to-end without credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"

SYSTEM = """You explain fantasy football start/sit recommendations.

A trained sequence model produced the projected point totals. They are given to \
you as facts. Your job is to explain the recommendation, not to make one of your \
own.

Rules, in priority order:
1. Never invent, recompute, adjust, or second-guess a projected point total. \
Quote the numbers you are given, exactly as given.
2. Reason only from the projections and the retrieved snippets below. If the \
snippets do not mention a player, say the context is thin rather than \
speculating about their situation.
3. The snippets are the only news you have. Do not add injury, depth-chart, or \
matchup claims from your own knowledge -- you have no way to know whether they \
are current.
4. If a snippet cuts against the projection (an injury designation on the higher \
projected player, say), surface that tension explicitly. The projection still \
stands; the reader deserves to know the risk.

Write 2-4 sentences. Open with the recommendation. Plain prose, no headings, no \
bullet points."""


@dataclass(frozen=True)
class Narration:
    text: str
    model: str
    grounded: bool  # did every quoted projection match what we handed the model?


def _template(a: dict, b: dict, snippets: list[str]) -> str:
    """Deterministic explanation used when no API key is configured."""
    hi, lo = (a, b) if a["projection"] >= b["projection"] else (b, a)
    margin = hi["projection"] - lo["projection"]
    confidence = "clear" if margin >= 3 else "narrow" if margin >= 1 else "effectively a coin flip"
    lines = [
        f"Start {hi['name']} over {lo['name']}. The model projects "
        f"{hi['projection']:.1f} PPR points for {hi['name']} against "
        f"{hi['opponent']}, versus {lo['projection']:.1f} for {lo['name']} "
        f"against {lo['opponent']} -- a {margin:.1f}-point edge, which is "
        f"{confidence}."
    ]
    if snippets:
        lines.append(f"Retrieved context: {snippets[0]}")
    else:
        lines.append("No relevant news was retrieved, so this rests on the projection alone.")
    return " ".join(lines)


def _verify(text: str, players: list[dict]) -> bool:
    """Check the narration quotes our numbers rather than numbers of its own."""
    return all(f"{p['projection']:.1f}" in text for p in players)


def narrate(player_a: dict, player_b: dict, snippets: list[str]) -> Narration:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        text = _template(player_a, player_b, snippets)
        return Narration(text=text, model="template (no ANTHROPIC_API_KEY set)", grounded=True)

    import anthropic

    client = anthropic.Anthropic()
    context = "\n".join(f"- {s}" for s in snippets) or "- (no relevant snippets retrieved)"
    prompt = f"""Compare these two players for the upcoming week.

PROJECTIONS (from the trained model -- treat as fact):
- {player_a['name']} ({player_a['position']}, {player_a['team']}) vs {player_a['opponent']}: {player_a['projection']:.1f} PPR points
- {player_b['name']} ({player_b['position']}, {player_b['team']}) vs {player_b['opponent']}: {player_b['projection']:.1f} PPR points

RETRIEVED NEWS SNIPPETS:
{context}

Which should the manager start, and why?"""

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        # Short, tightly-constrained narration -- medium effort is the right
        # balance here; the reasoning load is in the model, not the prose.
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        log.warning("narrator declined (%s) -- using the template",
                    getattr(response.stop_details, "category", None))
        return Narration(
            text=_template(player_a, player_b, snippets),
            model="template (model declined)",
            grounded=True,
        )

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    grounded = _verify(text, [player_a, player_b])
    if not grounded:
        log.warning("narration did not quote the projected totals verbatim")
    return Narration(text=text, model=response.model, grounded=grounded)
