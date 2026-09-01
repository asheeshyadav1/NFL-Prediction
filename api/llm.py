"""The narrator.

The LLM's only job is to explain a number it did not produce. The prompt gives
it the projections and the retrieved snippets and tells it, explicitly, that
inventing or adjusting a number is out of bounds -- and the response is checked
for the projected values so a drifting narration is caught rather than shipped.

Falls back to a deterministic template when GEMINI_API_KEY is unset, so the
service runs end-to-end without credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Overridable because hosted model names churn faster than this code does.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    return str(getattr(candidates[0], "finish_reason", "no candidates")) if candidates \
        else "no candidates"


def _completed(response) -> bool:
    """True when the model stopped normally rather than being cut off or blocked."""
    if getattr(response, "prompt_feedback", None) and response.prompt_feedback.block_reason:
        return False
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    reason = getattr(candidates[0], "finish_reason", None)
    return reason is None or str(reason).endswith("STOP")


def narrate(player_a: dict, player_b: dict, snippets: list[str]) -> Narration:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        text = _template(player_a, player_b, snippets)
        return Narration(text=text, model="template (no GEMINI_API_KEY set)", grounded=True)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    context = "\n".join(f"- {s}" for s in snippets) or "- (no relevant snippets retrieved)"
    prompt = f"""Compare these two players for the upcoming week.

PROJECTIONS (from the trained model -- treat as fact):
- {player_a['name']} ({player_a['position']}, {player_a['team']}) vs {player_a['opponent']}: {player_a['projection']:.1f} PPR points
- {player_b['name']} ({player_b['position']}, {player_b['team']}) vs {player_b['opponent']}: {player_b['projection']:.1f} PPR points

RETRIEVED NEWS SNIPPETS:
{context}

Which should the manager start, and why?"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=1024,
                # The numbers are fixed before we get here, so the only thing
                # temperature can vary is the prose. Keep it low.
                temperature=0.3,
                # We pass no tools, and leaving this on makes the SDK log an
                # advisory warning on every single request.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except Exception as exc:  # a narration outage must not fail the request
        log.warning("narrator call failed (%s) -- using the template", exc)
        return Narration(
            text=_template(player_a, player_b, snippets),
            model="template (narrator unavailable)",
            grounded=True,
        )

    # A safety block, a recitation stop or an empty candidate all arrive as
    # "no usable text" rather than an exception, so treat them the same way:
    # fall back rather than return an empty explanation next to real numbers.
    text = (response.text or "").strip() if _completed(response) else ""
    if not text:
        log.warning("narrator returned no usable text (%s) -- using the template",
                    _finish_reason(response))
        return Narration(
            text=_template(player_a, player_b, snippets),
            model="template (model declined)",
            grounded=True,
        )

    grounded = _verify(text, [player_a, player_b])
    if not grounded:
        log.warning("narration did not quote the projected totals verbatim")
    return Narration(text=text, model=MODEL, grounded=grounded)
