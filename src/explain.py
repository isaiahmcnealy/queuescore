"""Natural-language layer over scores, powered by the Anthropic API.

Two capabilities:
  * ``explain_project`` — a plain-English verdict for a single scored project.
  * ``answer_question`` — free-form Q&A over the whole scored leaderboard.

Both are gated by ``config.DRY_RUN`` (defaults on): with DRY_RUN the Anthropic
call is skipped and canned text is returned, so the app runs with no API key and
no network. The real client call is written out below the flag check so wiring
it up day-of is a one-line switch.
"""

from __future__ import annotations

import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Prompt templates (module constants = source of truth for prompt wording)
# --------------------------------------------------------------------------- #
VERDICT_SYSTEM_PROMPT: str = (
    "You are QueueScore, an analyst explaining why an ERCOT interconnection "
    "project is likely or unlikely to reach a signed Interconnection Agreement. "
    "Be concise, specific, and honest about uncertainty. Two or three sentences."
)

VERDICT_USER_TEMPLATE: str = (
    "Project {queue_id}: {generation_type}, {capacity_mw} MW, {county} County, "
    "queued {queue_year}.\n"
    "Model completion probability: {probability:.0%}.\n"
    "Top factors (feature: signed contribution): {attributions}.\n"
    "Explain the verdict for a developer deciding whether to pursue this project."
)

QA_SYSTEM_PROMPT: str = (
    "You are QueueScore, answering questions about a table of scored ERCOT "
    "interconnection-queue projects. Answer only from the provided data. If the "
    "data does not support an answer, say so."
)

QA_USER_TEMPLATE: str = (
    "Here is the scored leaderboard (CSV):\n{leaderboard_csv}\n\n"
    "Question: {question}"
)

RECORD_VERDICT_SYSTEM_PROMPT: str = (
    "You are Project Radar, an origination analyst for Texas power projects. "
    "Given one record from a public source (ERCOT interconnection queue or a "
    "TCEQ air permit), explain in two or three sentences what it signals about "
    "the project's momentum and whether it's worth a business-development "
    "conversation now. Be concrete and honest about uncertainty."
)

RECORD_VERDICT_TEMPLATE: str = (
    "Source: {source}\nRecord ID: {source_id}\nProject: {project_name}\n"
    "Company: {company}\nCounty: {county}\nType: {kind}\n"
    "Status: {status}\nStage signal: {stage_signal}\n"
    "Capacity (MW): {capacity_mw}\nRecord date: {record_date}\n\n"
    "What does this record signal for deal origination?"
)

_CANNED_RECORD_VERDICT: str = (
    "[DRY_RUN] Placeholder read: this record's status and stage signal suggest "
    "an active project worth tracking. Set DRY_RUN=False (and an "
    "ANTHROPIC_API_KEY) for a real origination read from Claude."
)


# Canned responses used when DRY_RUN is on.
_CANNED_VERDICT: str = (
    "[DRY_RUN] This project scores moderately. Size and technology are the main "
    "drivers; queue age slightly lowers the estimate. Enable the Anthropic API "
    "(set DRY_RUN=False and ANTHROPIC_API_KEY) for a real verdict."
)
_CANNED_ANSWER: str = (
    "[DRY_RUN] Placeholder answer. With DRY_RUN off, this returns a real, "
    "data-grounded response from Claude over the scored leaderboard."
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def explain_project(
    queue_id: str,
    generation_type: str,
    capacity_mw: float,
    county: str,
    queue_year: int,
    probability: float,
    attributions: dict[str, float],
) -> str:
    """Return a plain-English verdict for one scored project."""
    if config.DRY_RUN:
        return _CANNED_VERDICT
    prompt = VERDICT_USER_TEMPLATE.format(
        queue_id=queue_id,
        generation_type=generation_type,
        capacity_mw=capacity_mw,
        county=county,
        queue_year=queue_year,
        probability=probability,
        attributions=_format_attributions(attributions),
    )
    return _call_claude(VERDICT_SYSTEM_PROMPT, prompt)


def explain_record(record: dict) -> str:
    """Plain-English origination read on one unified-schema record."""
    if config.DRY_RUN:
        return _CANNED_RECORD_VERDICT
    prompt = RECORD_VERDICT_TEMPLATE.format(
        **{k: record.get(k, "?") for k in (
            "source", "source_id", "project_name", "company", "county",
            "kind", "status", "stage_signal", "capacity_mw", "record_date",
        )}
    )
    return _call_claude(RECORD_VERDICT_SYSTEM_PROMPT, prompt)


def answer_question(question: str, scored: pd.DataFrame) -> str:
    """Answer a free-form question over the scored leaderboard frame."""
    if config.DRY_RUN:
        return _CANNED_ANSWER
    prompt = QA_USER_TEMPLATE.format(
        leaderboard_csv=scored.to_csv(index=False),
        question=question,
    )
    return _call_claude(QA_SYSTEM_PROMPT, prompt)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _format_attributions(attributions: dict[str, float]) -> str:
    ordered = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ", ".join(f"{name} {val:+.3f}" for name, val in ordered)


def _call_claude(system: str, user: str) -> str:
    """Single Anthropic Messages call. Only reached when DRY_RUN is False."""
    import anthropic  # imported lazily so DRY_RUN runs need no dependency/key

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text
