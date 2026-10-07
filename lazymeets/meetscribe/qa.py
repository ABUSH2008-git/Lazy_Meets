"""Ask the meeting: answer questions from the transcript, with timestamps."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Settings
from .llm import ModelClient, est_tokens
from .minutes import format_line
from .refine import load_prompt
from .schemas import QA_SCHEMA, Segment

_STOP = set("a an the and or but if of to in on at for with about is are was were be been do did does what who whom when where why how which "
            "we you i he she they it this that these those our your their my me us them there here any some did can could will would should "
            "meeting said say talk talked discuss discussed".split())


@dataclass
class Answer:
    text: str
    found: bool
    segment_ids: list[int] = field(default_factory=list)


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in _STOP and len(w) > 1]


def select_context(segments: list[Segment], question: str, limit_tokens: int) -> list[Segment]:
    """Whole transcript if it fits, otherwise the segments that share the most words with the question
    (plus their neighbours), kept in time order."""
    if sum(est_tokens(s.text) + 12 for s in segments) <= limit_tokens:
        return segments
    q = Counter(_tokens(question))
    df = Counter(w for s in segments for w in set(_tokens(s.text)))
    n = len(segments)
    scores = []
    for i, s in enumerate(segments):
        toks = Counter(_tokens(s.text))
        sc = sum(min(c, toks[w]) * (1.0 + (n / (1 + df[w])) ** 0.5) for w, c in q.items() if w in toks)
        scores.append((sc, i))
    picked: set[int] = set()
    used = 0
    for sc, i in sorted(scores, reverse=True):
        if sc <= 0 and picked:
            break
        for j in (i - 1, i, i + 1):
            if 0 <= j < n and j not in picked:
                t = est_tokens(segments[j].text) + 12
                if used + t > limit_tokens:
                    break
                picked.add(j)
                used += t
        if used >= limit_tokens * 0.95:
            break
    return [segments[i] for i in sorted(picked)]


def ask(question: str, segments: list[Segment], names: dict[str, str], settings: Settings, client: ModelClient) -> Answer:
    system = load_prompt("qa")
    budget = client.budget_for(settings.minutes_model)
    max_out = 1500
    limit = int(budget * 0.85) - est_tokens(system + question) - max_out - 50
    ctx = select_context(segments, question, max(limit, 600))
    user = "Transcript:\n" + "\n".join(format_line(s, names) for s in ctx) + f"\n\nQuestion: {question.strip()}"
    data = client.chat_json(
        stage="qa",
        model=settings.minutes_model,
        system=system,
        user=user,
        schema_name="meeting_answer",
        schema=QA_SCHEMA,
        max_tokens=max_out,
        reasoning_effort="low",
    )
    valid = {s.id for s in segments}
    ids = []
    for x in data.get("segment_ids") or []:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if v in valid and v not in ids:
            ids.append(v)
    return Answer(text=str(data.get("answer") or "").strip() or "No answer.", found=bool(data.get("found_in_transcript")), segment_ids=ids)
