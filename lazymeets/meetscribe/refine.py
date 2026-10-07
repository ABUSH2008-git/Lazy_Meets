"""Stage 2: domain-aware transcript refinement (language model #1).

The model never rewrites the transcript. It returns a list of small corrections
(segment id, original span, replacement). Code then:
  1. finds each span in the right segment,
  2. runs the meaning guards (numbers, negation, commitments, names, sound-alike check),
  3. applies the safe ones and keeps the blocked ones with a reason.
Timestamps and segment boundaries never change, so raw and refined line up 1:1.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from rapidfuzz import fuzz

from .config import PROMPTS_DIR, Settings
from .errors import ModelOutputError, RequestTooLargeError
from .guards import check_correction
from .llm import ModelClient, est_tokens
from .schemas import REFINE_SCHEMA, Correction, MeetingContext, RefinedTranscript, Segment, Transcript

Notify = Callable[[str], None]


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def context_block(ctx: MeetingContext) -> str:
    lines = []
    if ctx.title:
        lines.append(f"Meeting title: {ctx.title}")
    if ctx.description:
        lines.append(f"About this meeting: {ctx.description}")
    if ctx.participants:
        lines.append("Participants (correct spellings): " + ", ".join(ctx.participants))
    terms = ctx.all_terms()
    if terms:
        lines.append("Glossary - terms likely to come up (preferred spellings): " + ", ".join(terms))
    return "\n".join(lines) if lines else "No extra context was given. Infer the domain from the transcript."


def _line(s: Segment, context_only: bool = False) -> str:
    return f"[{s.id}]{' (context only)' if context_only else ''} {s.text}"


def plan_batches(segments: list[Segment], limit_tokens: int) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    cur: list[Segment] = []
    used = 0
    for s in segments:
        t = est_tokens(_line(s))
        if cur and used + t > limit_tokens:
            batches.append(cur)
            cur, used = [], 0
        cur.append(s)
        used += t
    if cur:
        batches.append(cur)
    return batches


def _parse(data: dict) -> list[Correction]:
    out = []
    for item in data.get("corrections") or []:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("segment_id"))
        except (TypeError, ValueError):
            continue
        o = str(item.get("original") or "")
        c = str(item.get("corrected") or "")
        if not o.strip() or o == c:
            continue
        out.append(
            Correction(
                segment_id=sid,
                original=o,
                corrected=c,
                category=str(item.get("category") or "terminology"),
                reason=str(item.get("reason") or ""),
            )
        )
    return out


def _ask(batch: list[Segment], before: list[Segment], ctx_text: str, system: str, settings: Settings,
         client: ModelClient, max_out: int, part: str) -> list[Correction]:
    lines = [_line(s, True) for s in before] + [_line(s) for s in batch]
    user = f"{ctx_text}\n\nTranscript segments{part}:\n" + "\n".join(lines)
    try:
        data = client.chat_json(
            stage="refine",
            model=settings.refine_model,
            system=system,
            user=user,
            schema_name="transcript_corrections",
            schema=REFINE_SCHEMA,
            max_tokens=max_out,
            reasoning_effort=settings.refine_reasoning,
        )
    except (RequestTooLargeError, ModelOutputError):
        if len(batch) < 2:
            raise
        half = len(batch) // 2
        return _ask(batch[:half], before, ctx_text, system, settings, client, max_out, part) + _ask(
            batch[half:], batch[max(0, half - 2): half], ctx_text, system, settings, client, max_out, part
        )
    allowed = {s.id for s in batch}
    return [c for c in _parse(data) if c.segment_id in allowed or c.segment_id not in {s.id for s in before}]


# ---------------------------------------------------------------- applying corrections


def _pattern(span: str, flags: int = 0) -> re.Pattern:
    left = r"(?<!\w)" if span[:1].isalnum() else ""
    right = r"(?!\w)" if span[-1:].isalnum() else ""
    return re.compile(left + re.escape(span) + right, flags)


def _find_span(text: str, original: str) -> Optional[str]:
    """The exact text in the segment that `original` refers to (exact, then case-insensitive, then fuzzy)."""
    o = original.strip()
    if not o:
        return None
    m = _pattern(o).search(text)
    if m:
        return m.group(0)
    m = _pattern(o, re.I).search(text)
    if m:
        return m.group(0)
    if len(o) >= 4:
        al = fuzz.partial_ratio_alignment(o.lower(), text.lower())
        if al and al.score >= 90:
            a, b = al.dest_start, al.dest_end
            while a > 0 and text[a - 1].isalnum():
                a -= 1
            while b < len(text) and text[b].isalnum():
                b += 1
            cand = text[a:b]
            if cand and fuzz.ratio(cand.lower(), o.lower()) >= 85:
                return cand
    return None


def apply_corrections(segments: list[Segment], proposals: list[Correction], ctx: MeetingContext) -> tuple[list[Segment], list[Correction]]:
    terms = ctx.all_terms()
    people = list(ctx.participants)
    out = [s.model_copy(deep=True) for s in segments]
    by_id = {s.id: s for s in out}
    results: list[Correction] = []
    seen: set[tuple] = set()

    for c in proposals:
        key = (c.segment_id, c.original.strip().lower(), c.corrected.strip())
        if key in seen:
            continue
        seen.add(key)
        seg = by_id.get(c.segment_id)
        span = _find_span(seg.text, c.original) if seg else None
        if span is None:
            # model may have cited the wrong segment id; accept it only if the span is unique elsewhere
            hits = [s for s in out if _find_span(s.text, c.original)]
            if len(hits) == 1:
                seg = hits[0]
                c = c.model_copy(update={"segment_id": seg.id})
                span = _find_span(seg.text, c.original)
        if seg is None or span is None:
            results.append(c.model_copy(update={"status": "not_found", "block_reason": "text not found in that segment"}))
            continue
        c = c.model_copy(update={"original": span})
        if span == c.corrected:
            continue
        reason = check_correction(span, c.corrected, c.category, terms, people)
        if reason:
            results.append(c.model_copy(update={"status": "blocked", "block_reason": reason}))
            continue
        seg.text = _pattern(span).sub(lambda _m: c.corrected, seg.text)
        results.append(c.model_copy(update={"status": "applied"}))

    # A multi-word mishearing fixed in one place is almost surely the same mishearing elsewhere.
    for c in [r for r in results if r.status == "applied" and len(r.original.split()) >= 2]:
        pat = _pattern(c.original, re.I)
        for s in out:
            if s.id != c.segment_id and pat.search(s.text):
                s.text = pat.sub(lambda _m: c.corrected, s.text)
                results.append(c.model_copy(update={"segment_id": s.id, "status": "propagated", "reason": "same mishearing as in segment " + str(c.segment_id)}))

    results.sort(key=lambda r: (r.segment_id, r.status != "applied"))
    return out, results


def refine(transcript: Transcript, ctx: MeetingContext, settings: Settings, client: ModelClient, notify: Notify) -> RefinedTranscript:
    system = load_prompt("refine")
    ctx_text = context_block(ctx)
    budget = client.budget_for(settings.refine_model)
    max_out = 3500 if budget < 20000 else 8000
    overhead = est_tokens(system + ctx_text) + 80
    limit = max(400, int(budget * 0.88) - overhead - max_out)
    if budget >= 20000:
        limit = min(limit, 6000)  # smaller batches keep the model attentive
    batches = plan_batches(transcript.segments, limit)
    proposals: list[Correction] = []
    prev: list[Segment] = []
    for i, batch in enumerate(batches, start=1):
        part = f" (part {i} of {len(batches)})" if len(batches) > 1 else ""
        notify(f"Refining terminology with {settings.refine_model}{part}…")
        proposals.extend(_ask(batch, prev[-2:], ctx_text, system, settings, client, max_out, part))
        prev = batch
    segments, corrections = apply_corrections(transcript.segments, proposals, ctx)
    return RefinedTranscript(segments=segments, corrections=corrections, model=settings.refine_model)
