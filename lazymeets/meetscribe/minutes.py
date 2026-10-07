"""Stage 3: meeting minutes, decisions and action items (language model #2).

The model reads the refined transcript and returns a structured record where every
decision / proposal / task carries an exact quote and segment ids. Then code checks it
against the transcript:
  - quotes must really be in the transcript (else the item is flagged as unverified)
  - an owner is kept only if the recording ties that person to the task
  - a deadline is kept only if it was said for that task
  - suggestions that nobody accepted go to "suggested follow-ups", not the task list
Anything the checks change is listed in `verification_notes`.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from rapidfuzz import fuzz

from .config import Settings
from .errors import ModelOutputError, RequestTooLargeError
from .guards import numbers_in
from .llm import ModelClient, est_tokens
from .refine import load_prompt, plan_batches
from .schemas import (
    MINUTES_SCHEMA,
    PART_SCHEMA,
    ActionItem,
    Decision,
    MeetingContext,
    MeetingRecord,
    MinutesSection,
    OpenQuestion,
    Proposal,
    RefinedTranscript,
    Segment,
    fmt_ts,
)

Notify = Callable[[str], None]

_NULLISH = {"", "none", "null", "n/a", "na", "unknown", "unspecified", "tbd", "unassigned", "not specified", "not stated", "-"}
_COLLECTIVE = {"we", "us", "team", "the team", "group", "the group", "all", "everyone", "everybody", "all of us"}
_STOP = {"by", "the", "of", "on", "at", "a", "an", "to", "and", "or", "in", "for"}
_CONTENT_STOP = {"that", "this", "with", "from", "will", "would", "should", "could", "they", "them", "their", "there", "about",
                 "into", "have", "been", "were", "what", "when", "which", "also", "then", "than", "more", "some", "make", "team",
                 "meeting", "agreed", "decided", "decision", "task", "item", "need", "needs"}


def format_line(s: Segment, names: dict[str, str]) -> str:
    who = s.label(names)
    return f"[{s.id} | {fmt_ts(s.start)}{' | ' + who if who else ''}] {s.text}"


def _header(ctx: MeetingContext) -> str:
    bits = []
    if ctx.title:
        bits.append(f"Meeting title given by the user: {ctx.title}")
    if ctx.date:
        bits.append(f"Meeting date: {ctx.date}")
    if ctx.description:
        bits.append(f"About this meeting: {ctx.description}")
    if ctx.participants:
        bits.append("Expected participants (spelling hint only, they still must be mentioned to be owners): " + ", ".join(ctx.participants))
    return "\n".join(bits)


# ---------------------------------------------------------------- calling the model


def _call(client: ModelClient, settings: Settings, system: str, user: str, schema: dict, name: str, max_out: int, effort: str) -> dict:
    return client.chat_json(
        stage="minutes",
        model=settings.minutes_model,
        system=system,
        user=user,
        schema_name=name,
        schema=schema,
        max_tokens=max_out,
        reasoning_effort=effort,
    )


def generate_raw(refined: RefinedTranscript, ctx: MeetingContext, settings: Settings, client: ModelClient,
                 notify: Notify, names: dict[str, str]) -> tuple[dict, str]:
    budget = client.budget_for(settings.minutes_model)
    usable = int(budget * 0.88)
    min_out = 3000
    want_out = 12000 if budget >= 30000 else 6000
    effort = settings.minutes_reasoning
    header = _header(ctx)
    lines = [format_line(s, names) for s in refined.segments]
    system = load_prompt("minutes")
    user = (header + "\n\n" if header else "") + "Transcript:\n" + "\n".join(lines)
    need = est_tokens(system + user)
    if need + min_out <= usable:
        notify(f"Writing minutes, decisions and action items with {settings.minutes_model}…")
        try:
            data = _call(client, settings, system, user, MINUTES_SCHEMA, "meeting_record", min(want_out, usable - need), effort)
            return data, "single-pass"
        except (RequestTooLargeError, ModelOutputError):
            notify("That was too much for one request. Switching to part-by-part mode…")
    return _rolling(refined, ctx, settings, client, notify, names, usable, effort), "part-by-part"


def _rolling(refined: RefinedTranscript, ctx: MeetingContext, settings: Settings, client: ModelClient, notify: Notify,
             names: dict[str, str], usable: int, effort: str) -> dict:
    part_system = load_prompt("minutes_part")
    header = _header(ctx)
    out_part = 3500
    overhead = est_tokens(part_system + header) + 400  # + running summary
    limit = max(500, usable - overhead - out_part)
    if usable > 30000:
        limit = min(limit, 9000)
    parts = plan_batches(refined.segments, limit)
    partials: list[dict] = []
    running: list[str] = []
    for i, part in enumerate(parts, start=1):
        notify(f"Writing minutes part by part with {settings.minutes_model} (part {i} of {len(parts)})…")
        earlier = " ".join(running)[-1500:] or "None - this is the first part."
        user = (header + "\n\n" if header else "") + f"Summary of earlier parts: {earlier}\n\nTranscript part {i} of {len(parts)}:\n" + "\n".join(
            format_line(s, names) for s in part
        )
        data = _call(client, settings, part_system, user, PART_SCHEMA, "meeting_part", out_part, effort)
        partials.append(data)
        running.append(f"Part {i}: {data.get('part_summary', '')}")
    if len(partials) == 1:
        p = partials[0]
        return {"title": ctx.title or "Meeting", "summary": p.get("part_summary", ""), **{k: v for k, v in p.items() if k != "part_summary"}}
    notify(f"Merging {len(partials)} parts into one record…")
    merge_system = load_prompt("minutes_merge")
    payload = [dict(part=i + 1, **p) for i, p in enumerate(partials)]
    user = (header + "\n\n" if header else "") + "Notes from each part, in order:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out = 6000 if usable > 12000 else 3500
    if est_tokens(merge_system + user) + out > usable:
        # too big to merge in one go: trim minutes bullets, keep every decision/task untouched
        for p in payload:
            for m in p.get("minutes", []):
                m["points"] = m.get("points", [])[:2]
        user = (header + "\n\n" if header else "") + "Notes from each part, in order:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    try:
        return _call(client, settings, merge_system, user, MINUTES_SCHEMA, "meeting_record", out, effort)
    except (RequestTooLargeError, ModelOutputError):
        notify("Merge step was too large; combining the parts without the model.")
        return _concat(partials, ctx)


def _concat(partials: list[dict], ctx: MeetingContext) -> dict:
    out: dict = {"title": ctx.title or "Meeting", "summary": " ".join(p.get("part_summary", "") for p in partials), "participants": []}
    for k in ("minutes", "decisions", "proposals_not_agreed", "action_items", "open_questions"):
        out[k] = [x for p in partials for x in (p.get(k) or [])]
    out["participants"] = sorted({n for p in partials for n in (p.get("participants") or [])})
    return out


# ---------------------------------------------------------------- checking the output


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (s or "").lower().replace("’", "'"))).strip()


class _Checker:
    def __init__(self, segments: list[Segment], names: dict[str, str]):
        self.segs = segments
        self.idx = {s.id: i for i, s in enumerate(segments)}
        self.norm = [_norm(s.text) for s in segments]
        self.names = names
        self.full_words = set(" ".join(self.norm).split())

    def ids(self, raw) -> list[int]:
        out = []
        for x in raw or []:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v in self.idx and v not in out:
                out.append(v)
        return out

    def window(self, ids: list[int], pad: int) -> str:
        if not ids:
            return ""
        pos = sorted(self.idx[i] for i in ids)
        a, b = max(0, pos[0] - pad), min(len(self.segs), pos[-1] + pad + 1)
        return " ".join(self.norm[a:b])

    def locate(self, quote: str, ids: list[int]) -> tuple[bool, list[int]]:
        q = _norm(quote)
        if len(q) < 4:
            return False, ids
        if ids and fuzz.partial_ratio(q, self.window(ids, 1)) >= 85:
            return True, ids
        best, best_i = 0.0, -1
        for i in range(len(self.segs)):
            text = " ".join(self.norm[i: i + 3])
            sc = fuzz.partial_ratio(q, text)
            if sc > best:
                best, best_i = sc, i
        if best >= 85:
            found = [self.segs[j].id for j in range(best_i, min(best_i + 3, len(self.segs))) if fuzz.partial_ratio(self.norm[j], q) >= 60] or [self.segs[best_i].id]
            return True, found
        return False, ids

    def quote_time(self, quote: str, ids: list[int]) -> Optional[float]:
        """Exact moment the quote starts, using Whisper's word timings (falls back to the segment start)."""
        if not ids:
            return None
        q = _norm(quote).split()
        best, best_t = 0.0, None
        pos = sorted(self.idx[i] for i in ids)
        cands = self.segs[max(0, pos[0] - 1): pos[-1] + 2]
        for seg in cands:
            words = [(_norm(w.word), w.start) for w in seg.words if _norm(w.word)]
            if not words or not q:
                continue
            n = min(len(q), 8)
            target = " ".join(q[:n])
            for i in range(len(words)):
                sc = fuzz.ratio(target, " ".join(w for w, _ in words[i: i + n]))
                if sc > best:
                    best, best_t = sc, words[i][1]
        if best_t is not None and best >= 70:
            return best_t
        return min(self.segs[self.idx[i]].start for i in ids)

    def grounded(self, text: str) -> bool:
        """Do most of the content words of `text` show up close together somewhere in the transcript?"""
        words = [w for w in re.findall(r"[a-z0-9']+", _norm(text)) if len(w) >= 4 and w not in _CONTENT_STOP]
        if not words:
            return True
        best = 0.0
        for i in range(len(self.segs)):
            win = set(" ".join(self.norm[max(0, i - 1): i + 3]).split())
            hit = sum(1 for w in words if w in win or any(fuzz.ratio(w, x) >= 85 for x in win if abs(len(x) - len(w)) <= 2))
            best = max(best, hit / len(words))
        return best >= 0.6

    def speaker_name(self, ids: list[int]) -> Optional[str]:
        """Name (or label) of the speaker of the first cited segment, if speaker detection ran."""
        for i in ids:
            spk = self.segs[self.idx[i]].speaker
            if spk:
                return self.names.get(spk, spk)
        return None

    def speakers_of(self, ids: list[int]) -> set[str]:
        out = set()
        for i in ids:
            s = self.segs[self.idx[i]]
            if s.speaker:
                out.add(s.speaker.lower())
                out.add(self.names.get(s.speaker, s.speaker).lower())
        return out

    def mentions(self, name: str, text: str) -> bool:
        toks = [t for t in re.findall(r"[a-z][a-z'-]+", name.lower()) if t not in _STOP and len(t) >= 2]
        if not toks:
            return False
        words = text.split()
        # "Arjun Sharma" counts as mentioned if "Arjun" was said (people use first names in meetings)
        return any(len(t) >= 3 and (t in words or any(fuzz.ratio(t, w) >= 88 for w in words if abs(len(w) - len(t)) <= 2)) for t in toks)

    def has_terms(self, phrase: str, text: str) -> bool:
        """Every meaningful word of `phrase` (and every number in it) appears in `text`."""
        words = text.split()
        p_nums, t_nums = numbers_in(phrase), numbers_in(text)
        if p_nums and any(t_nums[k] < v for k, v in p_nums.items()):
            return False
        for t in re.findall(r"[a-z]+", _norm(phrase)):
            if t in _STOP or len(t) < 3 or numbers_in(t):
                continue
            if t not in words and not any(fuzz.ratio(t, w) >= 85 for w in words):
                return False
        return True


def _s(v) -> str:
    return str(v).strip() if v is not None else ""


def verify(raw: dict, segments: list[Segment], ctx: MeetingContext, names: dict[str, str], model: str, mode: str) -> MeetingRecord:
    ck = _Checker(segments, names)
    notes: list[str] = []

    decisions: list[Decision] = []
    for d in raw.get("decisions") or []:
        text = _s(d.get("decision"))
        if not text:
            continue
        ok, ids = ck.locate(_s(d.get("evidence_quote")), ck.ids(d.get("segment_ids")))
        item = Decision(decision=text, evidence_quote=_s(d.get("evidence_quote")), segment_ids=ids, verified=ok,
                        start=ck.quote_time(_s(d.get("evidence_quote")), ids))
        if not ok:
            if not ck.grounded(text):
                notes.append(f"Dropped decision \"{text}\": nothing in the transcript supports it.")
                continue
            item.flags.append("The exact quote wasn't found, but the content matches the transcript. Worth a quick check.")
        else:
            near = ck.window(ids, 2)
            if re.search(r"\b(maybe|could we|what if|should we|i suggest|i propose|how about)\b", _norm(item.evidence_quote)) and not re.search(
                r"\b(agree|agreed|yes|yeah|ok|okay|sounds good|let's|lets|decided|go with|approved|fine|deal|done|sure)\b", near
            ):
                item.flags.append("Sounds like a proposal and no clear agreement was found nearby.")
        decisions.append(item)

    proposals: list[Proposal] = []
    for p in raw.get("proposals_not_agreed") or []:
        text = _s(p.get("proposal"))
        if not text:
            continue
        ok, ids = ck.locate(_s(p.get("evidence_quote")), ck.ids(p.get("segment_ids")))
        if not ok and not ck.grounded(text):
            notes.append(f"Dropped proposal \"{text}\": nothing in the transcript supports it.")
            continue
        status = _s(p.get("status")).lower()
        proposals.append(Proposal(proposal=text, status=status if status in {"rejected", "deferred", "undecided"} else "undecided",
                                  evidence_quote=_s(p.get("evidence_quote")), segment_ids=ids, verified=ok,
                                  start=ck.quote_time(_s(p.get("evidence_quote")), ids)))

    actions: list[ActionItem] = []
    followups: list[ActionItem] = []
    for a in raw.get("action_items") or []:
        task = _s(a.get("task"))
        if not task:
            continue
        quote = _s(a.get("evidence_quote"))
        ok, ids = ck.locate(quote, ck.ids(a.get("segment_ids")))
        item = ActionItem(task=task, evidence_quote=quote, segment_ids=ids, verified=ok, start=ck.quote_time(quote, ids),
                          status="tentative" if _s(a.get("status")).lower() == "tentative" else "confirmed")
        if not ok:
            if not ck.grounded(task):
                notes.append(f"Dropped task \"{task}\": nothing in the transcript supports it.")
                continue
            item.flags.append("The exact quote wasn't found, but the task matches the transcript. Worth a quick check.")

        # ---- owner
        owner = _s(a.get("owner"))
        o_ev = _s(a.get("owner_evidence"))
        if owner.lower() in _NULLISH:
            owner = ""
        if owner:
            ev_ok, ev_ids = ck.locate(o_ev, ids) if o_ev else (False, [])
            near = ck.window(ids + ev_ids, 3)
            label_hit = owner.lower() in ck.speakers_of(ids + ev_ids)
            if owner.lower() in _COLLECTIVE:
                keep = owner.lower() in {"everyone", "everybody", "all of us"} and ev_ok and owner.lower() in _norm(o_ev)
                why = "a whole group (\"we\" / \"the team\") isn't a specific owner"
            else:
                keep = label_hit or ck.mentions(owner, near) or (ev_ok and ck.mentions(owner, _norm(o_ev)))
                why = "the recording doesn't tie that name to this task" + (
                    "" if any(s.speaker for s in segments) else " (turn on speaker detection to attribute \"I'll do it\" commitments)"
                )
            if keep:
                item.owner, item.owner_evidence = owner, (o_ev or None)
            else:
                notes.append(f"Owner \"{owner}\" removed from \"{task}\": {why}.")
                item.flags.append(f"Owner \"{owner}\" was suggested by the model but isn't supported by the recording.")

        # ---- target: who the task is aimed at
        ttype = _s(a.get("target_type")).lower()
        tname = _s(a.get("target_name"))
        if tname.lower() in _NULLISH:
            tname = ""
        item.target_type = ttype if ttype in {"named_person", "everyone", "self", "unclear"} else "unclear"
        if item.target_type == "everyone":
            item.target = "Everyone"
        elif item.target_type == "self":
            # only known when speaker detection labelled who said it
            item.target = ck.speaker_name(ids)
        elif item.target_type == "named_person" and tname:
            if ck.mentions(tname, ck.window(ids, 3)) or (o_ev and ck.mentions(tname, _norm(o_ev))):
                item.target = tname
            else:
                notes.append(f"Target \"{tname}\" removed from \"{task}\": that name isn't said near this task.")

        # ---- deadline
        deadline = _s(a.get("deadline"))
        d_ev = _s(a.get("deadline_evidence"))
        if deadline.lower() in _NULLISH:
            deadline = ""
        if deadline:
            ev_ok, ev_ids = ck.locate(d_ev, ids) if d_ev else (False, [])
            near = ck.window(ids + ev_ids, 2)
            if (ev_ok and ck.has_terms(deadline, _norm(d_ev))) or ck.has_terms(deadline, near):
                item.deadline, item.deadline_evidence = deadline, (d_ev or None)
            else:
                notes.append(f"Deadline \"{deadline}\" removed from \"{task}\": no such time was said for this task.")

        if item.status == "tentative":
            if item.owner:
                item.flags.append(f"{item.owner} was suggested but didn't confirm.")
                notes.append(f"\"{task}\" is only a suggestion, so it isn't listed as a confirmed task for {item.owner}.")
                item.owner, item.owner_evidence = None, None
            followups.append(item)
        else:
            actions.append(item)

    actions = _dedupe(actions)
    followups = _dedupe(followups)

    minutes = []
    for m in raw.get("minutes") or []:
        pts = [_s(x) for x in (m.get("points") or []) if _s(x)]
        if _s(m.get("topic")) or pts:
            minutes.append(MinutesSection(topic=_s(m.get("topic")) or "Discussion", points=pts, segment_ids=ck.ids(m.get("segment_ids"))))
    questions = [OpenQuestion(question=_s(q.get("question")), segment_ids=ck.ids(q.get("segment_ids")))
                 for q in raw.get("open_questions") or [] if _s(q.get("question"))]

    full = " ".join(ck.norm)
    people: list[str] = []
    for n in list(raw.get("participants") or []) + list(ctx.participants):
        n = _s(n)
        if n and not re.match(r"(?i)speaker\s*\d+$", n) and n.lower() not in {p.lower() for p in people}:
            if ck.mentions(n, full) or n in ctx.participants:
                people.append(n)
    for lab in sorted({s.speaker for s in segments if s.speaker}):
        nm = names.get(lab)
        if nm and nm.lower() not in {p.lower() for p in people}:
            people.append(nm)

    for i, d in enumerate(decisions, 1):
        d.id = f"D{i}"
    for i, p in enumerate(proposals, 1):
        p.id = f"P{i}"
    for i, a in enumerate(actions, 1):
        a.id = f"A{i}"
    for i, f in enumerate(followups, 1):
        f.id = f"S{i}"

    return MeetingRecord(
        title=_s(raw.get("title")) or ctx.title or "Meeting",
        date=ctx.date,
        summary=_s(raw.get("summary")),
        participants=people,
        minutes=minutes,
        decisions=decisions,
        proposals=proposals,
        action_items=actions,
        suggested_followups=followups,
        open_questions=questions,
        verification_notes=notes,
        model=model,
        mode=mode,
    )


def _dedupe(items: list[ActionItem]) -> list[ActionItem]:
    out: list[ActionItem] = []
    for it in items:
        dup = next((o for o in out if fuzz.token_set_ratio(o.task.lower(), it.task.lower()) >= 90), None)
        if dup:
            dup.owner = dup.owner or it.owner
            dup.target = dup.target or it.target
            dup.deadline = dup.deadline or it.deadline
            dup.owner_evidence = dup.owner_evidence or it.owner_evidence
            dup.deadline_evidence = dup.deadline_evidence or it.deadline_evidence
            dup.segment_ids = sorted(set(dup.segment_ids) | set(it.segment_ids))
        else:
            out.append(it)
    return out


def generate_minutes(refined: RefinedTranscript, ctx: MeetingContext, settings: Settings, client: ModelClient,
                     notify: Notify, names: Optional[dict[str, str]] = None) -> MeetingRecord:
    names = names or {}
    raw, mode = generate_raw(refined, ctx, settings, client, notify, names)
    notify("Checking every decision, owner and deadline against the transcript…")
    return verify(raw, refined.segments, ctx, names, settings.minutes_model, mode)
