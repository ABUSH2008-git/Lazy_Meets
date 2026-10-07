"""Turn a finished run into files: transcripts (.txt/.srt), the meeting record (.md/.json),
action items (.csv), an interactive HTML report, an email draft, and a zip of everything.

The Markdown, JSON, CSV and HTML versions are all built from the same MeetingRecord object,
so they always list the same decisions and tasks.
"""
from __future__ import annotations

import base64
import csv
import io
import re
from typing import Optional

from . import __version__
from .render import player_html, record_html
from .schemas import UNSPECIFIED, ActionItem, MeetingRecord, Segment, fmt_ts


def _hours(duration: float) -> bool:
    return duration >= 3600


def transcript_txt(segments: list[Segment], title: str, kind: str, names: dict[str, str], duration: float, model: str) -> str:
    h = _hours(duration)
    lines = [
        f"{title} - {kind}",
        f"Model: {model}",
        f"Duration: {fmt_ts(duration, h)}",
        "",
    ]
    for s in segments:
        who = s.label(names)
        lines.append(f"[{fmt_ts(s.start, h)}] " + (f"{who}: " if who else "") + s.text.strip())
    return "\n".join(lines) + "\n"


def _srt_ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d},{ms:03d}"


def transcript_srt(segments: list[Segment], names: dict[str, str]) -> str:
    out = []
    for i, s in enumerate(segments, start=1):
        who = s.label(names)
        out.append(f"{i}\n{_srt_ts(s.start)} --> {_srt_ts(max(s.end, s.start + 0.5))}\n" + (f"{who}: " if who else "") + s.text.strip() + "\n")
    return "\n".join(out)


def _when(ids: list[int], seg_index: dict[int, Segment], duration: float, start: Optional[float] = None) -> Optional[str]:
    if start is not None:
        return fmt_ts(start, _hours(duration))
    times = [seg_index[i].start for i in ids if i in seg_index]
    return fmt_ts(min(times), _hours(duration)) if times else None


def record_dict(record: MeetingRecord, meta: dict, segments: list[Segment]) -> dict:
    """The machine-readable meeting record. Missing owners/deadlines are the string "Unspecified"."""
    idx = {s.id: s for s in segments}
    dur = meta.get("duration_seconds", 0) or 0

    def task(a: ActionItem) -> dict:
        return {
            "id": a.id,
            "task": a.task,
            "target": a.target,
            "owner": a.owner or UNSPECIFIED,
            "deadline": a.deadline or UNSPECIFIED,
            "status": a.status,
            "evidence": a.evidence_quote,
            "owner_evidence": a.owner_evidence,
            "deadline_evidence": a.deadline_evidence,
            "timestamp": _when(a.segment_ids, idx, dur, a.start),
            "segment_ids": a.segment_ids,
            "verified_in_transcript": a.verified,
            "flags": a.flags,
        }

    return {
        "schema_version": "1.0",
        "title": record.title,
        "date": record.date or None,
        "generated_at": meta.get("generated_at"),
        "source": {"audio_file": meta.get("audio_file"), "duration_seconds": round(dur, 1)},
        "models": meta.get("models", {}),
        "summary": record.summary,
        "participants": record.participants,
        "minutes": [
            {"topic": m.topic, "points": m.points, "timestamp": _when(m.segment_ids, idx, dur), "segment_ids": m.segment_ids}
            for m in record.minutes
        ],
        "decisions": [
            {
                "id": d.id,
                "decision": d.decision,
                "evidence": d.evidence_quote,
                "timestamp": _when(d.segment_ids, idx, dur, d.start),
                "segment_ids": d.segment_ids,
                "verified_in_transcript": d.verified,
                "flags": d.flags,
            }
            for d in record.decisions
        ],
        "action_items": [task(a) for a in record.action_items],
        "proposals_not_agreed": [
            {
                "id": p.id,
                "proposal": p.proposal,
                "status": p.status,
                "evidence": p.evidence_quote,
                "timestamp": _when(p.segment_ids, idx, dur, p.start),
                "segment_ids": p.segment_ids,
            }
            for p in record.proposals
        ],
        "suggested_followups_not_confirmed": [task(a) for a in record.suggested_followups],
        "open_questions": [{"question": q.question, "timestamp": _when(q.segment_ids, idx, dur), "segment_ids": q.segment_ids} for q in record.open_questions],
        "verification_notes": record.verification_notes,
        "minutes_mode": record.mode,
    }


def _md_cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def record_markdown(record: MeetingRecord, meta: dict, segments: list[Segment]) -> str:
    d = record_dict(record, meta, segments)
    m = meta.get("models", {})
    out = [f"# {record.title}", ""]
    facts = []
    if record.date:
        facts.append(f"**Date:** {record.date}")
    facts.append(f"**Duration:** {fmt_ts(meta.get('duration_seconds', 0), _hours(meta.get('duration_seconds', 0)))}")
    if record.participants:
        facts.append("**Participants:** " + ", ".join(record.participants))
    out += [" · ".join(facts), ""]
    out += [
        f"_Generated by LazyMeets from `{meta.get('audio_file', 'recording')}`. Speech-to-text: {m.get('speech_to_text', '?')} · "
        f"Refinement: {m.get('refinement', '?')} · Minutes: {m.get('minutes', '?')}_",
        "",
        "## Summary",
        "",
        record.summary or "_No summary._",
        "",
        "## Minutes",
        "",
    ]
    if d["minutes"]:
        for sec in d["minutes"]:
            out.append(f"### {sec['topic']}" + (f" ({sec['timestamp']})" if sec["timestamp"] else ""))
            out += [f"- {p}" for p in sec["points"]]
            out.append("")
    else:
        out += ["_No discussion points were extracted._", ""]

    out += ["## Key decisions", ""]
    if d["decisions"]:
        for x in d["decisions"]:
            warn = " ⚠️ " + " ".join(x["flags"]) if x["flags"] else ""
            out.append(f"{x['id']}. **{x['decision']}**" + (f" ({x['timestamp']})" if x["timestamp"] else "") + warn)
            if x["evidence"]:
                out.append(f"    > \"{x['evidence']}\"")
        out.append("")
    else:
        out += ["_No decisions were reached in this meeting._", ""]

    out += ["## Action items", ""]
    if d["action_items"]:
        out += ["| # | Task | Target | Owner | Deadline | Said at |", "|---|---|---|---|---|---|"]
        for a in d["action_items"]:
            flag = " ⚠️" if a["flags"] else ""
            out.append(f"| {a['id']} | {_md_cell(a['task'])}{flag} | {_md_cell(a['target'] or '')} | {_md_cell(a['owner'])} | "
                       f"{_md_cell(a['deadline'])} | {a['timestamp'] or ''} |")
        out.append("")
    else:
        out += ["_No action items were assigned._", ""]

    if d["proposals_not_agreed"]:
        out += ["## Proposals not agreed", ""]
        for p in d["proposals_not_agreed"]:
            out.append(f"- **{p['proposal']}** ({p['status']})" + (f" ({p['timestamp']})" if p["timestamp"] else ""))
        out.append("")
    if d["suggested_followups_not_confirmed"]:
        out += ["## Suggested follow-ups (not confirmed)", "", "_Ideas that came up but nobody committed to. Not part of the action items._", ""]
        for a in d["suggested_followups_not_confirmed"]:
            extra = f" ({'; '.join(a['flags'])})" if a["flags"] else ""
            out.append(f"- {a['task']}{extra}" + (f" ({a['timestamp']})" if a["timestamp"] else ""))
        out.append("")
    if d["open_questions"]:
        out += ["## Open questions", ""]
        out += [f"- {q['question']}" + (f" ({q['timestamp']})" if q["timestamp"] else "") for q in d["open_questions"]]
        out.append("")
    if d["verification_notes"]:
        out += ["## Automatic checks", "", "_Things the app changed after checking the model's output against the transcript:_", ""]
        out += [f"- {n}" for n in d["verification_notes"]]
        out.append("")
    return "\n".join(out)


def action_items_csv(record: MeetingRecord, meta: dict, segments: list[Segment]) -> str:
    d = record_dict(record, meta, segments)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "task", "target", "owner", "deadline", "status", "timestamp", "evidence"])
    for a in d["action_items"] + d["suggested_followups_not_confirmed"]:
        w.writerow([a["id"], a["task"], a["target"] or "", a["owner"], a["deadline"], a["status"], a["timestamp"] or "", a["evidence"]])
    return buf.getvalue()


def corrections_csv(corrections) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["segment_id", "status", "original", "corrected", "category", "reason", "blocked_because"])
    for c in corrections:
        w.writerow([c.segment_id, c.status, c.original, c.corrected, c.category, c.reason, c.block_reason or ""])
    return buf.getvalue()


def email_draft(record: MeetingRecord) -> tuple[str, str]:
    subject = f"Meeting notes: {record.title}" + (f" ({record.date})" if record.date else "")
    lines = ["Hi all,", "", "Here are the notes from our meeting.", "", "Summary", record.summary, ""]
    lines.append("Decisions")
    lines += [f"- {d.decision}" for d in record.decisions] or ["- None"]
    lines += ["", "Action items"]
    lines += [f"- {a.task} (owner: {a.owner_display}, due: {a.deadline_display})" for a in record.action_items] or ["- None"]
    if record.open_questions:
        lines += ["", "Open questions"] + [f"- {q.question}" for q in record.open_questions]
    if record.suggested_followups:
        lines += ["", "Ideas not confirmed yet"] + [f"- {a.task}" for a in record.suggested_followups]
    lines += ["", "Thanks!"]
    return subject, "\n".join(lines)


def report_html(record: MeetingRecord, meta: dict, raw_segments: list[Segment], refined_segments: list[Segment],
                names: dict[str, str], audio_bytes: Optional[bytes] = None) -> str:
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes and len(audio_bytes) < 20 * 1024 * 1024 else None
    body = record_html(record, meta, refined_segments, names, clickable=audio_b64 is not None)
    player = player_html(refined_segments, raw_segments, names, audio_b64, embed=False)
    title = re.sub(r"[<>&\"]", "", record.title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{REPORT_CSS}</style></head>
<body><main class="ms-report">
{body}
<h2>Transcript</h2>
{player}
<p class="ms-foot">Generated by LazyMeets {__version__} on {meta.get('generated_at', '')}.</p>
</main></body></html>"""


REPORT_CSS = """
:root{--bg:#ffffff;--fg:#1d2330;--muted:#5d6678;--line:#e3e6ec;--card:#f7f8fa;--accent:#3056d3;--warn:#a15c00;--ok:#1f7a4d;--ins:#d9f5e3;--del:#fde2e1}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--fg:#e8eaee;--muted:#a3abba;--line:#2b313b;--card:#1b1f26;--accent:#7f9cff;--warn:#f0b35a;--ok:#5ccf93;--ins:#1d4030;--del:#4a2424}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}
.ms-report{max-width:920px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:1.7rem;margin:0 0 6px} h2{font-size:1.2rem;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:1.02rem;margin:14px 0 4px}
.ms-foot{color:var(--muted);font-size:.85rem;margin-top:30px}
"""
