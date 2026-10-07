"""HTML pieces shared by the app and the downloadable report."""
from __future__ import annotations

import difflib
import html
import json
import re
from typing import Optional

from .schemas import UNSPECIFIED, MeetingRecord, Segment, fmt_ts


def esc(s: object) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def diff_html(raw: str, refined: str) -> str:
    """Word-level diff: removed words struck through in red, new words in green."""
    a, b = raw.split(), refined.split()
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(esc(" ".join(a[i1:i2])))
        else:
            if i2 > i1:
                out.append(f'<del class="ms-del">{esc(" ".join(a[i1:i2]))}</del>')
            if j2 > j1:
                out.append(f'<ins class="ms-ins">{esc(" ".join(b[j1:j2]))}</ins>')
    return " ".join(out)


PLAYER_CSS = """
.msp{font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg,#1d2330)}
.msp *{box-sizing:border-box}
.msp-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px;position:sticky;top:0;background:var(--bg,#fff);padding:6px 0;z-index:2}
.msp-bar audio{flex:1 1 320px;min-width:240px;height:38px}
.msp-bar input{flex:1 1 160px;padding:7px 10px;border:1px solid var(--line,#d9dde5);border-radius:8px;background:var(--bg,#fff);color:inherit;font:inherit}
.msp-tabs{display:flex;gap:4px}
.msp-tabs button{border:1px solid var(--line,#d9dde5);background:var(--card,#f6f7f9);color:inherit;border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;font-size:13px}
.msp-tabs button.on{background:var(--accent,#3056d3);border-color:var(--accent,#3056d3);color:#fff}
.msp-list{display:flex;flex-direction:column;gap:2px}
.msp-row{display:grid;grid-template-columns:62px 1fr;gap:10px;padding:6px 8px;border-radius:8px;cursor:default;border-left:3px solid transparent}
.msp.has-audio .msp-row{cursor:pointer}
.msp-row:hover{background:var(--card,#f6f7f9)}
.msp-row.now{background:color-mix(in srgb,var(--accent,#3056d3) 12%,transparent);border-left-color:var(--accent,#3056d3)}
.msp-row.hide{display:none}
.msp-ts{color:var(--muted,#6b7385);font-variant-numeric:tabular-nums;font-size:13px;padding-top:2px}
.msp-spk{font-weight:600;margin-right:6px}
.msp-low{text-decoration:underline dotted color-mix(in srgb,var(--warn,#a15c00) 70%,transparent);text-underline-offset:3px}
.msp-v{display:none}.msp[data-view="refined"] .v-refined,.msp[data-view="raw"] .v-raw,.msp[data-view="diff"] .v-diff{display:inline}
.ms-del{background:var(--del,#fde2e1);text-decoration:line-through;border-radius:3px;padding:0 2px}
.ms-ins{background:var(--ins,#d9f5e3);text-decoration:none;border-radius:3px;padding:0 2px}
mark.msp-hit{background:#ffe58a;color:#111;border-radius:2px}
.msp-legend{color:var(--muted,#6b7385);font-size:12.5px;margin:2px 0 8px}
"""

SPEAKER_COLORS = ["#3056d3", "#c2410c", "#0f766e", "#9333ea", "#b91c1c", "#4d7c0f", "#a16207", "#0369a1"]


def player_html(refined: list[Segment], raw: list[Segment], names: dict[str, str], audio_b64: Optional[str],
                embed: bool = True, start_view: str = "refined", theme_css: Optional[str] = None) -> str:
    """Transcript you can read along with the audio: click a line to jump there, the current line
    is highlighted while playing, and you can switch between refined, raw and the changes."""
    raw_by_id = {s.id: s for s in raw}
    speakers = [s.speaker for s in refined if s.speaker]
    order = list(dict.fromkeys(speakers))
    color = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(order)}
    hours = (refined[-1].end if refined else 0) >= 3600
    rows = []
    for s in refined:
        r = raw_by_id.get(s.id, s)
        who = s.label(names)
        spk = f'<span class="msp-spk" style="color:{color.get(s.speaker, "inherit")}">{esc(who)}:</span>' if who else ""
        low = ' msp-low" title="The speech recognizer was not confident about this part' if r.low_confidence else ""
        rows.append(
            f'<div class="msp-row" data-t="{s.start:.2f}" data-e="{s.end:.2f}">'
            f'<span class="msp-ts">{fmt_ts(s.start, hours)}</span>'
            f'<span class="msp-txt">{spk}'
            f'<span class="msp-v v-refined">{esc(s.text)}</span>'
            f'<span class="msp-v v-raw{low}">{esc(r.text)}</span>'
            f'<span class="msp-v v-diff">{diff_html(r.text, s.text)}</span></span></div>'
        )
    audio = f'<audio id="msp-audio" controls preload="metadata" src="data:audio/mpeg;base64,{audio_b64}"></audio>' if audio_b64 else ""
    head = f"<style>{PLAYER_CSS}</style>"
    if embed and theme_css:
        head = f"<style>{theme_css}</style>" + head
    elif embed:
        head = (
            "<style>:root{--bg:#fff;--fg:#1d2330;--muted:#6b7385;--line:#d9dde5;--card:#f6f7f9;--accent:#3056d3;--warn:#a15c00;--ins:#d9f5e3;--del:#fde2e1}"
            "@media (prefers-color-scheme: dark){:root{--bg:#0e1117;--fg:#e8eaee;--muted:#a3abba;--line:#2b313b;--card:#1b1f26;--accent:#7f9cff;--warn:#f0b35a;--ins:#1d4030;--del:#4a2424}}"
            "body{margin:0;background:var(--bg)}</style>" + head
        )
    script = """
<script>
(function(){
  const root=document.currentScript.previousElementSibling;
  const a=root.querySelector('audio'); const rows=[...root.querySelectorAll('.msp-row')];
  root.querySelectorAll('.msp-tabs button').forEach(b=>b.onclick=()=>{root.dataset.view=b.dataset.v;
    root.querySelectorAll('.msp-tabs button').forEach(x=>x.classList.toggle('on',x===b));});
  if(a){root.classList.add('has-audio');
    // only one player at a time: this one pauses the app's main player and vice versa
    try{ const pd=window.parent&&window.parent.document; if(pd&&pd!==document){
      a.addEventListener('play',()=>pd.querySelectorAll('audio').forEach(x=>{if(!x.paused)x.pause();}));
      pd.addEventListener('play',e=>{if(e.target!==a&&!a.paused)a.pause();},true);
    }}catch(e){}
    rows.forEach(r=>r.onclick=()=>{a.currentTime=parseFloat(r.dataset.t);a.play();});
    let last=null;
    a.addEventListener('timeupdate',()=>{const t=a.currentTime; let cur=null;
      for(const r of rows){if(parseFloat(r.dataset.t)<=t+0.05) cur=r; else break;}
      if(cur!==last){if(last) last.classList.remove('now'); if(cur){cur.classList.add('now');
        const rb=cur.getBoundingClientRect(); if(rb.top<60||rb.bottom>window.innerHeight-10) cur.scrollIntoView({block:'center',behavior:'smooth'});} last=cur;}});
    window.msSeek=(t)=>{a.currentTime=t;a.play();};
  }
  const q=root.querySelector('.msp-search');
  if(q){q.oninput=()=>{const v=q.value.trim().toLowerCase();
    rows.forEach(r=>{const hit=!v||r.textContent.toLowerCase().includes(v); r.classList.toggle('hide',!hit);});};}
})();
</script>"""
    tabs = (
        '<div class="msp-tabs">'
        f'<button data-v="refined" class="{"on" if start_view == "refined" else ""}">Refined</button>'
        f'<button data-v="raw" class="{"on" if start_view == "raw" else ""}">Raw</button>'
        f'<button data-v="diff" class="{"on" if start_view == "diff" else ""}">Changes</button></div>'
    )
    legend = '<div class="msp-legend">' + ("Click any line to play from there. " if audio_b64 else "") + \
        'In Raw view, dotted underline = low recognizer confidence. In Changes view, red = removed, green = added by refinement.</div>'
    return (
        f'{head}<div class="msp" data-view="{esc(start_view)}">'
        f'<div class="msp-bar">{audio}<input class="msp-search" placeholder="Search transcript…">{tabs}</div>{legend}'
        f'<div class="msp-list">{"".join(rows)}</div></div>{script}'
    )


RECORD_CSS = """
.msr{font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.msr .meta{color:var(--muted,#5d6678);font-size:.92rem;margin-bottom:6px}
.msr .models{color:var(--muted,#5d6678);font-size:.82rem}
.msr .card{background:var(--card,#f7f8fa);border:1px solid var(--line,#e3e6ec);border-radius:10px;padding:10px 14px;margin:8px 0}
.msr .q{color:var(--muted,#5d6678);font-style:italic;font-size:.92rem}
.msr .ts{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--accent,#3056d3);text-decoration:none;font-size:.85rem;margin-left:6px}
.msr .flag{color:var(--warn,#a15c00);font-size:.85rem}
.msr table{border-collapse:collapse;width:100%;font-size:.95rem}
.msr td:first-child{white-space:nowrap}.msr th,.msr td{border-bottom:1px solid var(--line,#e3e6ec);text-align:left;padding:7px 8px;vertical-align:top}
.msr th{font-size:.82rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted,#5d6678)}
.msr .unspec{color:var(--muted,#5d6678);font-style:italic}
.msr .empty{color:var(--muted,#5d6678);font-style:italic}
.msr .pill{display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:99px;border:1px solid var(--line,#e3e6ec);margin-left:6px;color:var(--muted,#5d6678)}
"""


def record_html(record: MeetingRecord, meta: dict, segments: list[Segment], names: dict[str, str], clickable: bool) -> str:
    idx = {s.id: s for s in segments}
    hours = (meta.get("duration_seconds") or 0) >= 3600

    def ts(ids: list[int], start: Optional[float] = None) -> str:
        times = [start] if start is not None else [idx[i].start for i in ids if i in idx]
        if not times:
            return ""
        t = min(times)
        label = fmt_ts(t, hours)
        if clickable:
            return f'<a class="ts" href="#" onclick="window.msSeek&&msSeek({t:.2f});return false;">▶ {label}</a>'
        return f'<span class="ts">{label}</span>'

    def cell(v: Optional[str]) -> str:
        return esc(v) if v else f'<span class="unspec">{UNSPECIFIED}</span>'

    m = meta.get("models", {})
    parts = [f"<style>{RECORD_CSS}</style><div class='msr'>", f"<h1>{esc(record.title)}</h1>"]
    facts = []
    if record.date:
        facts.append(f"Date: {esc(record.date)}")
    facts.append(f"Duration: {fmt_ts(meta.get('duration_seconds', 0), hours)}")
    if record.participants:
        facts.append("Participants: " + esc(", ".join(record.participants)))
    parts.append(f"<div class='meta'>{' · '.join(facts)}</div>")
    parts.append(
        f"<div class='models'>Speech-to-text: {esc(m.get('speech_to_text'))} · Refinement: {esc(m.get('refinement'))} · "
        f"Minutes: {esc(m.get('minutes'))}</div>"
    )
    parts.append(f"<h2>Summary</h2><p>{esc(record.summary)}</p>")

    parts.append("<h2>Minutes</h2>")
    if record.minutes:
        for sec in record.minutes:
            parts.append(f"<h3>{esc(sec.topic)}{ts(sec.segment_ids)}</h3><ul>" + "".join(f"<li>{esc(p)}</li>" for p in sec.points) + "</ul>")
    else:
        parts.append("<p class='empty'>No discussion points were extracted.</p>")

    parts.append("<h2>Key decisions</h2>")
    if record.decisions:
        for d in record.decisions:
            flag = "".join(f"<div class='flag'>⚠ {esc(f)}</div>" for f in d.flags)
            parts.append(f"<div class='card'><b>{esc(d.id)}</b> {esc(d.decision)}{ts(d.segment_ids, d.start)}"
                         + (f"<div class='q'>“{esc(d.evidence_quote)}”</div>" if d.evidence_quote else "") + flag + "</div>")
    else:
        parts.append("<p class='empty'>No decisions were reached in this meeting.</p>")

    parts.append("<h2>Action items</h2>")
    if record.action_items:
        parts.append("<table><tr><th>#</th><th>Task</th><th>Target</th><th>Owner</th><th>Deadline</th><th>Said at</th></tr>")
        for a in record.action_items:
            flag = "".join(f"<div class='flag'>⚠ {esc(f)}</div>" for f in a.flags)
            parts.append(
                f"<tr><td>{esc(a.id)}</td><td>{esc(a.task)}{flag}<div class='q'>“{esc(a.evidence_quote)}”</div></td>"
                f"<td>{esc(a.target or '')}</td><td>{cell(a.owner)}</td><td>{cell(a.deadline)}</td><td>{ts(a.segment_ids, a.start)}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p class='empty'>No action items were assigned.</p>")

    if record.proposals:
        parts.append("<h2>Proposals not agreed</h2><ul>")
        for p in record.proposals:
            parts.append(f"<li>{esc(p.proposal)}<span class='pill'>{esc(p.status)}</span>{ts(p.segment_ids, p.start)}</li>")
        parts.append("</ul>")
    if record.suggested_followups:
        parts.append("<h2>Suggested follow-ups (not confirmed)</h2><ul>")
        for a in record.suggested_followups:
            extra = "".join(f" <span class='flag'>({esc(f)})</span>" for f in a.flags)
            parts.append(f"<li>{esc(a.task)}{extra}{ts(a.segment_ids, a.start)}</li>")
        parts.append("</ul>")
    if record.open_questions:
        parts.append("<h2>Open questions</h2><ul>" + "".join(f"<li>{esc(q.question)}{ts(q.segment_ids)}</li>" for q in record.open_questions) + "</ul>")
    if record.verification_notes:
        parts.append("<h2>Automatic checks</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in record.verification_notes) + "</ul>")
    parts.append("</div>")
    return "".join(parts)


SIDE_CSS = """
<style>
.mss{font-size:15px;line-height:1.6}
.mss .row{display:grid;grid-template-columns:52px 1fr;gap:8px;padding:3px 0;border-bottom:1px dashed rgba(128,128,128,.18)}
.mss .ts{opacity:.6;font-variant-numeric:tabular-nums;font-size:13px;padding-top:2px}
.mss .spk{font-weight:600;margin-right:5px}
.mss .low{text-decoration:underline dotted rgba(200,120,0,.9);text-underline-offset:3px}
.mss del{background:rgba(239,68,68,.18);text-decoration:line-through;border-radius:3px;padding:0 2px}
.mss ins{background:rgba(34,197,94,.22);text-decoration:none;border-radius:3px;padding:0 2px}
</style>"""


def _marked(a: list[str], b: list[str], side: str) -> str:
    """Words of one side, with the words that differ from the other side wrapped in <del>/<ins>."""
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if side == "raw":
            chunk = esc(" ".join(a[i1:i2]))
            out.append(chunk if op == "equal" or not chunk else f"<del>{chunk}</del>")
        else:
            chunk = esc(" ".join(b[j1:j2]))
            out.append(chunk if op == "equal" or not chunk else f"<ins>{chunk}</ins>")
    return " ".join(x for x in out if x)


def side_by_side_html(raw: list[Segment], refined: list[Segment], names: dict[str, str]) -> tuple[str, str]:
    ref_by_id = {s.id: s for s in refined}
    hours = (raw[-1].end if raw else 0) >= 3600
    left, right = [SIDE_CSS + "<div class='mss'>"], [SIDE_CSS + "<div class='mss'>"]
    for r in raw:
        f = ref_by_id.get(r.id, r)
        who = r.label(names)
        spk = f"<span class='spk'>{esc(who)}:</span>" if who else ""
        a, b = r.text.split(), f.text.split()
        low = " class='low' title='Low recognizer confidence'" if r.low_confidence else ""
        ts = f"<span class='ts'>{fmt_ts(r.start, hours)}</span>"
        left.append(f"<div class='row'>{ts}<span>{spk}<span{low}>{_marked(a, b, 'raw')}</span></span></div>")
        right.append(f"<div class='row'>{ts}<span>{spk}{_marked(a, b, 'refined')}</span></div>")
    left.append("</div>")
    right.append("</div>")
    return "".join(left), "".join(right)


def json_for_script(obj) -> str:
    """JSON that is safe to drop inside a <script> tag."""
    return re.sub(r"</", r"<\\/", json.dumps(obj))
