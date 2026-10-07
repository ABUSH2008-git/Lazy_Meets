"""Process a meeting recording from the command line (same pipeline as the web app).

Examples:
  python cli.py samples/demo_meeting.mp3
  python cli.py meeting.m4a --participants "Priya, Arjun, Meera" --glossary "Redis, Kafka" --domain "Software engineering"
  python cli.py meeting.wav --diarize --speakers 3 --out outputs/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from meetscribe.config import Settings
from meetscribe.errors import MeetScribeError, StageError
from meetscribe.glossaries import PRESETS, parse_terms
from meetscribe.pipeline import STAGE_LABELS, ingest, new_run, run_pipeline
from meetscribe.schemas import MeetingContext


def main() -> int:
    ap = argparse.ArgumentParser(description="Recorded meeting -> transcript -> refined transcript -> minutes, decisions, action items")
    ap.add_argument("audio", help="meeting recording (mp3, wav, m4a, ogg, webm, flac, mp4…)")
    ap.add_argument("--out", default="outputs", help="folder to write the results into (default: outputs/)")
    ap.add_argument("--title", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--about", default="", help="one line about the meeting, e.g. 'weekly sync of the payments team'")
    ap.add_argument("--participants", default="", help="comma-separated names (spelling hints only)")
    ap.add_argument("--glossary", default="", help="comma-separated terms, acronyms, product names")
    ap.add_argument("--domain", action="append", default=[], choices=list(PRESETS), help="add a preset term list (repeatable)")
    ap.add_argument("--diarize", action="store_true", help="detect who is speaking (offline)")
    ap.add_argument("--speakers", type=int, default=0, help="number of speakers if known (with --diarize)")
    ap.add_argument("--offline-stt", action="store_true", help="transcribe offline with Moonshine instead of Whisper via the API")
    ap.add_argument("--stt-model")
    ap.add_argument("--refine-model")
    ap.add_argument("--minutes-model")
    args = ap.parse_args()

    s = Settings.from_env()
    s.runs_dir = Path(args.out)
    s.persist_runs = True
    s.diarize = args.diarize or s.diarize
    s.num_speakers = args.speakers
    if args.offline_stt:
        s.stt_backend = "local"
    s.stt_model = args.stt_model or s.stt_model
    s.refine_model = args.refine_model or s.refine_model
    s.minutes_model = args.minutes_model or s.minutes_model

    src = Path(args.audio)
    if not src.exists():
        print(f"error: {src} doesn't exist", file=sys.stderr)
        return 2
    ctx = MeetingContext(title=args.title, date=args.date, description=args.about, participants=parse_terms(args.participants),
                         glossary=parse_terms(args.glossary), domains=args.domain)
    t0 = time.time()

    def on_event(stage: str, kind: str, msg: str) -> None:
        tag = STAGE_LABELS.get(stage, stage)
        if kind == "start":
            print(f"\n== {tag}")
        elif kind == "info":
            print(f"   {msg}")
        elif kind == "done":
            print(f"   done: {msg}")
        elif kind == "error":
            print(f"   FAILED: {msg}")

    try:
        run = new_run(src.name, s, ctx)
        ingest(run, src, s)
        run_pipeline(run, s, on_event)
    except StageError as e:
        print(f"\n{STAGE_LABELS.get(e.stage, e.stage)} failed: {e.message}", file=sys.stderr)
        if e.hint:
            print(f"hint: {e.hint}", file=sys.stderr)
        return 1
    except MeetScribeError as e:
        print(f"\nerror: {e.message}", file=sys.stderr)
        if e.hint:
            print(f"hint: {e.hint}", file=sys.stderr)
        return 1

    for name, data in run.export_files().items():
        (run.dir / name).write_bytes(data)
    (run.dir / "all_outputs.zip").write_bytes(run.zip_bytes())
    r = run.record
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {run.dir}/")
    print(f"  {r.title}")
    print(f"  {len(r.decisions)} decisions, {len(r.action_items)} action items, {len(r.open_questions)} open questions")
    for a in r.action_items:
        print(f"  - {a.task}  [owner: {a.owner_display} | deadline: {a.deadline_display}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
