"""Runs the three stages in order and keeps everything about one processed recording together.

    audio file
      -> Stage 1  speech-to-text  (Whisper)          -> raw transcript (+ speakers, optional)
      -> Stage 2  refinement      (language model #1) -> refined transcript + list of corrections
      -> Stage 3  documentation   (language model #2) -> minutes, decisions, action items

Each stage's output is saved in the run folder, so a failed stage can be retried without
redoing the ones before it, and past runs can be reopened.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from . import exports
from .audio import prepare, validate_upload
from .config import Settings, provider_for_url
from .diarize import apply_speakers, diarize_samples
from .errors import MeetScribeError, StageError
from .llm import ModelClient
from .minutes import generate_minutes
from .refine import refine
from .schemas import MeetingContext, MeetingRecord, RefinedTranscript, Transcript
from .stt import transcribe

STAGES = ["transcribe", "refine", "minutes"]
STAGE_LABELS = {
    "transcribe": "Speech-to-text",
    "refine": "Transcript refinement",
    "minutes": "Minutes, decisions & tasks",
}

Event = Callable[[str, str, str], None]  # (stage, kind: start|info|done|error, message)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", Path(s).stem).strip("-").lower()
    return s[:40] or "meeting"


@dataclass
class Run:
    run_id: str
    dir: Path
    audio_name: str
    created: str
    context: MeetingContext = field(default_factory=MeetingContext)
    settings: dict = field(default_factory=dict)
    speaker_names: dict = field(default_factory=dict)
    duration: float = 0.0
    transcript: Optional[Transcript] = None
    refined: Optional[RefinedTranscript] = None
    record: Optional[MeetingRecord] = None
    errors: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    usage: list = field(default_factory=list)
    models: dict = field(default_factory=dict)
    audio_stats: dict = field(default_factory=dict)

    # ------------------------------------------------------------ files

    @property
    def playback_path(self) -> Path:
        return self.dir / "playback.mp3"

    @property
    def source_path(self) -> Path:
        return self.dir / ("source" + Path(self.audio_name).suffix.lower())

    def done(self, stage: str) -> bool:
        return {"transcribe": self.transcript, "refine": self.refined, "minutes": self.record}[stage] is not None

    def next_stage(self) -> Optional[str]:
        for s in STAGES:
            if not self.done(s):
                return s
        return None

    def save(self) -> None:
        data = {
            "run_id": self.run_id,
            "audio_name": self.audio_name,
            "created": self.created,
            "context": self.context.model_dump(),
            "settings": self.settings,
            "speaker_names": self.speaker_names,
            "duration": self.duration,
            "transcript": self.transcript.model_dump() if self.transcript else None,
            "refined": self.refined.model_dump() if self.refined else None,
            "record": self.record.model_dump() if self.record else None,
            "errors": self.errors,
            "timings": self.timings,
            "usage": self.usage,
            "models": self.models,
            "audio_stats": self.audio_stats,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.dir / "run.json")

    @classmethod
    def load(cls, folder: Path) -> "Run":
        d = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        return cls(
            run_id=d["run_id"],
            dir=folder,
            audio_name=d["audio_name"],
            created=d["created"],
            context=MeetingContext(**d.get("context", {})),
            settings=d.get("settings", {}),
            speaker_names=d.get("speaker_names", {}),
            duration=d.get("duration", 0.0),
            transcript=Transcript(**d["transcript"]) if d.get("transcript") else None,
            refined=RefinedTranscript(**d["refined"]) if d.get("refined") else None,
            record=MeetingRecord(**d["record"]) if d.get("record") else None,
            errors=d.get("errors", {}),
            timings=d.get("timings", {}),
            usage=d.get("usage", []),
            models=d.get("models", {}),
            audio_stats=d.get("audio_stats", {}),
        )

    # ------------------------------------------------------------ exports

    def meta(self) -> dict:
        return {
            "audio_file": self.audio_name,
            "duration_seconds": self.duration,
            "generated_at": self.created,
            "models": self.models,
        }

    def export_files(self) -> dict[str, bytes]:
        files: dict[str, str | bytes] = {}
        title = (self.record.title if self.record else None) or self.context.title or Path(self.audio_name).stem
        if self.transcript:
            files["raw_transcript.txt"] = exports.transcript_txt(
                self.transcript.segments, title, "Raw transcript (speech-to-text, before refinement)", self.speaker_names,
                self.duration, self.transcript.model)
            files["raw_transcript.srt"] = exports.transcript_srt(self.transcript.segments, self.speaker_names)
        if self.refined:
            files["refined_transcript.txt"] = exports.transcript_txt(
                self.refined.segments, title, "Refined transcript (after terminology correction)", self.speaker_names,
                self.duration, self.refined.model)
            files["refined_transcript.srt"] = exports.transcript_srt(self.refined.segments, self.speaker_names)
            files["refinement_changes.csv"] = exports.corrections_csv(self.refined.corrections)
        if self.record and self.refined:
            meta = self.meta()
            segs = self.refined.segments
            files["meeting_record.md"] = exports.record_markdown(self.record, meta, segs)
            files["meeting_record.json"] = json.dumps(exports.record_dict(self.record, meta, segs), ensure_ascii=False, indent=2)
            files["action_items.csv"] = exports.action_items_csv(self.record, meta, segs)
            audio = self.playback_path.read_bytes() if self.playback_path.exists() else None
            files["meeting_report.html"] = exports.report_html(self.record, meta, self.transcript.segments, segs, self.speaker_names, audio)
        out = {k: (v.encode("utf-8") if isinstance(v, str) else v) for k, v in files.items()}
        return out

    def zip_bytes(self) -> bytes:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in self.export_files().items():
                z.writestr(name, data)
        return buf.getvalue()


def new_run(audio_name: str, settings: Settings, context: MeetingContext) -> Run:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{_slug(audio_name)}"
    if settings.persist_runs:
        folder = settings.runs_dir / run_id
    else:
        _cleanup_temp_runs()
        folder = Path(tempfile.mkdtemp(prefix="meetscribe_")) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return Run(run_id=run_id, dir=folder, audio_name=audio_name, created=datetime.now().isoformat(timespec="seconds"),
               context=context, settings=settings.public_dict())


def _cleanup_temp_runs(max_age_hours: float = 12) -> None:
    """On a public server runs aren't kept; delete temporary run folders older than a few hours."""
    root = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_hours * 3600
    for p in root.glob("meetscribe_*"):
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def list_runs(settings: Settings) -> list[Path]:
    if not settings.runs_dir.exists():
        return []
    return sorted((p for p in settings.runs_dir.iterdir() if (p / "run.json").exists()), reverse=True)


def models_used(settings: Settings) -> dict:
    return {
        "speech_to_text": settings.stt_model if settings.stt_backend == "api" else "moonshine-base-en (offline)",
        "refinement": settings.refine_model,
        "minutes": settings.minutes_model,
        "provider": provider_for_url(settings.base_url).name if provider_for_url(settings.base_url).id != "custom" else settings.base_url,
        "speech_to_text_provider": (provider_for_url(settings.stt_url).name if provider_for_url(settings.stt_url).id != "custom"
                                    else settings.stt_url) if settings.stt_backend == "api" else "offline",
        "speaker_detection": "pyannote-segmentation-3.0 + TitaNet-small (offline)" if settings.diarize else None,
    }


def ingest(run: Run, src: Path, settings: Settings) -> None:
    """Copy the uploaded file into the run folder (validation happens again in stage 1)."""
    validate_upload(run.audio_name, src.stat().st_size, settings.max_upload_mb)
    shutil.copyfile(src, run.source_path)


def run_pipeline(run: Run, settings: Settings, on_event: Event, start: Optional[str] = None,
                 client: Optional[ModelClient] = None) -> Run:
    """Run the stages from `start` (default: the first one not done yet) to the end."""
    start = start or run.next_stage() or "minutes"
    todo = STAGES[STAGES.index(start):]
    current = {"stage": todo[0]}

    def notify(msg: str) -> None:
        on_event(current["stage"], "info", msg)

    # reset results from the stages we're about to (re)run
    for s in todo:
        run.errors.pop(s, None)
        if s == "transcribe":
            run.transcript = None
        if s in ("transcribe", "refine"):
            run.refined = None
        run.record = None
    run.models = models_used(settings)
    run.settings = settings.public_dict()

    try:
        if client is None and (settings.stt_backend == "api" or set(todo) & {"refine", "minutes"}):
            client = ModelClient(settings, notify)
        elif client is not None:
            client.notify = notify
        if client is not None:
            needed = {}
            if "transcribe" in todo and settings.stt_backend == "api":
                needed["speech-to-text"] = settings.stt_model
            if "refine" in todo:
                needed["refinement"] = settings.refine_model
            if "minutes" in todo:
                needed["minutes"] = settings.minutes_model
            on_event(todo[0], "info", "Checking the API key and models…")
            client.preflight(needed)
    except MeetScribeError as e:
        run.errors[todo[0]] = {"message": e.message, "hint": e.hint, "detail": e.detail}
        on_event(todo[0], "error", e.message)
        run.save()
        raise StageError(todo[0], e)

    for stage in todo:
        current["stage"] = stage
        on_event(stage, "start", STAGE_LABELS[stage])
        t0 = time.time()
        try:
            if stage == "transcribe":
                _stage_transcribe(run, settings, client, notify)
            elif stage == "refine":
                run.refined = refine(run.transcript, run.context, settings, client, notify)
            else:
                run.record = generate_minutes(run.refined, run.context, settings, client, notify, run.speaker_names)
        except MeetScribeError as e:
            run.errors[stage] = {"message": e.message, "hint": e.hint, "detail": e.detail, "kind": type(e).__name__}
            run.usage = client.usage_summary() if client else run.usage
            run.save()
            on_event(stage, "error", e.message)
            raise StageError(stage, e)
        except Exception as e:  # anything unexpected still gets a readable message
            err = MeetScribeError(f"Something went wrong during {STAGE_LABELS[stage].lower()}: {e}",
                                  hint="Retry this step. If it keeps failing, check the details below.",
                                  detail=traceback.format_exc()[-3000:])
            run.errors[stage] = {"message": err.message, "hint": err.hint, "detail": err.detail}
            run.save()
            on_event(stage, "error", err.message)
            raise StageError(stage, err)
        run.timings[stage] = round(time.time() - t0, 1)
        run.usage = client.usage_summary() if client else []
        run.save()
        on_event(stage, "done", _done_message(run, stage))
    return run


def _stage_transcribe(run: Run, settings: Settings, client: Optional[ModelClient], notify) -> None:
    notify("Checking and converting the audio…")
    audio = prepare(run.source_path, run.dir / "audio", run.audio_name, settings.chunk_minutes)
    shutil.copyfile(audio.playback_path, run.playback_path)
    run.duration = audio.duration
    run.audio_stats = {"peak_dbfs": round(float(audio.stats.peak_dbfs), 1), "rms_dbfs": round(float(audio.stats.rms_dbfs), 1),
                       "chunks": len(audio.chunks)}
    transcript = transcribe(audio, run.context, settings, client, notify)
    if settings.diarize:
        notify("Identifying speakers (runs on this machine)…")
        turns = diarize_samples(audio.samples, settings, notify)
        transcript = apply_speakers(transcript, turns)
    run.transcript = transcript


def _done_message(run: Run, stage: str) -> str:
    if stage == "transcribe" and run.transcript:
        t = run.transcript
        extra = f", {len(t.speakers())} speakers" if t.diarized else ""
        dropped = f", {len(t.dropped)} noise segment(s) removed" if t.dropped else ""
        return f"{len(t.segments)} segments, {len(t.plain_text().split())} words{extra}{dropped}"
    if stage == "refine" and run.refined:
        a = len(run.refined.applied())
        b = sum(1 for c in run.refined.corrections if c.status == "blocked")
        return f"{a} correction(s) applied" + (f", {b} blocked by the meaning checks" if b else "")
    if stage == "minutes" and run.record:
        r = run.record
        return f"{len(r.decisions)} decision(s), {len(r.action_items)} action item(s)"
    return "done"
