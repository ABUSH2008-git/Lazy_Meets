"""Full pipeline against the local mock API (no network, no API key needed)."""
import json
import urllib.request

import pytest

from meetscribe.errors import StageError
from meetscribe.llm import ModelClient
from meetscribe.pipeline import Run, ingest, new_run, run_pipeline
from meetscribe.schemas import MeetingContext

CTX = MeetingContext(participants=["Priya", "Arjun", "Meera"], glossary=["Redis", "PostgreSQL", "Kafka", "OAuth"],
                     domains=["Software engineering"])


def _post(base, path, body):
    req = urllib.request.Request(base.replace("/openai/v1", "") + path, data=json.dumps(body).encode(), method="POST")
    urllib.request.urlopen(req).read()


def test_end_to_end(settings, demo_audio):
    events = []
    run = new_run(demo_audio.name, settings, CTX)
    ingest(run, demo_audio, settings)
    run_pipeline(run, settings, lambda s, k, m: events.append((s, k)))
    assert [e for e in events if e[1] == "done"] == [("transcribe", "done"), ("refine", "done"), ("minutes", "done")]

    # stage 1: hallucinated "Thanks for watching!" over silence is removed
    assert run.transcript.dropped and "watching" in run.transcript.dropped[0].text
    # stage 2: real fixes applied, meaning-changing ones blocked, ids/timestamps unchanged
    applied = {(c.original, c.corrected) for c in run.refined.applied()}
    assert ("Ridis", "Redis") in applied and ("Mira", "Meera") in applied
    blocked = {c.original for c in run.refined.corrections if c.status == "blocked"}
    assert {"not this quarter", "420 milliseconds", "maybe Rohan could"} <= blocked
    assert [s.id for s in run.refined.segments] == [s.id for s in run.transcript.segments]
    assert "not this quarter" in " ".join(s.text for s in run.refined.segments)
    # stage 3: checked record
    r = run.record
    assert len(r.decisions) == 3
    owners = {a.task: (a.owner_display, a.deadline_display) for a in r.action_items}
    assert owners["Ship the Redis cache behind a feature flag"] == ("Arjun", "by Thursday")
    assert owners["Write the rollback runbook for the cache"] == ("Unspecified", "Unspecified")
    assert owners["Check GPU node budget with finance"][0] == "Unspecified"  # "Vikram" was never said
    assert [f.task for f in r.suggested_followups] == ["Look into the flaky integration tests"]

    # exports: markdown and json carry the same decisions and tasks
    files = run.export_files()
    for name in ["raw_transcript.txt", "refined_transcript.txt", "meeting_record.md", "meeting_record.json", "action_items.csv",
                 "meeting_report.html"]:
        assert name in files and files[name]
    rec = json.loads(files["meeting_record.json"])
    md = files["meeting_record.md"].decode()
    for d in rec["decisions"]:
        assert d["decision"] in md
    for a in rec["action_items"]:
        assert a["task"] in md and a["owner"] in md
    assert rec["action_items"][1]["owner"] == "Unspecified"

    # the run can be reloaded from disk
    again = Run.load(run.dir)
    assert again.record.title == r.title


def test_failed_stage_keeps_earlier_results_and_can_resume(settings, demo_audio, mock_api):
    _post(mock_api, "/__fail_schema", {"name": "meeting_record", "n": 4})
    run = new_run(demo_audio.name, settings, CTX)
    ingest(run, demo_audio, settings)
    with pytest.raises(StageError) as e:
        run_pipeline(run, settings, lambda *a: None)
    assert e.value.stage == "minutes"
    assert run.transcript and run.refined and run.record is None
    assert "provider" in run.errors["minutes"]["message"]
    calls_before = len(json.loads(urllib.request.urlopen(mock_api.replace("/openai/v1", "") + "/__calls").read())["calls"])
    run_pipeline(run, settings, lambda *a: None, start="minutes")
    calls = json.loads(urllib.request.urlopen(mock_api.replace("/openai/v1", "") + "/__calls").read())["calls"][calls_before:]
    assert all(c["kind"] == "chat" and c["schema"] == "meeting_record" for c in calls)  # only stage 3 re-ran
    assert run.record is not None and not run.errors


def test_rate_limit_is_retried(settings, mock_api):
    _post(mock_api, "/__fail429", {"n": 1})
    waits = []
    client = ModelClient(settings, notify=waits.append)
    data = client.chat_json(stage="t", model="openai/gpt-oss-20b", system="x", user="[1] hello", schema_name="transcript_corrections",
                            schema={"type": "object"}, max_tokens=100)
    assert "corrections" in data
    assert any("Rate limit" in w for w in waits)
    assert client.budget_for("openai/gpt-oss-20b") == 8000  # read from the response headers


def test_bad_key_gives_clear_error(settings):
    from meetscribe.errors import ConfigError

    settings.api_key = "bad-key"
    with pytest.raises(ConfigError, match="API key was rejected"):
        ModelClient(settings).available_models()


def test_unknown_model_fails_before_transcribing(settings, demo_audio):
    settings.minutes_model = "llama-9000"
    run = new_run(demo_audio.name, settings, CTX)
    ingest(run, demo_audio, settings)
    with pytest.raises(StageError) as e:
        run_pipeline(run, settings, lambda *a: None)
    assert "llama-9000" in e.value.message
    assert run.transcript is None  # no time wasted on transcription
