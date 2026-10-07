"""A tiny fake of Groq's OpenAI-compatible API, for offline tests of the whole app.

It returns the demo meeting transcript for any audio, rule-based corrections for the
refinement call, and a hand-written meeting record (with a few deliberate mistakes that the
app's checks should catch). It also sends rate-limit headers and can fake a 429.

Run:  python tests/mock_groq.py --port 8765
Then: LLM_BASE_URL=http://127.0.0.1:8765/openai/v1 GROQ_API_KEY=test streamlit run app.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "demo_whisper.json").read_text())
MODELS = ["whisper-large-v3", "whisper-large-v3-turbo", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

STATE = {"fail_next_429": 0, "fail_schema": None, "calls": [], "lock": threading.Lock()}

RULES = [
    (r"\bRidis\b", "Redis", "product_or_tool", "sound-alike of Redis; talk is about caching"),
    (r"\briddies\b", "Redis", "product_or_tool", "sound-alike of Redis"),
    (r"\bRiddy's cash\b", "Redis cache", "product_or_tool", "Redis cache misheard"),
    (r"\bthe cash\b", "the cache", "terminology", "cache, not money, in this context"),
    (r"\bpost-gur SQL\b", "PostgreSQL", "product_or_tool", "PostgreSQL misheard"),
    (r"\bOauth\b", "OAuth", "acronym", "casing of OAuth"),
    (r"\bMira\b", "Meera", "person_name", "participant list spells it Meera"),
    # deliberately bad suggestions: the guards must block these
    (r"\bnot this quarter\b", "this quarter", "other", "(bad) drops a negation"),
    (r"\b420 milliseconds\b", "240 milliseconds", "number_format", "(bad) changes a number"),
    (r"\bmaybe Rohan could\b", "Rohan will", "other", "(bad) turns a suggestion into a commitment"),
]


def refine_answer(user: str) -> dict:
    out = []
    for line in user.splitlines():
        m = re.match(r"\[(\d+)\](?: \(context only\))? (.*)", line)
        if not m or "(context only)" in line:
            continue
        sid, text = int(m.group(1)), m.group(2)
        for pat, rep, cat, why in RULES:
            mm = re.search(pat, text)
            if mm:
                out.append({"segment_id": sid, "original": mm.group(0), "category": cat, "reason": why, "corrected": rep})
    return {"corrections": out}


def _seg_id(user: str, needle: str) -> list[int]:
    for line in user.splitlines():
        m = re.match(r"\[(\d+) \|", line)
        if m and needle.lower() in line.lower():
            return [int(m.group(1))]
    return []


def minutes_answer(user: str) -> dict:
    if "Redis" not in user and "riddies" not in user.lower():
        first = [l for l in user.splitlines() if l.startswith("[")][:3]
        return {"title": "Short recording", "summary": "A short recording. " + " ".join(first)[:200], "participants": [],
                "minutes": [{"topic": "Discussion", "points": ["General discussion."], "segment_ids": [1]}],
                "decisions": [], "proposals_not_agreed": [], "action_items": [], "open_questions": []}
    sid = lambda n: _seg_id(user, n)
    return {
        "title": "Payments Platform Weekly Sync",
        "summary": "The team reviewed rising p99 latency on the checkout API, agreed to add a Redis cache for merchant config, "
                   "deferred the Kafka migration, and assigned follow-ups ahead of the release on the 24th.",
        "participants": ["Priya", "Arjun", "Meera", "Rohan", "Vikram"],
        "minutes": [
            {"topic": "Checkout latency", "points": ["p99 latency on the checkout API rose to 420 ms last week.",
                                                    "Most of it comes from repeated PostgreSQL queries for merchant config; p50 is fine."],
             "segment_ids": sid("420 milliseconds")},
            {"topic": "Caching plan", "points": ["Arjun proposed Redis in front of the merchant config table with a 5-minute TTL.",
                                                "In staging this brought p99 down to 180 ms."], "segment_ids": sid("5 minute TTL")},
            {"topic": "Event pipeline", "points": ["Moving the event pipeline to Kafka was raised but there is no bandwidth this quarter."],
             "segment_ids": sid("Kafka")},
            {"topic": "Release and backlog", "points": ["The release stays on the 24th; the OAuth token refresh bug is not blocking."],
             "segment_ids": sid("OAuth token")},
        ],
        "decisions": [
            {"evidence_quote": "let's go with Redis for the merchant config cache that's decided", "segment_ids": sid("that's decided"),
             "decision": "Use Redis as a cache for the merchant config table (5-minute TTL)."},
            {"evidence_quote": "we are not migrating to Kafka this quarter", "segment_ids": sid("not migrating"),
             "decision": "Do not migrate the event pipeline to Kafka this quarter; revisit in January."},
            {"evidence_quote": "it stays in the backlog", "segment_ids": sid("stays in the backlog"),
             "decision": "Keep the OAuth token refresh bug in the backlog."},
            {"evidence_quote": "we will hire two more engineers next sprint", "segment_ids": [3],
             "decision": "Hire two more engineers (made-up decision the checks should flag)."},
        ],
        "proposals_not_agreed": [
            {"evidence_quote": "There was also the idea of moving the event pipeline to Kafka", "segment_ids": sid("idea of moving"),
             "proposal": "Move the event pipeline to Kafka.", "status": "deferred"},
        ],
        "action_items": [
            {"evidence_quote": "can you ship the Redis cache behind a feature flag by Thursday", "segment_ids": sid("feature flag"),
             "task": "Ship the Redis cache behind a feature flag", "status": "confirmed", "target_type": "named_person", "target_name": "Arjun",
             "owner_evidence": "Arjun, can you ship the Redis cache", "owner": "Arjun",
             "deadline_evidence": "by Thursday", "deadline": "by Thursday"},
            {"evidence_quote": "I'll write the rollback runbook for the cache", "segment_ids": sid("rollback runbook"),
             "task": "Write the rollback runbook for the cache", "status": "confirmed", "target_type": "self", "target_name": None,
             "owner_evidence": "I'll write the rollback runbook", "owner": "Meera",
             "deadline_evidence": None, "deadline": "by Friday"},
            {"evidence_quote": "We also need to update the on-call rotation before the release", "segment_ids": sid("on-call"),
             "task": "Update the on-call rotation", "status": "confirmed", "target_type": "unclear", "target_name": None,
             "owner_evidence": None, "owner": None,
             "deadline_evidence": "before the release", "deadline": "before the release"},
            {"evidence_quote": "maybe Rohan could look into the flaky integration tests", "segment_ids": sid("flaky"),
             "task": "Look into the flaky integration tests", "status": "tentative", "target_type": "named_person", "target_name": "Rohan",
             "owner_evidence": "maybe Rohan could look into",
             "owner": "Rohan", "deadline_evidence": None, "deadline": None},
            {"evidence_quote": "I'll check with finance, and get back to you by Monday", "segment_ids": sid("finance"),
             "task": "Check GPU node budget with finance", "status": "confirmed", "target_type": "self", "target_name": None,
             "owner_evidence": None, "owner": "Vikram",
             "deadline_evidence": "get back to you by Monday", "deadline": "by Monday"},
        ],
        "open_questions": [{"question": "Is there budget for the extra GPU nodes for the fraud model (about $2,500 a month)?",
                            "segment_ids": sid("GPU nodes")}],
    }


def part_answer(user: str) -> dict:
    full = minutes_answer(user)
    return {"part_summary": full["summary"], **{k: v for k, v in full.items() if k not in ("title", "summary")}}


def qa_answer(user: str) -> dict:
    q = user.rsplit("Question:", 1)[-1].lower()
    if "thursday" in q or "cache" in q or "redis" in q:
        return {"found_in_transcript": True, "segment_ids": _seg_id(user, "feature flag") or [1],
                "answer": "Arjun will ship the Redis cache behind a feature flag by Thursday."}
    return {"found_in_transcript": False, "segment_ids": [], "answer": "That wasn't discussed in the meeting."}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict, headers: dict | None = None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            if self.headers.get("authorization", "") == "Bearer bad-key":
                return self._send(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error"}})
            return self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]})
        if self.path.endswith("/__calls"):
            return self._send(200, {"calls": STATE["calls"]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(n)
        if self.path.endswith("/__fail_schema"):
            STATE["fail_schema"] = json.loads(raw or b"{}")  # {"name": "meeting_record", "n": 1}
            return self._send(200, {"ok": True})
        if self.path.endswith("/__fail429"):
            STATE["fail_next_429"] = int(json.loads(raw or b"{}").get("n", 1))
            return self._send(200, {"ok": True})
        if STATE["fail_next_429"] > 0:
            STATE["fail_next_429"] -= 1
            return self._send(429, {"error": {"message": "Rate limit reached. Please try again in 1.2s.", "type": "tokens"}},
                              {"retry-after": "1"})
        if self.path.endswith("/audio/transcriptions"):
            STATE["calls"].append({"kind": "stt", "bytes": n})
            time.sleep(0.3)
            return self._send(200, FIXTURE)
        if self.path.endswith("/chat/completions"):
            req = json.loads(raw)
            rf = req.get("response_format") or {}
            name = (rf.get("json_schema") or {}).get("name", "")
            user = req["messages"][-1]["content"]
            STATE["calls"].append({"kind": "chat", "model": req.get("model"), "schema": name, "max": req.get("max_completion_tokens")})
            fs = STATE["fail_schema"]
            if fs and fs.get("name") == name and fs.get("n", 0) > 0:
                fs["n"] -= 1
                return self._send(503, {"error": {"message": "Service unavailable (simulated outage)", "type": "server_error"}})
            if req.get("max_completion_tokens", 0) <= 32:
                content = "{}"
            elif name == "transcript_corrections":
                content = json.dumps(refine_answer(user))
            elif name == "meeting_record":
                content = json.dumps(minutes_answer(user))
            elif name == "meeting_part":
                content = json.dumps(part_answer(user))
            elif name == "meeting_answer":
                content = json.dumps(qa_answer(user))
            else:
                content = "{}"
            time.sleep(float(os.environ.get("MOCK_DELAY", "0.2")))  # MOCK_DELAY=3 to test the UI while busy
            resp = {
                "id": "mock", "object": "chat.completion", "created": int(time.time()), "model": req.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4, "total_tokens": (len(user) + len(content)) // 4},
            }
            return self._send(200, resp, {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7000",
                                          "x-ratelimit-reset-tokens": "7.5s", "x-ratelimit-limit-requests": "1000"})
        self._send(404, {"error": {"message": "not found"}})


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock Groq on http://127.0.0.1:{args.port}/openai/v1")
    srv.serve_forever()
