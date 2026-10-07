"""Data models shared by all stages, plus the JSON schemas we send to the language models."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

UNSPECIFIED = "Unspecified"


def fmt_ts(seconds: float, always_hours: bool = False) -> str:
    seconds = max(0.0, float(seconds or 0))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- transcripts


class Word(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    low_confidence: bool = False
    words: list[Word] = Field(default_factory=list)

    def label(self, names: dict[str, str] | None = None) -> Optional[str]:
        if not self.speaker:
            return None
        return (names or {}).get(self.speaker, self.speaker)


class DroppedSegment(BaseModel):
    start: float
    end: float
    text: str
    reason: str


class Transcript(BaseModel):
    segments: list[Segment]
    duration: float
    model: str
    language: str = "en"
    whisper_prompt: str = ""
    dropped: list[DroppedSegment] = Field(default_factory=list)
    diarized: bool = False

    def by_id(self) -> dict[int, Segment]:
        return {s.id: s for s in self.segments}

    def plain_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.speaker and s.speaker not in seen:
                seen.append(s.speaker)
        return seen


CorrectionCategory = Literal["terminology", "acronym", "product_or_tool", "person_name", "number_format", "other"]


class Correction(BaseModel):
    segment_id: int
    original: str
    corrected: str
    category: str = "terminology"
    reason: str = ""
    status: Literal["applied", "blocked", "not_found", "propagated", "unchanged"] = "applied"
    block_reason: Optional[str] = None


class RefinedTranscript(BaseModel):
    segments: list[Segment]
    corrections: list[Correction]
    model: str

    def plain_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def applied(self) -> list[Correction]:
        return [c for c in self.corrections if c.status in ("applied", "propagated")]


# --------------------------------------------------------------------------- meeting record


class MeetingContext(BaseModel):
    """Optional hints the user gives before processing."""

    title: str = ""
    date: str = ""
    description: str = ""
    participants: list[str] = Field(default_factory=list)
    glossary: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)

    def all_terms(self) -> list[str]:
        from .glossaries import terms_for

        seen, out = set(), []
        for t in list(self.glossary) + terms_for(self.domains):
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(t.strip())
        return out


class MinutesSection(BaseModel):
    topic: str
    points: list[str]
    segment_ids: list[int] = Field(default_factory=list)


class Decision(BaseModel):
    id: str = ""
    decision: str
    evidence_quote: str = ""
    segment_ids: list[int] = Field(default_factory=list)
    start: Optional[float] = None  # when the quote was said (word-level), if known
    verified: bool = True
    flags: list[str] = Field(default_factory=list)


class Proposal(BaseModel):
    id: str = ""
    proposal: str
    status: str = "undecided"  # rejected | deferred | undecided
    evidence_quote: str = ""
    segment_ids: list[int] = Field(default_factory=list)
    start: Optional[float] = None
    verified: bool = True


class ActionItem(BaseModel):
    id: str = ""
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    status: str = "confirmed"  # confirmed | tentative
    target: Optional[str] = None  # who the task is aimed at: a name, "Everyone", or the speaker (self)
    target_type: str = "unclear"  # named_person | everyone | self | unclear
    evidence_quote: str = ""
    owner_evidence: Optional[str] = None
    deadline_evidence: Optional[str] = None
    segment_ids: list[int] = Field(default_factory=list)
    start: Optional[float] = None
    verified: bool = True
    flags: list[str] = Field(default_factory=list)

    @property
    def owner_display(self) -> str:
        return self.owner or UNSPECIFIED

    @property
    def deadline_display(self) -> str:
        return self.deadline or UNSPECIFIED


class OpenQuestion(BaseModel):
    question: str
    segment_ids: list[int] = Field(default_factory=list)


class MeetingRecord(BaseModel):
    title: str
    date: str = ""
    summary: str
    participants: list[str] = Field(default_factory=list)
    minutes: list[MinutesSection] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    suggested_followups: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)
    model: str = ""
    mode: str = "single-pass"


# --------------------------------------------------------------------------- JSON schemas for the LLMs
# Strict structured outputs need every property listed in "required" and
# additionalProperties: false on every object. Optional values use ["string", "null"].


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props.keys()), "additionalProperties": False}


_STR = {"type": "string"}
_NSTR = {"type": ["string", "null"]}
_IDS = {"type": "array", "items": {"type": "integer"}}

REFINE_SCHEMA = _obj(
    {
        "corrections": {
            "type": "array",
            "items": _obj(
                {
                    "segment_id": {"type": "integer"},
                    "original": _STR,
                    "category": {
                        "type": "string",
                        "enum": ["terminology", "acronym", "product_or_tool", "person_name", "number_format", "other"],
                    },
                    "reason": _STR,
                    "corrected": _STR,
                }
            ),
        }
    }
)

_DECISION = _obj({"evidence_quote": _STR, "segment_ids": _IDS, "decision": _STR})
_PROPOSAL = _obj(
    {
        "evidence_quote": _STR,
        "segment_ids": _IDS,
        "proposal": _STR,
        "status": {"type": "string", "enum": ["rejected", "deferred", "undecided"]},
    }
)
_ACTION = _obj(
    {
        "evidence_quote": _STR,
        "segment_ids": _IDS,
        "task": _STR,
        "status": {"type": "string", "enum": ["confirmed", "tentative"]},
        "target_type": {"type": "string", "enum": ["named_person", "everyone", "self", "unclear"]},
        "target_name": _NSTR,
        "owner_evidence": _NSTR,
        "owner": _NSTR,
        "deadline_evidence": _NSTR,
        "deadline": _NSTR,
    }
)
_MINUTES = _obj({"topic": _STR, "points": {"type": "array", "items": _STR}, "segment_ids": _IDS})
_QUESTION = _obj({"question": _STR, "segment_ids": _IDS})

_RECORD_BODY = {
    "minutes": {"type": "array", "items": _MINUTES},
    "decisions": {"type": "array", "items": _DECISION},
    "proposals_not_agreed": {"type": "array", "items": _PROPOSAL},
    "action_items": {"type": "array", "items": _ACTION},
    "open_questions": {"type": "array", "items": _QUESTION},
}

MINUTES_SCHEMA = _obj(
    {
        "title": _STR,
        "summary": _STR,
        "participants": {"type": "array", "items": _STR},
        **_RECORD_BODY,
    }
)

# Used for long meetings: each part is processed on its own, then merged.
PART_SCHEMA = _obj({"part_summary": _STR, "participants": {"type": "array", "items": _STR}, **_RECORD_BODY})

QA_SCHEMA = _obj({"found_in_transcript": {"type": "boolean"}, "segment_ids": _IDS, "answer": _STR})
