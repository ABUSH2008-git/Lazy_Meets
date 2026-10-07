from meetscribe.minutes import verify
from meetscribe.refine import apply_corrections
from meetscribe.schemas import Correction, MeetingContext, Segment


def segs(*texts, speakers=None):
    return [Segment(id=i + 1, start=i * 5.0, end=i * 5.0 + 4.5, text=t, speaker=(speakers[i] if speakers else None))
            for i, t in enumerate(texts)]


def test_apply_corrections_keeps_ids_and_reports_blocked():
    s = segs("we put Ridis in front of the config", "latency is 420 milliseconds", "we are not doing Kafka")
    props = [
        Correction(segment_id=1, original="Ridis", corrected="Redis", category="product_or_tool"),
        Correction(segment_id=2, original="420 milliseconds", corrected="240 milliseconds"),
        Correction(segment_id=3, original="not doing", corrected="doing"),
        Correction(segment_id=3, original="nonexistent words", corrected="whatever"),
    ]
    out, res = apply_corrections(s, props, MeetingContext(glossary=["Redis", "Kafka"]))
    assert [x.id for x in out] == [1, 2, 3]
    assert out[0].text == "we put Redis in front of the config"
    assert out[1].text == s[1].text and out[2].text == s[2].text
    status = {(r.segment_id, r.original): r.status for r in res}
    assert status[(1, "Ridis")] == "applied"
    assert status[(2, "420 milliseconds")] == "blocked"
    assert status[(3, "not doing")] == "blocked"
    assert status[(3, "nonexistent words")] == "not_found"


def test_wrong_segment_id_is_recovered_and_word_boundaries_respected():
    s = segs("the cashflow is fine", "clear the cash before deploy")
    out, res = apply_corrections(s, [Correction(segment_id=1, original="the cash", corrected="the cache")], MeetingContext())
    assert out[0].text == "the cashflow is fine"  # not touched inside "cashflow"
    assert out[1].text == "clear the cache before deploy"
    assert res[0].segment_id == 2


def test_multiword_fix_propagates():
    s = segs("we use cooper netties now", "cooper netties is down again")
    out, res = apply_corrections(s, [Correction(segment_id=1, original="cooper netties", corrected="Kubernetes")],
                                 MeetingContext(glossary=["Kubernetes"]))
    assert out[1].text == "Kubernetes is down again"
    assert any(r.status == "propagated" for r in res)


TRANSCRIPT = segs(
    "Arjun, can you ship the Redis cache behind a feature flag by Thursday?",
    "Yes, I'll have it behind a flag by Thursday.",
    "And I'll write the rollback runbook for the cache.",
    "Maybe Rohan could look into the flaky tests, but I'll have to ask him first.",
    "Let's go with Redis for the config cache, that's decided.",
    "What if we move to Kafka? Not this quarter.",
)


def raw_record(**over):
    base = {
        "title": "Sync", "summary": "s", "participants": ["Arjun", "Rohan", "Zed"], "minutes": [],
        "decisions": [
            {"decision": "Use Redis for the config cache", "evidence_quote": "let's go with Redis for the config cache", "segment_ids": [5]},
            {"decision": "Hire two engineers", "evidence_quote": "we will hire two engineers", "segment_ids": [1]},
        ],
        "proposals_not_agreed": [{"proposal": "Move to Kafka", "status": "deferred", "evidence_quote": "What if we move to Kafka", "segment_ids": [6]}],
        "action_items": [
            {"task": "Ship the Redis cache behind a feature flag", "status": "confirmed", "owner": "Arjun", "owner_evidence": "Arjun, can you ship",
             "deadline": "by Thursday", "deadline_evidence": "by Thursday", "evidence_quote": "can you ship the Redis cache behind a feature flag", "segment_ids": [1]},
            {"task": "Write the rollback runbook", "status": "confirmed", "owner": "Meera", "owner_evidence": "I'll write the rollback runbook",
             "deadline": "by Friday", "deadline_evidence": None, "evidence_quote": "I'll write the rollback runbook for the cache", "segment_ids": [3]},
            {"task": "Look into the flaky tests", "status": "tentative", "owner": "Rohan", "owner_evidence": "Maybe Rohan could",
             "deadline": None, "deadline_evidence": None, "evidence_quote": "Maybe Rohan could look into the flaky tests", "segment_ids": [4]},
            {"task": "Buy new laptops for the team", "status": "confirmed", "owner": None, "owner_evidence": None, "deadline": None,
             "deadline_evidence": None, "evidence_quote": "we should buy laptops", "segment_ids": [2]},
        ],
        "open_questions": [],
    }
    base.update(over)
    return base


def test_verify_keeps_stated_owner_and_deadline():
    r = verify(raw_record(), TRANSCRIPT, MeetingContext(), {}, "m", "single-pass")
    a1 = r.action_items[0]
    assert (a1.owner, a1.deadline) == ("Arjun", "by Thursday")


def test_verify_removes_unsupported_owner_and_deadline():
    r = verify(raw_record(), TRANSCRIPT, MeetingContext(), {}, "m", "single-pass")
    runbook = next(a for a in r.action_items if "runbook" in a.task)
    assert runbook.owner is None and runbook.deadline is None
    assert runbook.owner_display == "Unspecified" and runbook.deadline_display == "Unspecified"


def test_speaker_label_supports_self_commitment():
    t = segs(*[x.text for x in TRANSCRIPT], speakers=["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 1", "Speaker 1", "Speaker 2"])
    r = verify(raw_record(), t, MeetingContext(), {"Speaker 3": "Meera"}, "m", "single-pass")
    runbook = next(a for a in r.action_items if "runbook" in a.task)
    assert runbook.owner == "Meera"


def test_tentative_items_are_not_confirmed_tasks():
    r = verify(raw_record(), TRANSCRIPT, MeetingContext(), {}, "m", "single-pass")
    assert all("flaky" not in a.task for a in r.action_items)
    f = r.suggested_followups[0]
    assert "flaky" in f.task and f.owner is None


def test_invented_items_are_dropped_and_noted():
    r = verify(raw_record(), TRANSCRIPT, MeetingContext(), {}, "m", "single-pass")
    assert [d.decision for d in r.decisions] == ["Use Redis for the config cache"]
    assert all("laptops" not in a.task for a in r.action_items)
    assert any("Hire two engineers" in n for n in r.verification_notes)
    assert r.proposals[0].status == "deferred"
    assert "Zed" not in r.participants  # never mentioned


def test_empty_lists_are_fine():
    r = verify(raw_record(decisions=[], action_items=[], proposals_not_agreed=[]), TRANSCRIPT, MeetingContext(), {}, "m", "single-pass")
    assert r.decisions == [] and r.action_items == []


def test_target_named_everyone_and_self():
    t = segs("Arjun, can you ship the cache by Thursday?", "Everyone please review the design doc.", "I'll visit Chennai next week.",
             speakers=["Speaker 1", "Speaker 1", "Speaker 2"])
    raw = raw_record(decisions=[], proposals_not_agreed=[], action_items=[
        {"task": "Ship the cache", "status": "confirmed", "target_type": "named_person", "target_name": "Arjun", "owner": "Arjun",
         "owner_evidence": "Arjun, can you ship", "deadline": None, "deadline_evidence": None,
         "evidence_quote": "can you ship the cache by Thursday", "segment_ids": [1]},
        {"task": "Review the design doc", "status": "confirmed", "target_type": "everyone", "target_name": None, "owner": None,
         "owner_evidence": None, "deadline": None, "deadline_evidence": None,
         "evidence_quote": "Everyone please review the design doc", "segment_ids": [2]},
        {"task": "Visit Chennai", "status": "confirmed", "target_type": "self", "target_name": None, "owner": None,
         "owner_evidence": None, "deadline": None, "deadline_evidence": None,
         "evidence_quote": "I'll visit Chennai next week", "segment_ids": [3]},
        {"task": "Book the venue", "status": "confirmed", "target_type": "named_person", "target_name": "Zoya", "owner": None,
         "owner_evidence": None, "deadline": None, "deadline_evidence": None,
         "evidence_quote": "I'll visit Chennai next week", "segment_ids": [3]},
    ])
    with_names = verify(raw, t, MeetingContext(), {"Speaker 2": "Meera"}, "m", "single-pass")
    targets = {a.task: a.target for a in with_names.action_items}
    assert targets["Ship the cache"] == "Arjun"
    assert targets["Review the design doc"] == "Everyone"
    assert targets["Visit Chennai"] == "Meera"      # self -> the speaker's name
    no_diar = verify(raw, segs(*[x.text for x in t]), MeetingContext(), {}, "m", "single-pass")
    targets = {a.task: a.target for a in no_diar.action_items}
    assert targets["Visit Chennai"] is None          # no speaker detection -> left empty
    assert targets["Review the design doc"] == "Everyone"
    assert all("Zoya" != a.target for a in no_diar.action_items)  # never said -> not a target


def test_announced_plan_becomes_decision_and_task():
    """'X will do Y on <date>' said as a plan: kept as a decision and as a task with owner and deadline."""
    t = segs("Siddhu will take the viva on 11th October 2028. He might shift it to 2029 depending on his mood. "
             "The TAs will not take the viva and Siddhu will take the TAs.", "Are you sure?")
    raw = raw_record(
        participants=["Siddhu"],
        decisions=[
            {"decision": "Siddhu will take the viva on 11 October 2028 (may move to 2029)",
             "evidence_quote": "Siddhu will take the viva on 11th October 2028", "segment_ids": [1]},
            {"decision": "The TAs will not take the viva", "evidence_quote": "The TAs will not take the viva", "segment_ids": [1]},
        ],
        proposals_not_agreed=[],
        action_items=[
            {"task": "Take the viva", "status": "confirmed", "target_type": "named_person", "target_name": "Siddhu",
             "owner": "Siddhu", "owner_evidence": "Siddhu will take the viva", "deadline": "on 11th October 2028",
             "deadline_evidence": "on 11th October 2028", "evidence_quote": "Siddhu will take the viva on 11th October 2028",
             "segment_ids": [1]},
            {"task": "Take the TAs", "status": "confirmed", "target_type": "named_person", "target_name": "Siddhu",
             "owner": "Siddhu", "owner_evidence": "Siddhu will take the TAs", "deadline": None, "deadline_evidence": None,
             "evidence_quote": "Siddhu will take the TAs", "segment_ids": [1]},
        ],
    )
    r = verify(raw, t, MeetingContext(), {}, "m", "single-pass")
    assert len(r.decisions) == 2
    viva = next(a for a in r.action_items if "viva" in a.task)
    assert (viva.owner, viva.deadline, viva.target) == ("Siddhu", "on 11th October 2028", "Siddhu")
    assert any("TAs" in a.task and a.owner == "Siddhu" for a in r.action_items)


def test_prompts_cover_announced_plans_and_ordinary_mishearings():
    from meetscribe.config import PROMPTS_DIR
    minutes = (PROMPTS_DIR / "minutes.md").read_text()
    assert "both a decision and an action item" in minutes and "Are you sure?" in minutes
    assert "Vaiva" in (PROMPTS_DIR / "refine.md").read_text()
