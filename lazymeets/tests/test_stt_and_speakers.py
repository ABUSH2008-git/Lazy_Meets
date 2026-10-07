import numpy as np

from meetscribe.diarize import apply_speakers
from meetscribe.schemas import MeetingContext, Segment, Transcript, Word
from meetscribe.stt import _clean, _segments_from_api, build_whisper_prompt


def test_whisper_prompt_stays_under_limit():
    ctx = MeetingContext(participants=["Priya", "Arjun"], glossary=["Project Kestrel"], domains=["Software engineering", "Machine learning / AI"])
    p = build_whisper_prompt(ctx)
    assert "Priya" in p and "Project Kestrel" in p
    assert len(p) / 3.4 < 200


def test_api_segments_get_offsets_and_words():
    data = {"segments": [{"start": 0.0, "end": 2.0, "text": " hello there", "avg_logprob": -0.1, "no_speech_prob": 0.01}],
            "words": [{"word": "hello", "start": 0.1, "end": 0.5}, {"word": "there", "start": 0.6, "end": 1.0}]}
    s = _segments_from_api(data, offset=600.0)
    assert s[0].start == 600.0 and [w.word for w in s[0].words] == ["hello", "there"] and s[0].words[0].start == 600.1


def test_clean_drops_hallucination_over_silence_and_flags_low_confidence():
    sr = 16000
    rng = np.random.default_rng(0)
    audio = np.concatenate([rng.normal(0, 0.2, 5 * sr), np.zeros(3 * sr)]).astype(np.float32)
    segs = [
        Segment(id=0, start=0, end=4, text="we ship on friday", avg_logprob=-0.2, no_speech_prob=0.01),
        Segment(id=0, start=1, end=4, text="something unclear", avg_logprob=-0.9, no_speech_prob=0.05),
        Segment(id=0, start=5.5, end=7.5, text="Thanks for watching!", avg_logprob=-0.4, no_speech_prob=0.35),
    ]
    kept, dropped = _clean(segs, audio, "")
    assert [k.text for k in kept] == ["we ship on friday", "something unclear"]
    assert kept[1].low_confidence and not kept[0].low_confidence
    assert dropped[0].text == "Thanks for watching!"


def test_speakers_split_a_segment_at_the_change():
    words = [Word(word=w, start=i * 0.5, end=i * 0.5 + 0.4) for i, w in enumerate("yes I agree . sure I will do it".replace(" .", "").split())]
    seg = Segment(id=1, start=0, end=4, text="yes I agree. sure I will do it", words=words)
    t = Transcript(segments=[seg], duration=4, model="x")
    turns = [(0.0, 1.45, 7), (1.5, 4.0, 3)]
    out = apply_speakers(t, turns)
    assert [(s.speaker, s.text) for s in out.segments] == [("Speaker 1", "yes I agree."), ("Speaker 2", "sure I will do it")]
    assert out.diarized and [s.id for s in out.segments] == [1, 2]


def test_long_segment_is_split_into_sentences():
    from meetscribe.stt import split_sentences

    text = "The project is due on 11th October. Lokesh will check it. Siddhu takes the viva."
    toks = text.split()
    words = [Word(word=t.strip("."), start=i * 0.6, end=i * 0.6 + 0.5) for i, t in enumerate(toks)]
    seg = Segment(id=1, start=0.0, end=9.0, text=text, words=words)
    out = split_sentences([seg])
    assert [s.text for s in out] == ["The project is due on 11th October.", "Lokesh will check it.", "Siddhu takes the viva."]
    assert out[1].start == words[7].start and out[0].start == 0.0 and out[-1].end == 9.0
