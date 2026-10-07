"""Stage 1: speech-to-text.

Default: Whisper large-v3 through the Groq API, chunk by chunk, with segment timestamps
and confidence values. Optional offline mode: Moonshine (sherpa-onnx) on CPU.

After transcription we drop the classic Whisper hallucinations (text invented over
silence like "Thanks for watching") and mark segments the model wasn't sure about.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np
from rapidfuzz import fuzz

from . import offline_models
from .audio import SAMPLE_RATE, PreparedAudio
from .config import Settings
from .errors import ConfigError, NoSpeechError
from .llm import ModelClient, est_tokens
from .schemas import DroppedSegment, MeetingContext, Segment, Transcript, Word

Notify = Callable[[str], None]

WHISPER_PROMPT_MAX_TOKENS = 160  # API limit is 224 tokens; our estimate is rough, so stay well under

_HALLUCINATIONS = [
    r"thanks? (you )?for watching",
    r"please (like and )?subscribe",
    r"like and subscribe",
    r"subtitles? (by|created by)",
    r"transcri(bed|ption) by",
    r"amara\.org",
    r"www\.[a-z]",
    r"see you (in the )?next (video|time)",
    r"^\W*(you|bye\.?)\W*$",
]


def build_whisper_prompt(ctx: MeetingContext) -> str:
    """Whisper's `prompt` is a spelling/style hint: names and terms we expect to hear."""
    head = "Meeting recording"
    if ctx.description:
        head += f" ({ctx.description.strip()[:120]})"
    head += "."
    parts = [head]
    if ctx.participants:
        parts.append("Participants: " + ", ".join(ctx.participants[:15]) + ".")
    terms = ctx.all_terms()
    prompt = " ".join(parts)
    if terms:
        kept: list[str] = []
        for t in terms:
            trial = prompt + " Terms: " + ", ".join(kept + [t]) + "."
            if est_tokens(trial) > WHISPER_PROMPT_MAX_TOKENS:
                break
            kept.append(t)
        if kept:
            prompt += " Terms: " + ", ".join(kept) + "."
    return prompt


def _frame_energy(samples: np.ndarray) -> tuple[np.ndarray, float]:
    frame = int(0.03 * SAMPLE_RATE)
    n = samples.size // frame
    if n == 0:
        return np.zeros(1), 0.0
    e = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    return e, float(np.percentile(e, 10))


def _is_audible(start: float, end: float, energy: np.ndarray, floor: float) -> bool:
    a, b = int(start / 0.03), max(int(end / 0.03), int(start / 0.03) + 1)
    seg = energy[a:b]
    if seg.size == 0:
        return True
    loud = float(np.percentile(seg, 90))
    return loud > max(floor * 3.0, 10 ** (-52 / 20))


def _clean(raw: list[Segment], samples: np.ndarray, prompt: str) -> tuple[list[Segment], list[DroppedSegment]]:
    energy, floor = _frame_energy(samples)
    kept: list[Segment] = []
    dropped: list[DroppedSegment] = []
    for s in raw:
        text = s.text.strip()
        reason = None
        nsp = s.no_speech_prob if s.no_speech_prob is not None else 0.0
        alp = s.avg_logprob if s.avg_logprob is not None else 0.0
        if not text:
            reason = "empty"
        elif (nsp > 0.6 and alp < -0.8) or (nsp > 0.9 and alp < -0.4):
            reason = "model reported no speech here"
        elif any(re.search(p, text, re.I) for p in _HALLUCINATIONS) and (nsp > 0.3 or alp < -0.5 or not _is_audible(s.start, s.end, energy, floor)):
            reason = "typical Whisper hallucination over silence"
        elif prompt and len(text) > 20 and fuzz.partial_ratio(text.lower(), prompt.lower()) > 92:
            reason = "repeated the spelling hint instead of speech"
        elif not _is_audible(s.start, s.end, energy, floor) and (nsp > 0.2 or alp < -0.6):
            reason = "no audible speech at this time"
        if reason:
            dropped.append(DroppedSegment(start=s.start, end=s.end, text=text, reason=reason))
            continue
        s.text = text
        s.low_confidence = alp < -0.65 or (s.compression_ratio or 0) > 2.4 or nsp > 0.5
        kept.append(s)
    return kept, dropped


def _segments_from_api(data: dict, offset: float) -> list[Segment]:
    words = [
        Word(word=str(w.get("word", "")).strip(), start=float(w.get("start", 0)) + offset, end=float(w.get("end", 0)) + offset)
        for w in (data.get("words") or [])
        if str(w.get("word", "")).strip()
    ]
    segs: list[Segment] = []
    raw = data.get("segments") or []
    if not raw and (data.get("text") or "").strip():
        raw = [{"start": 0.0, "end": float(data.get("duration") or 0), "text": data["text"]}]
    wi = 0
    for r in raw:
        start = float(r.get("start", 0)) + offset
        end = float(r.get("end", 0)) + offset
        seg_words: list[Word] = []
        while wi < len(words) and (words[wi].start + words[wi].end) / 2 <= end + 0.05:
            if (words[wi].start + words[wi].end) / 2 >= start - 0.05:
                seg_words.append(words[wi])
            wi += 1
        segs.append(
            Segment(
                id=0,
                start=start,
                end=end,
                text=str(r.get("text", "")),
                avg_logprob=r.get("avg_logprob"),
                no_speech_prob=r.get("no_speech_prob"),
                compression_ratio=r.get("compression_ratio"),
                words=seg_words,
            )
        )
    return segs


def transcribe_api(audio: PreparedAudio, ctx: MeetingContext, settings: Settings, client: ModelClient, notify: Notify) -> Transcript:
    prompt = build_whisper_prompt(ctx)
    all_segs: list[Segment] = []
    n = len(audio.chunks)
    for i, ch in enumerate(audio.chunks, start=1):
        notify(f"Transcribing with {settings.stt_model}" + (f" (part {i} of {n})" if n > 1 else "") + "…")
        data = client.transcribe(ch.path, settings.stt_model, prompt=prompt, language=settings.language)
        all_segs.extend(_segments_from_api(data, ch.offset))
    return _finish(all_segs, audio, settings.stt_model, prompt)


def transcribe_local(audio: PreparedAudio, settings: Settings, notify: Notify) -> Transcript:
    try:
        import sherpa_onnx
    except ImportError as e:  # pragma: no cover
        raise ConfigError("Offline transcription needs the `sherpa-onnx` package.", hint="pip install sherpa-onnx", detail=str(e))
    paths = offline_models.ensure(offline_models.LOCAL_STT, settings.models_dir, notify)
    mdir = paths["moonshine"].parent
    rec = sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=str(mdir / "preprocess.onnx"),
        encoder=str(mdir / "encode.int8.onnx"),
        uncached_decoder=str(mdir / "uncached_decode.int8.onnx"),
        cached_decoder=str(mdir / "cached_decode.int8.onnx"),
        tokens=str(mdir / "tokens.txt"),
        num_threads=2,
    )
    vcfg = sherpa_onnx.VadModelConfig()
    vcfg.silero_vad.model = str(paths["vad"])
    vcfg.silero_vad.min_silence_duration = 0.35
    vcfg.silero_vad.min_speech_duration = 0.25
    vcfg.silero_vad.max_speech_duration = 18
    vcfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60)
    samples = audio.samples
    win = vcfg.silero_vad.window_size
    pieces: list[tuple[float, np.ndarray]] = []
    for i in range(0, samples.size, win):
        vad.accept_waveform(samples[i: i + win])
        while not vad.empty():
            pieces.append((vad.front.start / SAMPLE_RATE, np.array(vad.front.samples, dtype=np.float32)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        pieces.append((vad.front.start / SAMPLE_RATE, np.array(vad.front.samples, dtype=np.float32)))
        vad.pop()
    segs: list[Segment] = []
    for k, (start, chunk) in enumerate(pieces, start=1):
        if k % 10 == 1:
            notify(f"Transcribing offline… {int(100 * k / max(len(pieces), 1))}%")
        st = rec.create_stream()
        st.accept_waveform(SAMPLE_RATE, chunk)
        rec.decode_stream(st)
        text = st.result.text.strip()
        if text:
            segs.append(Segment(id=0, start=start, end=start + chunk.size / SAMPLE_RATE, text=text))
    return _finish(segs, audio, "moonshine-base-en (offline)", "")


def split_sentences(segs: list[Segment], min_len_s: float = 6.0) -> list[Segment]:
    """Whisper sometimes returns one long segment for several sentences (a short clip can come back
    as a single segment at 00:00). Split those at sentence ends using the word timings, so every
    line, quote and timestamp points at the right moment."""
    out: list[Segment] = []
    for seg in segs:
        tokens = seg.text.split()
        if seg.end - seg.start < min_len_s or not seg.words or len(seg.words) != len(tokens):
            out.append(seg)
            continue
        start = 0
        for i, tok in enumerate(tokens):
            last = i == len(tokens) - 1
            if last or (tok[-1:] in ".?!" and i - start >= 2):
                ws = seg.words[start: i + 1]
                out.append(seg.model_copy(update={
                    "text": " ".join(tokens[start: i + 1]),
                    "start": seg.start if start == 0 else ws[0].start,
                    "end": seg.end if last else ws[-1].end,
                    "words": ws,
                }))
                start = i + 1
    return out


def _finish(segs: list[Segment], audio: PreparedAudio, model: str, prompt: str) -> Transcript:
    kept, dropped = _clean(segs, audio.samples, prompt)
    kept = split_sentences(kept)
    if not kept:
        raise NoSpeechError(
            "No speech was found in this recording.",
            hint="Make sure the file actually contains people talking (in English) and the volume isn't too low.",
        )
    for i, s in enumerate(kept, start=1):
        s.id = i
    return Transcript(
        segments=kept,
        duration=audio.duration,
        model=model,
        whisper_prompt=prompt,
        dropped=dropped,
    )


def transcribe(audio: PreparedAudio, ctx: MeetingContext, settings: Settings, client: Optional[ModelClient], notify: Notify) -> Transcript:
    if settings.stt_backend == "local":
        return transcribe_local(audio, settings, notify)
    if client is None:
        raise ConfigError("No API client available for transcription.")
    return transcribe_api(audio, ctx, settings, client, notify)
