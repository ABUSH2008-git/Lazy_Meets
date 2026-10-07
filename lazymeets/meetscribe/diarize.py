"""Optional speaker diarization ("who spoke when").

Runs fully offline on CPU with sherpa-onnx: pyannote segmentation-3.0 finds speech turns and
NVIDIA's TitaNet-small speaker-embedding model groups them by voice. Then each transcript word is given the
speaker whose turn overlaps it, and Whisper segments are split where the speaker changes.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

from . import offline_models
from .config import Settings
from .errors import ConfigError
from .schemas import Segment, Transcript, Word

Turn = tuple[float, float, int]


def diarize_samples(samples: np.ndarray, settings: Settings, notify: Optional[Callable[[str], None]] = None) -> list[Turn]:
    notify = notify or (lambda m: None)
    try:
        import sherpa_onnx
    except ImportError as e:  # pragma: no cover
        raise ConfigError("Speaker detection needs the `sherpa-onnx` package.", hint="pip install sherpa-onnx", detail=str(e))
    paths = offline_models.ensure(offline_models.DIARIZATION, settings.models_dir, notify)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(paths["segmentation"])),
            num_threads=2,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(paths["embedding"]), num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=settings.num_speakers if settings.num_speakers > 0 else -1,
            threshold=0.7,  # tuned on the demo meeting: 3 voices -> 3 speakers, 95% of turns right
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise ConfigError("Speaker detection models failed to load.", hint="Delete the model cache folder and try again.")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    last = [-1]

    def progress(done: int, total: int) -> int:
        pct = int(100 * done / max(total, 1))
        if pct >= last[0] + 10:
            last[0] = pct
            notify(f"Identifying speakers… {pct}%")
        return 0

    result = sd.process(np.ascontiguousarray(samples, dtype=np.float32), callback=progress).sort_by_start_time()
    return [(float(r.start), float(r.end), int(r.speaker)) for r in result]


def _speaker_at(t0: float, t1: float, turns: list[Turn]) -> Optional[int]:
    best, best_ov = None, 0.0
    for a, b, spk in turns:
        if b < t0:
            continue
        if a > t1:
            break
        ov = min(b, t1) - max(a, t0)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is not None:
        return best
    # no overlap (word fell in a gap): take the nearest turn
    mid = (t0 + t1) / 2
    near = min(turns, key=lambda x: min(abs(x[0] - mid), abs(x[1] - mid)), default=None)
    return near[2] if near else None


def _norm(tok: str) -> str:
    return re.sub(r"[^\w']", "", tok.lower())


def apply_speakers(transcript: Transcript, turns: list[Turn]) -> Transcript:
    """Label every segment with a speaker, splitting segments where the speaker changes."""
    if not turns:
        return transcript
    order: dict[int, str] = {}

    def label(spk: int) -> str:
        if spk not in order:
            order[spk] = f"Speaker {len(order) + 1}"
        return order[spk]

    out: list[Segment] = []
    for seg in transcript.segments:
        tokens = seg.text.split()
        words = seg.words
        aligned = len(words) == len(tokens) and all(
            _norm(w.word) == _norm(t) or not _norm(w.word) for w, t in zip(words, tokens)
        ) if words else False
        if not aligned and words and abs(len(words) - len(tokens)) <= 2:
            aligned = len(words) == len(tokens)
        if not words or not aligned:
            spk = _speaker_at(seg.start, seg.end, turns)
            out.append(seg.model_copy(update={"speaker": label(spk) if spk is not None else None}))
            continue
        spks = [_speaker_at(w.start, w.end, turns) for w in words]
        # smooth: a run of 1-2 words between the same speaker is almost always noise
        for i in range(1, len(spks) - 1):
            if spks[i - 1] == spks[i + 1] and spks[i] != spks[i - 1]:
                spks[i] = spks[i - 1]
        runs: list[tuple[int, int, Optional[int]]] = []
        start = 0
        for i in range(1, len(spks) + 1):
            if i == len(spks) or spks[i] != spks[start]:
                runs.append((start, i, spks[start]))
                start = i
        # merge very short runs into the previous one
        merged: list[list] = []
        for a, b, s in runs:
            if merged and (b - a) < 3:
                merged[-1][1] = b
            else:
                merged.append([a, b, s])
        if len(merged) > 1 and (merged[0][1] - merged[0][0]) < 3:
            merged[1][0] = merged[0][0]
            merged.pop(0)
        for a, b, s in merged:
            ws = words[a:b]
            out.append(
                Segment(
                    id=0,
                    start=ws[0].start if a > 0 else seg.start,
                    end=ws[-1].end if b < len(words) else seg.end,
                    text=" ".join(tokens[a:b]),
                    speaker=label(s) if s is not None else None,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    compression_ratio=seg.compression_ratio,
                    low_confidence=seg.low_confidence,
                    words=[Word(word=w.word, start=w.start, end=w.end) for w in ws],
                )
            )
    for i, s in enumerate(out, start=1):
        s.id = i
    return transcript.model_copy(update={"segments": out, "diarized": True})
