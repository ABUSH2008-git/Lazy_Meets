"""Make the synthetic demo meeting (samples/demo_meeting.mp3) from samples/demo_meeting_script.json.

This is a dev helper. It uses the offline Kokoro TTS voices from sherpa-onnx so we have a
shareable multi-speaker test recording. A real recording of people talking is better for the
final submission (see samples/RECORDING_SCRIPT.md), but this one is handy for quick tests.

Download the voices first (about 320 MB):
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2

Usage:
  python scripts/make_demo_audio.py --kokoro path/to/kokoro-en-v0_19 --out samples/demo_meeting.mp3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetscribe.audio import ffmpeg_exe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kokoro", required=True, help="folder with model.onnx, voices.bin, tokens.txt, espeak-ng-data")
    ap.add_argument("--script", default=str(ROOT / "samples" / "demo_meeting_script.json"))
    ap.add_argument("--out", default=str(ROOT / "samples" / "demo_meeting.mp3"))
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    import sherpa_onnx

    k = Path(args.kokoro)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(k / "model.onnx"),
                voices=str(k / "voices.bin"),
                tokens=str(k / "tokens.txt"),
                data_dir=str(k / "espeak-ng-data"),
            ),
            num_threads=2,
        ),
        max_num_sentences=1,
    )
    tts = sherpa_onnx.OfflineTts(cfg)

    script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    rng = np.random.default_rng(7)
    pieces: list[np.ndarray] = []
    sr = None
    prev = None
    for turn in script["turns"]:
        sid = script["speakers"][turn["speaker"]]["kokoro_sid"]
        audio = tts.generate(turn["text"], sid=sid, speed=args.speed)
        sr = audio.sample_rate
        samples = np.asarray(audio.samples, dtype=np.float32)
        # shorter gap when the same person keeps talking, a bit longer between people
        gap = rng.uniform(0.25, 0.4) if prev == turn["speaker"] else rng.uniform(0.45, 0.9)
        pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
        pieces.append(samples)
        prev = turn["speaker"]
    pieces.append(np.zeros(int(0.8 * sr), dtype=np.float32))
    mix = np.concatenate(pieces)
    # a little room noise so it is not studio-silent between turns
    mix = mix + rng.normal(0, 0.0015, size=mix.shape).astype(np.float32)
    mix = np.clip(mix / max(1e-6, np.abs(mix).max()) * 0.85, -1, 1)
    pcm = (mix * 32767).astype("<i2").tobytes()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
           "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "64k", str(out)]
    subprocess.run(cmd, input=pcm, check=True)
    print(f"wrote {out} ({len(mix) / sr:.1f} s)")


if __name__ == "__main__":
    main()
