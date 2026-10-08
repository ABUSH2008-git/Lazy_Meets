"""Audio checks and conversion.

Everything goes through ffmpeg. If ffmpeg isn't installed system-wide we use the copy that
ships with the `imageio-ffmpeg` pip package, so Windows users don't need to install anything.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import AudioError, ConfigError

SAMPLE_RATE = 16000

ALLOWED_EXTS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".oga", ".opus", ".webm", ".flac",
    ".aac", ".wma", ".mov", ".mkv", ".amr", ".3gp", ".aiff", ".aif",
}
MAX_DURATION_S = 4 * 3600
MAX_CHUNK_BYTES = 24 * 1024 * 1024
MIN_DURATION_S = 1.0


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - depends on the machine
        raise ConfigError(
            "ffmpeg was not found.",
            hint="Run `pip install imageio-ffmpeg` (it ships ffmpeg) or install ffmpeg and add it to PATH.",
            detail=str(e),
        )


def validate_upload(filename: str, size_bytes: int, max_mb: int = 300) -> None:
    """Cheap checks before we even try to decode the file."""
    ext = Path(filename or "").suffix.lower()
    if not ext:
        raise AudioError(
            f"'{filename}' has no file extension, so I can't tell what kind of file it is.",
            hint="Upload an audio file such as .mp3, .wav, .m4a, .ogg, .webm or .flac.",
        )
    if ext not in ALLOWED_EXTS:
        nice = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTS))
        raise AudioError(
            f"'{ext}' files aren't supported. This app needs a meeting audio recording.",
            hint=f"Supported types: {nice}.",
        )
    if size_bytes <= 0:
        raise AudioError(f"'{filename}' is empty (0 bytes).", hint="Check that the recording saved properly and try again.")
    if size_bytes > max_mb * 1024 * 1024:
        raise AudioError(
            f"'{filename}' is {size_bytes / 1024 / 1024:.0f} MB, which is over the {max_mb} MB limit.",
            hint="Trim the recording or export it as a compressed format like .mp3 or .m4a.",
        )


_DECODE_HINTS = [
    (r"matches no streams|does not contain any stream|no audio", "'{f}' has no audio track in it."),
    (r"Invalid data found|could not find codec parameters|moov atom not found|Header missing|EBML header parsing failed",
     "'{f}' couldn't be decoded. It may be corrupted, cut off during upload, or not really an audio file."),
    (r"Permission denied", "'{f}' couldn't be opened (permission denied)."),
]


def _friendly_decode_error(stderr: str, filename: str) -> AudioError:
    for pattern, msg in _DECODE_HINTS:
        if re.search(pattern, stderr, re.I):
            return AudioError(msg.format(f=filename), hint="Try re-exporting the recording as .mp3 or .wav.", detail=stderr[-1500:])
    return AudioError(
        f"'{filename}' couldn't be decoded as audio. It may be corrupted, or not actually an audio file.",
        hint="Try re-exporting the recording as .mp3 or .wav.",
        detail=stderr[-1500:],
    )


def _hms(text: str) -> Optional[float]:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe_duration(path: Path) -> Optional[float]:
    """Length of the audio in seconds, read quickly without decoding everything. None if unknown."""
    try:
        proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, timeout=60)
        m = re.search(r"Duration:\s*(\d+:\d+:\d+(?:\.\d+)?)", proc.stderr.decode("utf-8", "replace"))
        if m:
            return _hms(m.group(1))
        # some formats (e.g. browser recordings) have no duration in the header: decode to nowhere and read the end time
        proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path), "-vn", "-f", "null", "-"],
                              capture_output=True, timeout=300)
        times = re.findall(r"time=\s*(\d+:\d+:\d+(?:\.\d+)?)", proc.stderr.decode("utf-8", "replace"))
        return _hms(times[-1]) if times else None
    except Exception:
        return None


def decode_to_pcm(path: Path, filename: str | None = None) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 samples in [-1, 1]."""
    filename = filename or Path(path).name
    cmd = [
        ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise AudioError(f"Decoding '{filename}' took too long.", hint="Try a shorter or smaller file.")
    pcm = proc.stdout
    if proc.returncode != 0 and len(pcm) < SAMPLE_RATE * 2:  # a few bad frames at the end are fine
        raise _friendly_decode_error(proc.stderr.decode("utf-8", "replace"), filename)
    if len(pcm) < 2:
        raise _friendly_decode_error(proc.stderr.decode("utf-8", "replace") or "no audio", filename)
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return samples


@dataclass
class LoudnessStats:
    peak_dbfs: float
    rms_dbfs: float
    speech_ratio: float  # share of 30 ms frames that are clearly above the noise floor


def loudness(samples: np.ndarray) -> LoudnessStats:
    if samples.size == 0:
        return LoudnessStats(-120.0, -120.0, 0.0)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    frame = int(0.03 * SAMPLE_RATE)
    n = samples.size // frame
    if n == 0:
        return LoudnessStats(_db(peak), _db(rms), 0.0)
    e = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    floor = np.percentile(e, 10)
    active = float(np.mean(e > max(floor * 4, 10 ** (-45 / 20))))
    return LoudnessStats(_db(peak), _db(rms), active)


def _db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def find_cut_points(samples: np.ndarray, target_s: float, search_s: float = 45.0) -> list[float]:
    """Pick places to split long audio, each in the quietest spot near every `target_s` seconds,
    so we never cut a word in half."""
    total = samples.size / SAMPLE_RATE
    if total <= target_s * 1.15:
        return []
    frame = int(0.05 * SAMPLE_RATE)  # 50 ms frames
    n = samples.size // frame
    energy = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    # smooth over ~0.5 s so we land in a real pause, not between two syllables
    k = 10
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    cuts: list[float] = []
    t = target_s
    while t < total - target_s * 0.3:
        lo = max(int((t - search_s) / 0.05), 1)
        hi = min(int((t + search_s) / 0.05), n - 1)
        if hi <= lo:
            break
        idx = lo + int(np.argmin(smooth[lo:hi]))
        cut = idx * 0.05
        if cuts and cut - cuts[-1] < target_s * 0.5:
            cut = t
        cuts.append(round(cut, 2))
        t = cut + target_s
    return cuts


@dataclass
class Chunk:
    path: Path
    offset: float
    duration: float


@dataclass
class PreparedAudio:
    source_name: str
    flac_path: Path
    playback_path: Path
    duration: float
    stats: LoudnessStats
    samples: np.ndarray = field(repr=False)
    chunks: list[Chunk] = field(default_factory=list)


def _encode(samples: np.ndarray, out: Path, codec_args: list[str]) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = [ffmpeg_exe(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "s16le",
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", *codec_args, str(out)]
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise AudioError("Couldn't convert the audio.", detail=proc.stderr.decode("utf-8", "replace")[-1500:])


def write_flac(samples: np.ndarray, out: Path) -> Path:
    _encode(samples, out, ["-c:a", "flac"])
    return out


def write_mp3(samples: np.ndarray, out: Path, bitrate: str = "48k") -> Path:
    _encode(samples, out, ["-c:a", "libmp3lame", "-b:a", bitrate])
    return out


def prepare(src: Path, work_dir: Path, source_name: str | None = None, chunk_minutes: float = 10.0) -> PreparedAudio:
    """Decode, sanity-check, convert to 16 kHz mono FLAC, and split long recordings."""
    source_name = source_name or Path(src).name
    work_dir.mkdir(parents=True, exist_ok=True)
    samples = decode_to_pcm(src, source_name)
    duration = samples.size / SAMPLE_RATE
    if duration < MIN_DURATION_S:
        raise AudioError(
            f"'{source_name}' is only {duration:.1f} seconds long, which is too short to be a meeting.",
            hint="Upload a longer recording.",
        )
    if duration > MAX_DURATION_S:
        raise AudioError(
            f"'{source_name}' is {duration / 3600:.1f} hours long. The limit is {MAX_DURATION_S // 3600} hours.",
            hint="Split the recording into parts and process them one by one.",
        )
    stats = loudness(samples)
    if stats.peak_dbfs < -50 or stats.rms_dbfs < -70:
        level = "has no sound at all" if stats.peak_dbfs < -100 else f"is nearly silent (loudest point {stats.peak_dbfs:.0f} dBFS)"
        raise AudioError(
            f"'{source_name}' {level}.",
            hint="Check that the microphone was on and the right file was uploaded.",
        )
    flac = write_flac(samples, work_dir / "audio_16k.flac")
    playback = write_mp3(samples, work_dir / "playback.mp3")
    prepared = PreparedAudio(source_name, flac, playback, duration, stats, samples)

    # keep every upload under the API's 25 MB file limit (free tier), whatever the compression ratio
    bytes_per_s = flac.stat().st_size / max(duration, 1.0)
    target_s = min(chunk_minutes * 60, 0.85 * MAX_CHUNK_BYTES / max(bytes_per_s, 1.0))
    cuts = find_cut_points(samples, target_s)
    if not cuts:
        prepared.chunks = [Chunk(flac, 0.0, duration)]
    else:
        bounds = [0.0, *cuts, duration]
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            part = samples[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
            p = write_flac(part, work_dir / f"chunk_{i:03d}.flac")
            prepared.chunks.append(Chunk(p, a, b - a))
    return prepared
