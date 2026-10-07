"""Downloads the small offline models used for speaker diarization and offline transcription.

They come from the sherpa-onnx GitHub releases and are cached in ~/.cache/meetscribe/models
(or MEETSCRIBE_MODELS_DIR). Nothing here needs a GPU.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .errors import ConfigError

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

DIARIZATION = {
    "segmentation": (
        f"{BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    ),
    "embedding": (
        f"{BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        "nemo_en_titanet_small.onnx",
    ),
}

LOCAL_STT = {
    "vad": (f"{BASE}/asr-models/silero_vad.onnx", "silero_vad.onnx"),
    "moonshine": (
        f"{BASE}/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
        "sherpa-onnx-moonshine-base-en-int8/tokens.txt",
    ),
}


def _download(url: str, dest_dir: Path, notify: Callable[[str], None]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    notify(f"Downloading {name} (one-time)…")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_file = Path(tmp) / name
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp_file, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        except Exception as e:
            raise ConfigError(
                f"Couldn't download the offline model {name}.",
                hint="Check your internet connection, or turn this option off in the sidebar.",
                detail=f"{url}: {e}",
            )
        if name.endswith(".tar.bz2"):
            with tarfile.open(tmp_file, "r:bz2") as t:
                try:
                    t.extractall(dest_dir, filter="data")
                except TypeError:  # Python without extraction filters
                    t.extractall(dest_dir)
        else:
            shutil.move(str(tmp_file), dest_dir / name)


def ensure(models: dict[str, tuple[str, str]], models_dir: Path, notify: Optional[Callable[[str], None]] = None) -> dict[str, Path]:
    notify = notify or (lambda m: None)
    out: dict[str, Path] = {}
    for key, (url, rel) in models.items():
        p = models_dir / rel
        if not p.exists():
            _download(url, models_dir, notify)
        if not p.exists():
            raise ConfigError(f"Offline model file missing after download: {p}")
        out[key] = p
    return out


def available(models: dict[str, tuple[str, str]], models_dir: Path) -> bool:
    return all((models_dir / rel).exists() for _, rel in models.values())
