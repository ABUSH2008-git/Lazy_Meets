"""Download the offline models ahead of time (used by the Dockerfile, optional locally).

  python scripts/download_models.py --diarization      # ~47 MB, speaker detection
  python scripts/download_models.py --offline-stt      # ~250 MB, Moonshine offline transcription
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meetscribe import offline_models  # noqa: E402
from meetscribe.config import Settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--diarization", action="store_true")
ap.add_argument("--offline-stt", action="store_true")
a = ap.parse_args()
s = Settings.from_env()
if a.diarization:
    print(offline_models.ensure(offline_models.DIARIZATION, s.models_dir, print))
if a.offline_stt:
    print(offline_models.ensure(offline_models.LOCAL_STT, s.models_dir, print))
