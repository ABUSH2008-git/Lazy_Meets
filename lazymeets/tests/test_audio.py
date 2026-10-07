import subprocess

import numpy as np
import pytest

from meetscribe.audio import SAMPLE_RATE, ffmpeg_exe, find_cut_points, prepare, validate_upload
from meetscribe.errors import AudioError


def make(path, args):
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", *args, str(path)], check=True)
    return path


def test_rejects_wrong_type_and_empty():
    with pytest.raises(AudioError, match="aren't supported"):
        validate_upload("notes.txt", 100)
    with pytest.raises(AudioError, match="empty"):
        validate_upload("meeting.mp3", 0)
    with pytest.raises(AudioError, match="limit"):
        validate_upload("meeting.mp3", 400 * 1024 * 1024, max_mb=300)
    validate_upload("meeting.M4A", 1000)


def test_unreadable_silent_short_and_video_only(tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(np.random.default_rng(0).bytes(40000))
    with pytest.raises(AudioError, match="decoded"):
        prepare(bad, tmp_path / "w1")
    silent = make(tmp_path / "silent.wav", ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "4"])
    with pytest.raises(AudioError, match="no sound|silent"):
        prepare(silent, tmp_path / "w2")
    short = make(tmp_path / "short.wav", ["-f", "lavfi", "-i", "sine=frequency=300:duration=0.4"])
    with pytest.raises(AudioError, match="too short"):
        prepare(short, tmp_path / "w3")
    video = make(tmp_path / "v.mp4", ["-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=5", "-c:v", "mpeg4"])
    with pytest.raises(AudioError, match="no audio track"):
        prepare(video, tmp_path / "w4")


def test_prepare_demo(demo_audio, tmp_path):
    a = prepare(demo_audio, tmp_path / "w")
    assert 120 < a.duration < 130
    assert a.flac_path.exists() and a.playback_path.exists()
    assert len(a.chunks) == 1


def test_cut_points_land_in_pauses():
    rng = np.random.default_rng(1)
    sr = SAMPLE_RATE
    speech = lambda s: (rng.normal(0, 0.2, int(s * sr))).astype(np.float32)
    pause = lambda s: np.zeros(int(s * sr), dtype=np.float32)
    x = np.concatenate([speech(50), pause(1.0), speech(30), pause(1.0), speech(50)])  # pauses at 50s and 81s
    cuts = find_cut_points(x, target_s=60, search_s=30)
    assert cuts, "expected at least one cut"
    assert min(abs(cuts[0] - 50.5), abs(cuts[0] - 81.5)) < 1.0
