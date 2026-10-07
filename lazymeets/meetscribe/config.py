"""Settings for the app. Everything can come from environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # optional, only used when a .env file exists
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
SAMPLES_DIR = ROOT / "samples"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Stage 1: speech-to-text. Stage 2: refinement (LLM #1). Stage 3: minutes (LLM #2).
DEFAULT_STT_MODEL = "whisper-large-v3"
DEFAULT_REFINE_MODEL = "openai/gpt-oss-20b"
DEFAULT_MINUTES_MODEL = "openai/gpt-oss-120b"

STT_MODEL_CHOICES = ["whisper-large-v3", "whisper-large-v3-turbo"]
LLM_MODEL_CHOICES = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
REASONING_CHOICES = ["low", "medium", "high"]

# Groq free tier allows 8,000 tokens per minute on the GPT-OSS models. One request
# (prompt + max output) has to fit inside that, so we plan requests around this number.
# The real limit is read from the API's response headers once we have one.
FREE_TIER_TPM = 8000


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    api_key: str = ""
    base_url: str = GROQ_BASE_URL
    stt_backend: str = "api"  # "api" (Whisper via Groq) or "local" (offline, sherpa-onnx)
    stt_model: str = DEFAULT_STT_MODEL
    refine_model: str = DEFAULT_REFINE_MODEL
    minutes_model: str = DEFAULT_MINUTES_MODEL
    refine_reasoning: str = "medium"
    minutes_reasoning: str = "medium"
    temperature: float = 0.3
    tpm_budget: int = 0  # 0 = detect from the API (falls back to FREE_TIER_TPM)
    diarize: bool = False
    num_speakers: int = 0  # 0 = detect automatically
    language: str = "en"
    max_upload_mb: int = 300
    chunk_minutes: float = 10.0
    persist_runs: bool = True
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")
    models_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "meetscribe" / "models")
    request_timeout: float = 180.0

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.api_key = _env("GROQ_API_KEY") or _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
        s.base_url = _env("LLM_BASE_URL", GROQ_BASE_URL)
        s.stt_backend = _env("STT_BACKEND", "api")
        s.stt_model = _env("STT_MODEL", DEFAULT_STT_MODEL)
        s.refine_model = _env("REFINE_MODEL", DEFAULT_REFINE_MODEL)
        s.minutes_model = _env("MINUTES_MODEL", DEFAULT_MINUTES_MODEL)
        s.refine_reasoning = _env("REFINE_REASONING", "medium")
        s.minutes_reasoning = _env("MINUTES_REASONING", "medium")
        s.tpm_budget = _env_int("TPM_BUDGET", 0)
        s.diarize = _env_bool("DIARIZATION", False)
        s.max_upload_mb = _env_int("MAX_UPLOAD_MB", 300)
        s.persist_runs = _env_bool("PERSIST_RUNS", True)
        if _env("RUNS_DIR"):
            s.runs_dir = Path(_env("RUNS_DIR"))
        if _env("MEETSCRIBE_MODELS_DIR"):
            s.models_dir = Path(_env("MEETSCRIBE_MODELS_DIR"))
        return s

    def public_dict(self) -> dict:
        """Settings without the API key, safe to save next to a run."""
        d = asdict(self)
        d.pop("api_key", None)
        d["runs_dir"] = str(self.runs_dir)
        d["models_dir"] = str(self.models_dir)
        return d
