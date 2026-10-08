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


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible API the app knows how to talk to."""

    id: str
    name: str
    base_url: str
    key_url: str  # where people get a key
    key_hint: str  # what a key looks like
    stt: bool  # offers /audio/transcriptions with timestamps (Whisper-style)
    stt_models: tuple = ()
    refine_default: str = ""
    minutes_default: str = ""
    tpm: int = FREE_TIER_TPM  # tokens per request we plan for when the API doesn't tell us
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in [
        Provider("groq", "Groq", GROQ_BASE_URL, "https://console.groq.com/keys", "gsk_…", True,
                 ("whisper-large-v3", "whisper-large-v3-turbo"), "openai/gpt-oss-20b", "openai/gpt-oss-120b", FREE_TIER_TPM,
                 "Free tier. Runs Whisper and the GPT-OSS models. Rate limits are tight on long meetings."),
        Provider("openai", "OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "sk-…", True,
                 ("whisper-1",), "gpt-4.1-mini", "gpt-4.1", 120000,
                 "Paid. Whisper for speech-to-text and GPT models for the two language-model stages."),
        Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "https://aistudio.google.com/apikey", "AIza…", False, (), "gemini-2.5-flash-lite", "gemini-2.5-flash", 120000,
                 "Free tier available. Language models only; speech-to-text comes from another option."),
        Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "sk-or-…", False, (),
                 "openai/gpt-oss-20b", "openai/gpt-oss-120b", 60000,
                 "One key for hundreds of models, some free. Language models only."),
        Provider("mistral", "Mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys", "…", False, (),
                 "mistral-small-latest", "mistral-medium-latest", 60000, "Free tier available. Language models only."),
        Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "https://cloud.cerebras.ai", "csk-…", False, (),
                 "gpt-oss-120b", "gpt-oss-120b", 30000, "Very fast GPT-OSS. Language models only."),
        Provider("custom", "Other (OpenAI-compatible)", "", "", "…", True, ("whisper-large-v3",), "", "", FREE_TIER_TPM,
                 "Any OpenAI-compatible endpoint, e.g. Together, Fireworks, DeepInfra, a local vLLM or Ollama server."),
    ]
}


def provider_for_url(url: str) -> Provider:
    for p in PROVIDERS.values():
        if p.base_url and url.rstrip("/") == p.base_url.rstrip("/"):
            return p
    return PROVIDERS["custom"]


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
    # language models (stages 2 and 3, and Q&A)
    api_key: str = ""
    base_url: str = GROQ_BASE_URL
    # speech-to-text (stage 1). Empty means: same provider and key as the language models.
    stt_api_key: str = ""
    stt_base_url: str = ""
    default_tpm: int = FREE_TIER_TPM  # used when the API doesn't send rate-limit headers
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
    # free, preloaded key: daily audio allowance per signed-in user (0 = no limit)
    free_daily_minutes: float = 60.0
    quota_tz: str = "Asia/Kolkata"
    usage_file: Path = field(default_factory=lambda: ROOT / "runs" / "_usage.json")

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.api_key = _env("GROQ_API_KEY") or _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
        s.base_url = _env("LLM_BASE_URL", GROQ_BASE_URL)
        s.stt_api_key = _env("STT_API_KEY")
        s.stt_base_url = _env("STT_BASE_URL")
        s.default_tpm = provider_for_url(s.base_url).tpm
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
        try:
            s.free_daily_minutes = float(_env("FREE_DAILY_MINUTES", "60"))
        except ValueError:
            pass
        s.quota_tz = _env("QUOTA_TZ", "Asia/Kolkata")
        s.usage_file = Path(_env("USAGE_FILE")) if _env("USAGE_FILE") else s.runs_dir / "_usage.json"
        if _env("MEETSCRIBE_MODELS_DIR"):
            s.models_dir = Path(_env("MEETSCRIBE_MODELS_DIR"))
        return s

    @property
    def stt_key(self) -> str:
        return self.stt_api_key or self.api_key

    @property
    def stt_url(self) -> str:
        return self.stt_base_url or self.base_url

    def public_dict(self) -> dict:
        """Settings without the API keys, safe to save next to a run."""
        d = asdict(self)
        d.pop("api_key", None)
        d.pop("stt_api_key", None)
        d["runs_dir"] = str(self.runs_dir)
        d["models_dir"] = str(self.models_dir)
        d["usage_file"] = str(self.usage_file)
        return d
