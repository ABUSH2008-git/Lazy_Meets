"""One small client for every model call (speech-to-text and both language models).

It talks to any OpenAI-compatible API (Groq by default) and takes care of the annoying parts:
- retries on rate limits and flaky network, telling the UI how long it's waiting
- pacing requests so we stay under the tokens-per-minute limit
- structured JSON output: strict schema first, then softer modes if a model doesn't support it
- turning provider errors into messages a user can act on
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import openai
from openai import OpenAI

from .config import FREE_TIER_TPM, Settings
from .errors import APIError, ConfigError, ModelOutputError, RateLimitError, RequestTooLargeError

Notify = Callable[[str], None]

MAX_AUTO_WAIT_S = 95  # longer waits mean a daily/hourly quota, so we stop and tell the user


def est_tokens(text: str) -> int:
    """Rough token count. English is ~4 chars per token; we round up to stay safe."""
    return int(len(text) / 3.4) + 8


def _parse_duration(s: str | None) -> Optional[float]:
    """'7.66s', '2m59.56s', '1h2m', '120ms' -> seconds."""
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s):
        found = True
        v = float(num)
        total += {"ms": v / 1000, "h": v * 3600, "m": v * 60, "s": v}[unit]
    return total if found else None


def _retry_after(err: openai.APIStatusError) -> Optional[float]:
    try:
        h = err.response.headers
        v = _parse_duration(h.get("retry-after"))
        if v is not None:
            return v
    except Exception:
        pass
    m = re.search(r"try again in ([\dhms.]+)", str(err), re.I)
    return _parse_duration(m.group(1)) if m else None


@dataclass
class CallRecord:
    stage: str
    model: str
    kind: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    attempts: int = 1
    mode: str = ""


@dataclass
class _Limits:
    tpm: Optional[int] = None
    remaining: Optional[int] = None
    reset_at: float = 0.0


@dataclass
class _Caps:
    json_mode: str = "strict"  # strict -> loose -> json_object
    reasoning_effort: bool = True
    include_reasoning: bool = True


class ModelClient:
    def __init__(self, settings: Settings, notify: Notify | None = None):
        if not settings.api_key and settings.stt_backend == "api":
            raise ConfigError(
                "No API key set.",
                hint="Add GROQ_API_KEY to your .env file (free key at console.groq.com), or paste it in the sidebar.",
            )
        self.settings = settings
        self.notify: Notify = notify or (lambda msg: None)
        self._client = OpenAI(
            api_key=settings.api_key or "none",
            base_url=settings.base_url,
            max_retries=0,
            timeout=settings.request_timeout,
        )
        self.calls: list[CallRecord] = []
        self._limits: dict[str, _Limits] = {}
        self._caps: dict[str, _Caps] = {}

    # ------------------------------------------------------------------ helpers

    def usage_summary(self) -> list[dict]:
        return [asdict(c) for c in self.calls]

    def budget_for(self, model: str) -> int:
        """How many tokens one request (prompt + max output) may use."""
        if self.settings.tpm_budget:
            return self.settings.tpm_budget
        lim = self._limits.get(model)
        if lim and lim.tpm:
            return lim.tpm
        # same account tier for every model, so any known limit is a good guess
        known = [l.tpm for l in self._limits.values() if l.tpm]
        return min(known) if known else FREE_TIER_TPM

    def _record_headers(self, model: str, headers: Any) -> None:
        try:
            lim = self._limits.setdefault(model, _Limits())
            tpm = headers.get("x-ratelimit-limit-tokens")
            rem = headers.get("x-ratelimit-remaining-tokens")
            reset = _parse_duration(headers.get("x-ratelimit-reset-tokens"))
            if tpm:
                lim.tpm = int(float(tpm))
            if rem:
                lim.remaining = int(float(rem))
            if reset is not None:
                lim.reset_at = time.time() + reset
        except Exception:
            pass

    def _pace(self, model: str, need: int) -> None:
        """Wait before a request if the last response said we're out of tokens for this minute."""
        lim = self._limits.get(model)
        if not lim or lim.remaining is None:
            return
        wait = lim.reset_at - time.time()
        if lim.remaining < need and 0 < wait <= MAX_AUTO_WAIT_S:
            self.notify(f"Pacing requests to stay under the rate limit for {model}. Waiting {wait:.0f}s…")
            time.sleep(wait + 0.5)
            lim.remaining = None

    def _call(self, fn: Callable[[], Any], what: str, model: str) -> tuple[Any, int]:
        """Run an API call with retries. Returns (result, attempts)."""
        attempt = 0
        net_failures = 0
        while True:
            attempt += 1
            try:
                return fn(), attempt
            except openai.AuthenticationError as e:
                raise ConfigError(
                    "The API key was rejected by the provider.",
                    hint="Check GROQ_API_KEY (no extra spaces) or create a new key at console.groq.com/keys.",
                    detail=str(e),
                )
            except openai.PermissionDeniedError as e:
                raise ConfigError(
                    f"Your API key isn't allowed to use '{model}'.",
                    hint="Pick a different model in the sidebar, or enable it in the provider console (model permissions).",
                    detail=str(e),
                )
            except openai.NotFoundError as e:
                raise ConfigError(
                    f"Model '{model}' wasn't found at {self.settings.base_url}.",
                    hint="It may have been retired. Choose another model in the sidebar.",
                    detail=str(e),
                )
            except openai.RateLimitError as e:
                msg = str(e)
                if "too large" in msg.lower():
                    raise RequestTooLargeError(f"The request to {model} is bigger than the per-minute token limit.", detail=msg)
                wait = _retry_after(e)
                daily = re.search(r"per day|TPD|RPD|ASD", msg)
                if wait is None:
                    wait = min(5 * attempt, 30)
                if daily or wait > MAX_AUTO_WAIT_S or attempt > 8:
                    raise RateLimitError(
                        f"Rate limit reached for {model} while {what}.",
                        hint=(
                            "You've used this model's daily allowance on the free tier. Try again later, pick a different "
                            "model in the sidebar, or use a key on Groq's Developer tier."
                            if daily
                            else f"The provider asked us to wait about {wait / 60:.0f} minutes. Try again in a bit."
                        ),
                        detail=msg,
                    )
                self.notify(f"Rate limit reached for {model}. Waiting {wait:.0f}s, then retrying ({what})…")
                time.sleep(wait + 0.5)
            except openai.APIStatusError as e:
                if e.status_code == 413:
                    raise RequestTooLargeError(f"The request to {model} was too large.", detail=str(e))
                if e.status_code >= 500 and net_failures < 3:
                    net_failures += 1
                    wait = 2 ** net_failures
                    self.notify(f"The provider had a hiccup ({e.status_code}). Retrying in {wait}s…")
                    time.sleep(wait)
                    continue
                if e.status_code >= 500:
                    raise APIError(
                        f"The model provider is having problems right now (error {e.status_code}) while {what}.",
                        hint="This is on the provider's side. Wait a minute and retry this step. Earlier steps are kept.",
                        detail=str(e),
                    )
                raise
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if net_failures < 3:
                    net_failures += 1
                    wait = 2 ** net_failures
                    self.notify(f"Network problem while {what}. Retrying in {wait}s…")
                    time.sleep(wait)
                    continue
                raise APIError(
                    f"Couldn't reach {self.settings.base_url} while {what}.",
                    hint="Check your internet connection and try again.",
                    detail=str(e),
                )

    # ------------------------------------------------------------------ model list / preflight

    def available_models(self) -> list[str]:
        res, _ = self._call(lambda: self._client.models.list(), "checking the API key", "models")
        return sorted(m.id for m in res.data)

    def preflight(self, needed: dict[str, str]) -> None:
        """Fail fast (before a long transcription) if the key or a model name is wrong.
        `needed` maps a role ("speech-to-text", ...) to a model id."""
        try:
            available = set(self.available_models())
        except ConfigError:
            raise
        except Exception:
            return  # some providers don't implement /models; we'll find out on first use
        missing = [(role, m) for role, m in needed.items() if m and m not in available]
        if missing:
            role, m = missing[0]
            close = [a for a in sorted(available) if any(p in a for p in re.split(r"[/-]", m) if len(p) > 3)][:6]
            raise ConfigError(
                f"The {role} model '{m}' isn't available with this API key.",
                hint=("Close matches: " + ", ".join(close)) if close else "Pick another model in the sidebar.",
            )

    # ------------------------------------------------------------------ speech-to-text

    def transcribe(self, path: Path, model: str, prompt: str = "", language: str = "en") -> dict:
        def go():
            with open(path, "rb") as f:
                kwargs: dict[str, Any] = dict(
                    file=(Path(path).name, f.read()),
                    model=model,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    temperature=0.0,
                )
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            return self._client.audio.transcriptions.create(**kwargs)

        t0 = time.time()
        try:
            try:
                res, attempts = self._call(go, "transcribing audio", model)
            except openai.BadRequestError as e:
                if not prompt or "prompt" not in str(e).lower():
                    raise
                self.notify("The spelling hint was rejected; transcribing without it…")
                prompt = ""
                res, attempts = self._call(go, "transcribing audio", model)
        except openai.BadRequestError as e:
            raise APIError(
                "The speech-to-text service couldn't process this audio.",
                hint="The file may be damaged. Try re-exporting it as .mp3 or .wav.",
                detail=str(e),
            )
        self.calls.append(CallRecord("transcribe", model, "stt", time.time() - t0, attempts=attempts))
        data = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        return data

    # ------------------------------------------------------------------ structured chat

    def chat_json(
        self,
        *,
        stage: str,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict,
        max_tokens: int,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        caps = self._caps.setdefault(model, _Caps())
        modes = ["strict", "loose", "json_object"]
        mode_i = modes.index(caps.json_mode)
        temperature = self.settings.temperature if temperature is None else temperature
        bad_json_retries = 0
        truncation_retries = 0
        effort = reasoning_effort

        while True:
            mode = modes[mode_i]
            sys_msg = system
            if mode == "json_object":
                sys_msg = system + "\n\nReply with a single JSON object that follows this JSON schema exactly:\n" + json.dumps(schema)
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            if mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": mode == "strict", "schema": schema},
                }
            if effort and caps.reasoning_effort:
                kwargs["reasoning_effort"] = effort
            if caps.include_reasoning and _is_reasoning_model(model):
                kwargs["extra_body"] = {"include_reasoning": False}

            need = est_tokens(sys_msg + user) + max_tokens
            self._pace(model, need)
            t0 = time.time()
            try:
                raw, attempts = self._call(
                    lambda: self._client.chat.completions.with_raw_response.create(**kwargs), stage, model
                )
            except openai.BadRequestError as e:
                msg = str(e)
                low = msg.lower()
                if "reasoning_effort" in low and caps.reasoning_effort:
                    caps.reasoning_effort = False
                    continue
                if "include_reasoning" in low and caps.include_reasoning:
                    caps.include_reasoning = False
                    continue
                if any(k in low for k in ("context length", "context_length", "maximum context", "too many tokens")):
                    raise RequestTooLargeError(f"The input is too long for {model}.", detail=msg)
                if ("does not match" in low or "json_validate_failed" in low or "failed to generate" in low) and bad_json_retries < 1:
                    bad_json_retries += 1
                    self.notify(f"{model} returned malformed JSON, retrying…")
                    continue
                if mode_i < len(modes) - 1 and any(k in low for k in ("json_schema", "response_format", "strict", "schema", "json")):
                    mode_i += 1
                    caps.json_mode = modes[mode_i]
                    bad_json_retries = 0
                    continue
                raise APIError(f"{model} rejected the request during {stage}.", hint="Try another model in the sidebar.", detail=msg)

            self._record_headers(model, raw.headers)
            completion = raw.parse()
            choice = completion.choices[0]
            content = (choice.message.content or "").strip()
            usage = getattr(completion, "usage", None)
            rec = CallRecord(stage, model, "chat", time.time() - t0, attempts=attempts, mode=mode)
            if usage:
                rec.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                rec.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                det = getattr(usage, "completion_tokens_details", None)
                rec.reasoning_tokens = (getattr(det, "reasoning_tokens", 0) or 0) if det else 0
            self.calls.append(rec)

            if not content or choice.finish_reason == "length":
                # the model spent its whole output allowance (usually on reasoning)
                if truncation_retries < 1:
                    truncation_retries += 1
                    if effort and effort != "low":
                        effort = "low"
                        self.notify(f"{model} ran out of output space; retrying with less reasoning…")
                    else:
                        raise ModelOutputError(
                            f"{model} ran out of output space during {stage}.",
                            hint="The meeting may be too long for one request. Try a model with a larger limit.",
                        )
                    continue
                raise ModelOutputError(
                    f"{model} didn't return a complete answer during {stage}.",
                    hint="Retry this stage, or pick a different model in the sidebar.",
                )
            data = _loads_lenient(content)
            if data is None or not isinstance(data, dict):
                if bad_json_retries < 1:
                    bad_json_retries += 1
                    continue
                if mode_i < len(modes) - 1:
                    mode_i += 1
                    continue
                raise ModelOutputError(f"{model} didn't return valid JSON during {stage}.", detail=content[:1500])
            return data


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return "gpt-oss" in m or "qwen3" in m or "deepseek-r1" in m or "minimax" in m


def _loads_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
