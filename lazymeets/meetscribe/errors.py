"""Errors that carry a message meant for the person using the app.

Every error raised inside the pipeline is turned into one of these, so the interface can
always say what went wrong in plain words instead of showing a stack trace.
"""
from __future__ import annotations


class MeetScribeError(Exception):
    """Base error. `message` is shown to the user, `hint` says what they can do about it."""

    def __init__(self, message: str, hint: str = "", detail: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail  # technical detail, shown in an expander

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class AudioError(MeetScribeError):
    """The uploaded file can't be used (wrong type, empty, unreadable, silent...)."""


class NoSpeechError(AudioError):
    """The file decodes fine but nobody is talking in it."""


class ConfigError(MeetScribeError):
    """Missing API key, unknown model, bad settings."""


class APIError(MeetScribeError):
    """The model provider returned an error we couldn't recover from."""


class RateLimitError(APIError):
    """Rate limit hit and the wait would be too long to retry automatically."""


class RequestTooLargeError(APIError):
    """A single request is bigger than the provider allows (usually the tokens-per-minute cap)."""


class ModelOutputError(APIError):
    """The model answered, but not with usable output (cut off, invalid JSON...)."""


class StageError(MeetScribeError):
    """Wraps any error with the pipeline stage it happened in."""

    def __init__(self, stage: str, error: MeetScribeError):
        super().__init__(error.message, error.hint, error.detail)
        self.stage = stage
        self.cause = error
