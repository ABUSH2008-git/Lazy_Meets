"""Daily allowance for the free, preloaded API key.

Each signed-in user may process a limited number of audio minutes per day with the app's own key,
so one person can't use up the shared quota for everyone. People who bring their own keys aren't
counted.

Usage is kept in a small JSON file. Emails are stored as a hash, not in plain text. Only today's
numbers are kept; older days are dropped on the next write.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_LOCK = threading.Lock()


def _user_id(user: str) -> str:
    return hashlib.sha256(user.strip().lower().encode()).hexdigest()[:24]


class UsageStore:
    def __init__(self, path: Path, daily_limit_s: float, tz: str = "Asia/Kolkata"):
        self.path = Path(path)
        self.limit = float(daily_limit_s)
        try:
            self.tz = ZoneInfo(tz)
        except Exception:
            self.tz = ZoneInfo("UTC")

    # ------------------------------------------------------------------ helpers
    def today(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ public
    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def used(self, user: str) -> float:
        day = self._load().get(self.today(), {})
        return float(day.get(_user_id(user), 0.0))

    def remaining(self, user: str) -> float:
        if not self.enabled:
            return float("inf")
        return max(0.0, self.limit - self.used(user))

    def allows(self, user: str, seconds: float) -> bool:
        return not self.enabled or seconds <= self.remaining(user) + 1.0  # 1 s of slack for rounding

    def charge(self, user: str, seconds: float) -> float:
        """Add `seconds` to today's usage and return the new total."""
        if seconds <= 0:
            return self.used(user)
        with _LOCK:
            data = self._load()
            today = self.today()
            data = {today: data.get(today, {})}  # drop older days
            uid = _user_id(user)
            data[today][uid] = float(data[today].get(uid, 0.0)) + float(seconds)
            self._save(data)
            return data[today][uid]


def fmt_minutes(seconds: float) -> str:
    if seconds == float("inf"):
        return "unlimited"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m} min" if s == 0 else f"{m} min {s} s"
