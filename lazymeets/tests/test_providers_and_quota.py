"""Multiple providers (separate speech-to-text and language-model keys) and the free-key daily allowance."""
import json
import urllib.request
from dataclasses import replace

from meetscribe.config import PROVIDERS, Settings, provider_for_url
from meetscribe.llm import ModelClient
from meetscribe.pipeline import new_run, run_pipeline
from meetscribe.schemas import MeetingContext
from meetscribe.usage import UsageStore, fmt_minutes


def _calls(mock_api):
    with urllib.request.urlopen(mock_api + "/__calls") as r:
        return json.load(r)["calls"]


def _post(mock_api, path, body):
    req = urllib.request.Request(mock_api + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req).read()


def test_provider_registry():
    assert provider_for_url("https://api.groq.com/openai/v1").id == "groq"
    assert provider_for_url("https://api.openai.com/v1/").id == "openai"
    assert provider_for_url("http://localhost:11434/v1").id == "custom"
    assert PROVIDERS["groq"].stt and PROVIDERS["openai"].stt
    assert not PROVIDERS["gemini"].stt  # Gemini has no Whisper-style endpoint


def test_keys_are_never_saved_with_a_run():
    s = Settings(api_key="secret-llm", stt_api_key="secret-stt")
    d = json.dumps(s.public_dict())
    assert "secret-llm" not in d and "secret-stt" not in d


def test_speech_and_language_models_use_their_own_keys(settings, demo_audio):
    """Whisper on one provider/key, the language models on another: each call carries the right key."""
    s = replace(settings, api_key="llm-key", stt_api_key="stt-key", stt_base_url=settings.base_url)
    before = len(_calls(settings.base_url))
    run = new_run(demo_audio.name, s, MeetingContext())
    run.source_path.write_bytes(demo_audio.read_bytes())
    run_pipeline(run, s, lambda *a: None)
    assert run.record is not None
    new = _calls(settings.base_url)[before:]
    stt = [c for c in new if c["kind"] == "stt"]
    chat = [c for c in new if c["kind"] == "chat"]
    assert stt and all(c["auth"] == "Bearer stt-key" for c in stt)
    assert chat and all(c["auth"] == "Bearer llm-key" for c in chat)


def test_falls_back_to_max_tokens_for_older_providers(settings):
    """Some OpenAI-compatible APIs reject `max_completion_tokens`; the client retries with `max_tokens`."""
    _post(settings.base_url, "/__reject_mct", {"on": True})
    try:
        client = ModelClient(settings)
        data = client.chat_json(stage="qa", model="openai/gpt-oss-120b", system="s", user="Question: what was decided?\n[1 | 00:00 | ] x",
                                schema_name="meeting_answer", schema={"type": "object"}, max_tokens=500)
        assert isinstance(data, dict)
        assert client._caps["openai/gpt-oss-120b"].completion_tokens_param == "max_tokens"
    finally:
        _post(settings.base_url, "/__reject_mct", {"on": False})


def test_daily_allowance(tmp_path):
    store = UsageStore(tmp_path / "usage.json", daily_limit_s=60 * 60)
    assert store.remaining("a@x.com") == 3600
    assert store.allows("a@x.com", 50 * 60)
    store.charge("A@x.com", 50 * 60)  # emails are case-insensitive
    assert store.remaining("a@x.com") == 600
    assert not store.allows("a@x.com", 11 * 60)
    assert store.allows("b@x.com", 59 * 60)  # other people are separate
    raw = (tmp_path / "usage.json").read_text()
    assert "a@x.com" not in raw  # stored as a hash
    assert fmt_minutes(600) == "10 min" and fmt_minutes(125) == "2 min 5 s"


def test_allowance_resets_each_day(tmp_path):
    store = UsageStore(tmp_path / "usage.json", daily_limit_s=3600)
    (tmp_path / "usage.json").write_text(json.dumps({"2000-01-01": {"whatever": 3600}}))
    assert store.remaining("a@x.com") == 3600
    store.charge("a@x.com", 60)
    assert "2000-01-01" not in json.loads((tmp_path / "usage.json").read_text())  # old days are dropped


def test_no_limit_when_disabled(tmp_path):
    store = UsageStore(tmp_path / "usage.json", daily_limit_s=0)
    assert not store.enabled and store.allows("a@x.com", 10 ** 6)


def test_chat_model_filter():
    import onboarding

    ids = ["gpt-4.1", "whisper-1", "text-embedding-3-small", "gpt-4o-mini-tts", "openai/gpt-oss-120b", "dall-e-3"]
    assert onboarding.chat_models(ids) == ["gpt-4.1", "openai/gpt-oss-120b"]
    assert onboarding.stt_models("openai", ["whisper-1", "gpt-4o-transcribe"]) == ["whisper-1"]
    assert onboarding.stt_models("groq", ["whisper-large-v3", "llama"]) == ["whisper-large-v3"]


def test_only_fresh_sign_ins_count():
    import onboarding

    now = 1_800_000_000
    assert onboarding.is_fresh_login({"iat": now - 30}, now)  # just signed in
    assert not onboarding.is_fresh_login({"iat": now - 3 * 86400}, now)  # remembered cookie from days ago
    assert onboarding.is_fresh_login({"exp": now + 3500}, now)  # no iat: token issued ~100 s ago
    assert not onboarding.is_fresh_login({"exp": now - 7200}, now)
    assert onboarding.is_fresh_login({}, now)  # can't tell: don't loop the user through sign-in
