"""The pages people see before the main app:

1. Sign in with Google (only when sign-in is set up in Streamlit secrets)
2. Choose: the free preloaded Groq key, or your own API keys
3. Own keys: pick a provider, paste a key, pick models, choose speech-to-text

Keys typed here live only in this browser session (st.session_state). They are never written to
disk, to run folders, or to logs.
"""
from __future__ import annotations

import hashlib
import html
import time
from dataclasses import replace
from typing import Optional

import streamlit as st

from meetscribe.config import GROQ_BASE_URL, PROVIDERS, Settings
from meetscribe.errors import MeetScribeError
from meetscribe.llm import ModelClient
from meetscribe.usage import UsageStore, fmt_minutes

ss = st.session_state

# model ids that can't be used for the chat stages
_NOT_CHAT = ("whisper", "tts", "embed", "dall-e", "image", "moderation", "transcribe", "realtime", "audio", "guard",
             "rerank", "search", "computer-use", "imagen", "veo", "aqa", "ocr", "playai", "orpheus", "distil")


def chat_models(ids: list[str]) -> list[str]:
    return [m for m in ids if not any(k in m.lower() for k in _NOT_CHAT)]


def stt_models(provider_id: str, ids: list[str]) -> list[str]:
    if provider_id == "openai":
        # only whisper-1 returns the segment and word timestamps the pipeline needs
        return [m for m in ids if m == "whisper-1"] or ["whisper-1"]
    found = [m for m in ids if "whisper" in m.lower() or "transcribe" in m.lower()]
    return found or list(PROVIDERS.get(provider_id, PROVIDERS["custom"]).stt_models)


# --------------------------------------------------------------------------- sign-in


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


# A sign-in only counts if it happened just now. Streamlit keeps a sign-in cookie for 30 days; without
# this check, anyone opening the site on that browser would land in the account (and its API keys).
FRESH_LOGIN_S = 10 * 60
_SESSION_KEYS = ("mode", "own", "own_ready", "run_dir", "qa", "login_ok", "user", "job", "pending", "busy")


def is_fresh_login(claims: dict, now: float, max_age_s: float = FRESH_LOGIN_S) -> bool:
    """True when the identity token was issued within the last `max_age_s` seconds.
    Uses `iat` (issued at); Google tokens last an hour, so `exp` - 3600 works as a fallback.
    If the token has neither, we can't tell, so we don't force a new sign-in (that would loop)."""
    try:
        if claims.get("iat") is not None:
            issued = float(claims["iat"])
        elif claims.get("exp") is not None:
            issued = float(claims["exp"]) - 3600
        else:
            return True
    except (TypeError, ValueError):
        return True
    return now - issued <= max_age_s


def sign_out() -> None:
    """Forget everything about this person in this tab (including pasted keys), then sign out."""
    for k in _SESSION_KEYS:
        ss.pop(k, None)
    st.logout()


def current_user() -> Optional[dict]:
    """{'email', 'name'} of the signed-in person, a local stand-in when sign-in isn't set up, or None."""
    if not auth_configured():
        return {"email": "local", "name": "Local user", "local": True}
    try:
        logged_in = bool(st.user.is_logged_in)
    except Exception:
        return None
    if not logged_in:
        return None
    if not ss.get("login_ok"):
        # a new browser session: only accept a sign-in that just happened, not a remembered cookie
        claims = {k: st.user.get(k) for k in ("iat", "exp")}
        if not is_fresh_login(claims, time.time()):
            sign_out()
            return None
        ss.login_ok = True
    return {"email": str(st.user.get("email") or ""), "name": str(st.user.get("name") or st.user.get("email") or "you"),
            "local": False}


def account_bar(user: dict, key: str) -> None:
    """Who is signed in, with a Sign out button (top of the setup pages)."""
    left, right = st.columns([5, 1.3])
    with left:
        if user.get("local"):
            st.markdown("<span class='ob-acct warn'>⚠️ Sign-in isn't set up on this server, so anyone who opens it gets in. "
                        "See docs/DEPLOY.md to turn on Google sign-in.</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='ob-acct'>Signed in as <b>{html.escape(user['email'])}</b></span>", unsafe_allow_html=True)
    with right:
        if not user.get("local"):
            if st.button("Sign out", width="stretch", key=f"ob_signout_{key}"):
                sign_out()


def user_folder(user: dict) -> str:
    return "u_" + hashlib.sha256(user["email"].lower().encode()).hexdigest()[:16]


def _provider_name() -> Optional[str]:
    try:
        auth = st.secrets["auth"]
        named = [k for k in auth.keys() if hasattr(auth[k], "keys")]
        return "google" if "google" in named else (named[0] if named else None)
    except Exception:
        return None


def _css() -> None:
    st.markdown(
        """<style>
.ob-wrap{max-width:860px;margin:4vh auto 0}
.ob-hero{text-align:center;margin-bottom:1.6rem}
.ob-hero img{width:110px;height:110px;border-radius:50%;object-fit:cover;box-shadow:0 0 0 4px color-mix(in srgb,var(--ms-primary) 40%,transparent)}
.ob-hero h1{font-size:3rem;font-weight:800;margin:.6rem 0 0;color:var(--ms-text)!important;letter-spacing:-.02em;padding:0}
.ob-hero h1 span{color:var(--ms-primary)!important}
.ob-hero p{color:var(--ms-muted);font-size:1.1rem;margin:.4rem auto 0;max-width:38rem;line-height:1.5}
.ob-step{color:var(--ms-primary);font-weight:700;font-size:.85rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.2rem}
.ob-h{font-size:1.5rem;font-weight:800;color:var(--ms-text);margin:0 0 .3rem}
.ob-sub{color:var(--ms-muted);margin:0 0 1rem}
.ob-card-t{font-size:1.3rem;font-weight:800;color:var(--ms-text);margin:.2rem 0 .3rem}
.ob-card-d{color:var(--ms-muted);line-height:1.5;margin-bottom:.6rem;min-height:4.6em}
.ob-pill{display:inline-block;font-size:.8rem;border-radius:999px;padding:.15rem .7rem;margin:0 .3rem .4rem 0;
  border:1px solid var(--ms-border);color:var(--ms-text);background:var(--ms-bg)}
.ob-pill.ok{border-color:var(--ms-primary);color:var(--ms-primary)}
.ob-note{color:var(--ms-muted);font-size:.85rem;text-align:center;margin-top:1rem}
.ob-prov{color:var(--ms-muted);font-size:.92rem;margin:-.2rem 0 .6rem}
.ob-acct{color:var(--ms-muted)!important;font-size:.9rem;line-height:2.4}
.ob-acct.warn{color:var(--ms-accent)!important}
.ob-acct b{color:var(--ms-text)}
[class*="st-key-ob_card"]{background:var(--ms-surface);border:1px solid var(--ms-border)!important;border-radius:16px;padding:1.2rem 1.3rem 1rem;
  box-shadow:0 6px 24px rgba(0,0,0,.10);height:100%}
[class*="st-key-ob_card"]:hover{border-color:var(--ms-primary)!important}
[class*="st-key-ob_box"]{background:var(--ms-surface);border:1px solid var(--ms-border)!important;border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:.9rem}
</style>""",
        unsafe_allow_html=True,
    )


def _hero(logo_b64: str, tagline: str) -> None:
    img = f"<img src='data:image/png;base64,{logo_b64}' alt=''>" if logo_b64 else ""
    st.markdown(f"<div class='ob-hero'>{img}<h1>Lazy<span>Meets</span></h1><p>{tagline}</p></div>", unsafe_allow_html=True)


def login_page(logo_b64: str) -> None:
    _css()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        _hero(logo_b64, "Turn a meeting recording into a transcript, minutes, decisions and action items. "
                        "Owners and deadlines are only filled in when someone actually said them.")
        with st.container(key="ob_box_login"):
            st.markdown("<div class='ob-step'>Step 1 of 2</div><div class='ob-h'>Sign in to continue</div>"
                        "<p class='ob-sub'>We only get your name and email from Google. They're used to keep your meetings "
                        "separate from everyone else's and to share the free daily allowance fairly. For safety you sign in "
                        "each time you open the site.</p>", unsafe_allow_html=True)
            if st.button("Sign in with Google", type="primary", width="stretch", key="ob_login"):
                name = _provider_name()
                st.login(name) if name else st.login()


# --------------------------------------------------------------------------- choose free key or own keys


def choose_page(logo_b64: str, user: dict, store: UsageStore, server_key: bool) -> None:
    _css()
    account_bar(user, "choose")
    first = "there" if user.get("local") else html.escape(user["name"].split(" ")[0])
    _hero(logo_b64, f"Hi {first}! How do you want to run the models?")
    st.markdown("<div class='ob-wrap'></div>", unsafe_allow_html=True)
    _, c1, c2, _ = st.columns([0.4, 2, 2, 0.4], gap="medium")
    left = store.remaining(user["email"])
    with c1:
        with st.container(key="ob_card_free"):
            st.markdown("<div class='ob-card-t'>⚡ Use the free key</div>"
                        "<div class='ob-card-d'>Nothing to set up. Runs on our Groq key with Whisper large-v3, "
                        "GPT-OSS 20B for refinement and GPT-OSS 120B for the minutes.</div>", unsafe_allow_html=True)
            if store.enabled:
                st.markdown(f"<span class='ob-pill ok'>{fmt_minutes(left)} of audio left today</span>"
                            f"<span class='ob-pill'>{fmt_minutes(store.limit)} per day</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='ob-pill ok'>No daily limit</span>", unsafe_allow_html=True)
            if not server_key:
                st.caption("Not available: this server has no Groq key set up.")
            if st.button("Continue with the free key", type="primary", width="stretch", key="ob_pick_free",
                         disabled=not server_key):
                ss.mode = "free"
                st.rerun()
    with c2:
        with st.container(key="ob_card_own"):
            st.markdown("<div class='ob-card-t'>🔑 Use your own API keys</div>"
                        "<div class='ob-card-d'>Bring a key from OpenAI, Google Gemini, Groq, OpenRouter, Mistral, Cerebras "
                        "or any OpenAI-compatible API, and pick the models yourself.</div>", unsafe_allow_html=True)
            st.markdown("<span class='ob-pill ok'>No daily limit from us</span><span class='ob-pill'>Keys stay in this tab</span>",
                        unsafe_allow_html=True)
            if st.button("Set up my keys", width="stretch", key="ob_pick_own"):
                ss.mode = "own"
                ss.own_ready = False
                st.rerun()
    st.markdown("<p class='ob-note'>You can switch later from the sidebar. Keys you paste are kept only in this browser tab, "
                "never saved on the server.</p>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- own keys wizard


def _own() -> dict:
    if "own" not in ss or not isinstance(ss.own, dict):
        ss.own = {"provider": "openai", "key": "", "url": "", "models": [], "refine": "", "minutes": "",
                  "stt": "", "stt_provider": "groq", "stt_key": "", "stt_url": "", "stt_models": [], "stt_model": "",
                  "checked": "", "stt_checked": ""}
    return ss.own


def _probe(base: Settings, url: str, key: str, stt: bool) -> list[str]:
    """List the models a key can use. Raises MeetScribeError with a readable message."""
    s = replace(base, api_key=key, base_url=url, stt_api_key="", stt_base_url="", stt_backend="api")
    return ModelClient(s).available_models(stt=stt)


def _pick(options: list[str], *prefer: str) -> int:
    for p in prefer:
        if p in options:
            return options.index(p)
    return 0


def keys_page(base: Settings, server_key: bool, free_minutes: str, user: Optional[dict] = None) -> None:
    _css()
    account_bar(user or {"local": True}, "keys")
    o = _own()
    top = st.columns([1, 5])
    with top[0]:
        if st.button("← Back", type="tertiary", key="ob_back"):
            ss.mode = None
            st.rerun()
    st.markdown("<div class='ob-wrap' style='margin-top:0'><div class='ob-step'>Your own keys</div>"
                "<div class='ob-h'>Set up your models</div><p class='ob-sub'>Three short steps. Your keys stay in this browser tab "
                "only and are never saved on the server.</p></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([0.5, 4, 0.5])
    with mid:
        # ---- step 1: provider + key
        with st.container(key="ob_box_1"):
            st.markdown("<div class='ob-step'>Step 1</div><div class='ob-h'>Language models: provider and key</div>"
                        "<p class='ob-sub'>These run transcript refinement, the minutes, and the Q&amp;A.</p>", unsafe_allow_html=True)
            ids = list(PROVIDERS)
            pid = st.selectbox("Provider", ids, index=ids.index(o["provider"]) if o["provider"] in ids else 0,
                               format_func=lambda i: PROVIDERS[i].name, key="ob_provider")
            if pid != o["provider"]:
                o.update(provider=pid, models=[], checked="", refine="", minutes="", stt="")
            p = PROVIDERS[pid]
            link = f" Get a key: <a href='{p.key_url}' target='_blank'>{p.key_url.split('//')[-1]}</a>" if p.key_url else ""
            st.markdown(f"<p class='ob-prov'>{html.escape(p.note)}{link}</p>", unsafe_allow_html=True)
            url = p.base_url
            if pid == "custom":
                url = st.text_input("Base URL", value=o["url"], placeholder="https://api.example.com/v1", key="ob_url").strip()
                o["url"] = url
            key = st.text_input("API key" if pid == "custom" else f"{p.name} API key", value=o["key"], type="password", placeholder=p.key_hint, key="ob_key").strip()
            if key != o["key"]:
                o.update(key=key, models=[], checked="")
            if st.button("Check key and load models", type="primary", key="ob_check", disabled=not key or not url):
                try:
                    with st.spinner("Checking the key…"):
                        found = chat_models(_probe(base, url, key, stt=False))
                    if not found:
                        st.error("The key works, but no chat models came back for it.")
                    else:
                        o.update(models=found, checked=key)
                except MeetScribeError as e:
                    st.error(e.message)
                    if e.hint:
                        st.caption(e.hint)
            if o["models"] and o["checked"] == key:
                st.success(f"Key works. {len(o['models'])} models available.")

        ready1 = bool(o["models"]) and o["checked"] == o["key"]
        # ---- step 2: models
        with st.container(key="ob_box_2"):
            st.markdown("<div class='ob-step'>Step 2</div><div class='ob-h'>Pick the two language models</div>"
                        "<p class='ob-sub'>Refinement only fixes misheard words, so a smaller, faster model is fine. "
                        "The minutes model does the harder work, so pick the stronger one there.</p>", unsafe_allow_html=True)
            if not ready1:
                st.caption("Check your key in step 1 first.")
            else:
                ms = o["models"]
                o["refine"] = st.selectbox("Language model #1: refinement", ms, index=_pick(ms, o["refine"], p.refine_default),
                                           key="ob_refine")
                o["minutes"] = st.selectbox("Language model #2: minutes, decisions and tasks", ms,
                                            index=_pick(ms, o["minutes"], p.minutes_default), key="ob_minutes")
                if o["refine"] == o["minutes"]:
                    st.warning("Both stages use the same model. It works, but the two stages then share one rate limit, "
                               "and the brief asks for two separate language models.")

        # ---- step 3: speech-to-text
        with st.container(key="ob_box_3"):
            st.markdown("<div class='ob-step'>Step 3</div><div class='ob-h'>Speech-to-text</div>", unsafe_allow_html=True)
            opts = []
            if p.stt:
                opts.append(("same", f"Use my {p.name} key ({'Whisper' if pid != 'custom' else 'its speech-to-text'})"))
            if server_key:
                opts.append(("free", f"Use the free Groq Whisper (counts toward your {free_minutes} per day)"))
            opts.append(("other", "Use another key for speech-to-text (Groq or OpenAI)"))
            opts.append(("offline", "Offline on the server (no key needed, slower and less accurate)"))
            codes = [c for c, _ in opts]
            if o["stt"] not in codes:
                o["stt"] = codes[0]
            o["stt"] = st.radio("How should the audio be transcribed?", codes, index=codes.index(o["stt"]),
                                format_func=dict(opts).get, key="ob_stt")
            if not p.stt:
                st.caption(f"{p.name} doesn't offer Whisper-style speech-to-text, so it comes from one of these instead.")
            stt_ready = True
            if o["stt"] == "same":
                if ready1:
                    try:
                        if not o["stt_models"] or o["stt_checked"] != o["key"] + url:
                            o["stt_models"] = stt_models(pid, _probe(base, url, o["key"], stt=True))
                            o["stt_checked"] = o["key"] + url
                    except MeetScribeError:
                        o["stt_models"] = stt_models(pid, [])
                    sm = o["stt_models"]
                    o["stt_model"] = st.selectbox("Speech-to-text model", sm, index=_pick(sm, o["stt_model"], "whisper-large-v3", "whisper-1"),
                                                  key="ob_stt_model_same")
                else:
                    stt_ready = False
            elif o["stt"] == "other":
                sp = st.selectbox("Speech-to-text provider", ["groq", "openai"], format_func=lambda i: PROVIDERS[i].name,
                                  index=0 if o["stt_provider"] == "groq" else 1, key="ob_stt_provider")
                if sp != o["stt_provider"]:
                    o.update(stt_provider=sp, stt_models=[], stt_checked="")
                skey = st.text_input(f"{PROVIDERS[sp].name} API key for speech-to-text", value=o["stt_key"], type="password",
                                     placeholder=PROVIDERS[sp].key_hint, key="ob_stt_key").strip()
                if skey != o["stt_key"]:
                    o.update(stt_key=skey, stt_models=[], stt_checked="")
                if st.button("Check speech-to-text key", key="ob_stt_check", disabled=not skey):
                    try:
                        with st.spinner("Checking…"):
                            o["stt_models"] = stt_models(sp, _probe(base, PROVIDERS[sp].base_url, skey, stt=True))
                            o["stt_checked"] = skey
                    except MeetScribeError as e:
                        st.error(e.message)
                stt_ready = bool(o["stt_models"]) and o["stt_checked"] == skey
                if stt_ready:
                    sm = o["stt_models"]
                    o["stt_model"] = st.selectbox("Speech-to-text model", sm, index=_pick(sm, o["stt_model"], "whisper-large-v3", "whisper-1"),
                                                  key="ob_stt_model_other")
            elif o["stt"] == "free":
                o["stt_model"] = "whisper-large-v3"
                st.caption("Whisper large-v3 on our Groq key. Only the audio minutes count toward the daily allowance; "
                           "your own key does the rest.")
            else:
                st.caption("Runs Moonshine on the server's CPU. The first run downloads the model (about 60 MB).")

        done = ready1 and stt_ready
        if st.button("Start using LazyMeets →", type="primary", width="stretch", key="ob_done", disabled=not done):
            ss.own_ready = True
            st.rerun()
        if not done:
            st.caption("Finish the steps above to continue.")


# --------------------------------------------------------------------------- turning the choice into settings


def apply_mode(s: Settings, server_key: str) -> Settings:
    """Fill in providers, keys and models for the chosen mode."""
    server_url = s.base_url  # from .env / secrets: where the free key works
    if ss.get("mode") == "own" and ss.get("own_ready"):
        o = _own()
        p = PROVIDERS[o["provider"]]
        s.base_url = o["url"] if o["provider"] == "custom" else p.base_url
        s.api_key = o["key"]
        s.default_tpm = p.tpm
        s.refine_model = o["refine"] or s.refine_model
        s.minutes_model = o["minutes"] or s.minutes_model
        s.stt_backend = "local" if o["stt"] == "offline" else "api"
        if o["stt"] == "same":
            s.stt_base_url, s.stt_api_key = "", ""
        elif o["stt"] == "free":
            s.stt_base_url, s.stt_api_key = server_url or GROQ_BASE_URL, server_key
        elif o["stt"] == "other":
            s.stt_base_url, s.stt_api_key = PROVIDERS[o["stt_provider"]].base_url, o["stt_key"]
        if o["stt_model"]:
            s.stt_model = o["stt_model"]
    else:
        # the free key: whatever the server owner configured (GROQ_API_KEY, Groq by default)
        s.api_key = server_key
    return s


def uses_free_key(s: Settings) -> bool:
    """True when this run spends the app's own key (so it counts toward the daily allowance)."""
    if ss.get("mode") != "own":
        return True
    o = _own()
    return o.get("stt") == "free" and s.stt_backend == "api"


def mode_label() -> str:
    if ss.get("mode") == "own":
        o = _own()
        p = "Custom API" if o["provider"] == "custom" else PROVIDERS[o["provider"]].name
        stt = {"same": p, "free": "free Groq", "other": PROVIDERS[o.get("stt_provider", "groq")].name, "offline": "offline"}.get(o["stt"], "")
        return f"Your keys · {p} · speech: {stt}"
    return "Free key · Groq"
