"""LazyMeets: Streamlit interface.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import html
import shutil
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from meetscribe import __version__
from meetscribe.audio import probe_duration, validate_upload
from meetscribe.config import (
    LLM_MODEL_CHOICES,
    PROVIDERS,
    REASONING_CHOICES,
    SAMPLES_DIR,
    STT_MODEL_CHOICES,
    Settings,
)
from meetscribe.usage import UsageStore, fmt_minutes
import onboarding as ob
from meetscribe.errors import MeetScribeError, StageError
from meetscribe.exports import email_draft
from meetscribe.glossaries import PRESETS, parse_terms
from meetscribe.llm import ModelClient
from meetscribe.pipeline import STAGE_LABELS, STAGES, Run, list_runs, new_run, run_pipeline
from meetscribe.qa import ask
from meetscribe.render import player_html, side_by_side_html
from meetscribe.schemas import MeetingContext, fmt_ts

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "lazymeets_logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

st.set_page_config(page_title="LazyMeets: meeting minutes from audio",
                   page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📝", layout="wide")

THEMES = {
    "Midnight Teal": {"bg": "#0F1A2B", "surface": "#16243A", "border": "#2A3F5F", "primary": "#14A3B8", "accent": "#F2994A",
                      "text": "#E6EDF5", "muted": "#8FA3BF", "ins": "rgba(20,163,184,.25)", "del": "rgba(242,153,74,.25)",
                      "warn": "#F2B36B", "dark": True},
    "Soft Daylight": {"bg": "#F3F6FA", "surface": "#FFFFFF", "border": "#D5DEE8", "primary": "#0E7C94", "accent": "#E8833A",
                      "text": "#1E2A3A", "muted": "#5B6B7F", "ins": "rgba(14,124,148,.16)", "del": "rgba(232,131,58,.18)",
                      "warn": "#9A5A12", "dark": False},
    "Warm Sand": {"bg": "#FAF5EC", "surface": "#FFFFFF", "border": "#E6DCC8", "primary": "#2F7F86", "accent": "#D9822B",
                  "text": "#2B2A28", "muted": "#76705F", "ins": "rgba(47,127,134,.16)", "del": "rgba(217,130,43,.18)",
                  "warn": "#94550F", "dark": False},
}


def theme() -> dict:
    return THEMES.get(st.session_state.get("theme_name", "Midnight Teal"), THEMES["Midnight Teal"])


def inject_css(t: dict) -> None:
    vars_ = (f"--ms-bg:{t['bg']};--ms-surface:{t['surface']};--ms-border:{t['border']};--ms-primary:{t['primary']};"
             f"--ms-accent:{t['accent']};--ms-text:{t['text']};--ms-muted:{t['muted']};--ms-ins:{t['ins']};--ms-del:{t['del']};"
             f"--ms-warn:{t['warn']};")
    on_primary = "#FFFFFF" if not t["dark"] else "#06121F"
    scheme = "dark" if t["dark"] else "light"
    css_scheme = f"audio, [data-testid=stAudio], [data-testid=stApp]{{color-scheme:{scheme}!important}}"
    st.markdown(
        "<style>" + css_scheme + ":root{" + vars_ + f"--ms-on-primary:{on_primary};--ms-on-accent:#1A1410;color-scheme:{scheme};" + "}" + r"""
/* ---------- base surfaces ---------- */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--ms-bg)!important;color:var(--ms-text)}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a, [data-testid="stMainMenu"] svg{color:var(--ms-muted)!important;fill:var(--ms-muted)}
[data-testid="stMainMenuPopover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"]{background:var(--ms-surface)!important;color:var(--ms-text)!important;border-color:var(--ms-border)!important}
[role="option"]{color:var(--ms-text)!important}
[role="option"]:hover, [role="option"][aria-selected="true"]{background:color-mix(in srgb,var(--ms-primary) 16%,transparent)!important}
[data-testid="stSidebar"]{background:var(--ms-surface)!important;border-right:1px solid var(--ms-border)}
[data-testid="stSidebar"] *{color:var(--ms-text)}
[data-testid="stApp"] :is(p,li,label,h1,h2,h3,h4,h5,h6,span,strong,em,td,th,summary,small){color:inherit}
[data-testid="stApp"] h1,[data-testid="stApp"] h2,[data-testid="stApp"] h3{color:var(--ms-text)}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stCaption{color:var(--ms-muted)!important}
[data-testid="stWidgetLabel"] *{color:var(--ms-text)!important}
a{color:var(--ms-primary)}
hr{border-color:var(--ms-border)!important}

/* ---------- layout & type ---------- */
.block-container{padding-top:2.4rem;padding-bottom:2.5rem;max-width:1200px}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]{gap:1.15rem}
.ms-hero{display:flex;align-items:center;gap:1.4rem;margin:.2rem 0 1.4rem}
.ms-logo{width:112px;height:112px;border-radius:50%;object-fit:cover;flex:none;border:3px solid color-mix(in srgb,var(--ms-primary) 70%,transparent);
  box-shadow:0 0 0 6px color-mix(in srgb,var(--ms-primary) 14%,transparent),0 8px 26px color-mix(in srgb,var(--ms-primary) 28%,transparent)}
.ms-brand{font-size:3.3rem;font-weight:800;line-height:1.05;letter-spacing:-.02em;color:var(--ms-text);margin:0}
.ms-brand span{color:var(--ms-primary)}
.ms-tag{font-size:1.15rem;line-height:1.5;color:var(--ms-muted);margin:.45rem 0 0;max-width:46rem}
.ms-side-logo{display:flex;align-items:center;gap:.7rem;margin:-.6rem 0 .6rem}
.ms-side-logo img{width:44px;height:44px;border-radius:50%;object-fit:cover;border:2px solid var(--ms-primary)}
.ms-side-logo b{font-size:1.25rem}
@media (max-width:640px){.ms-hero{flex-direction:column;align-items:flex-start;gap:.8rem}.ms-logo{width:88px;height:88px}.ms-brand{font-size:2.5rem}}
h1.ms-title{font-size:2.4rem;font-weight:800;margin:0;color:var(--ms-text)}
.ms-sub{color:var(--ms-muted);margin:.1rem 0 1.2rem}

/* ---------- cards, expanders, containers ---------- */
[data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {border-color:var(--ms-border)!important;border-radius:12px!important}
[data-testid="stExpander"] details{background:var(--ms-surface)!important}
[data-testid="stExpander"] summary{font-size:1.02rem;padding:.85rem 1rem}
[data-testid="stExpander"] summary:hover{color:var(--ms-primary)!important}
.ms-stage{background:var(--ms-surface);border:1px solid var(--ms-border);border-radius:12px;padding:.8rem 1rem;height:100%;box-shadow:0 2px 10px rgba(0,0,0,.08)}
.ms-stage .t{font-weight:700;font-size:1.02rem}
.ms-stage .m{font-size:.88rem;color:var(--ms-muted)}
.ms-quote{color:var(--ms-muted);font-style:italic;font-size:.92rem}
.ms-flag{color:var(--ms-warn);font-size:.86rem}
.ms-unspec{color:var(--ms-muted);font-style:italic}
div[data-testid="stMetricValue"]{font-size:1.7rem;color:var(--ms-text)}
div[data-testid="stMetricLabel"] *{color:var(--ms-muted)!important}

/* ---------- tabs ---------- */
[role="tablist"]{gap:.4rem;border-bottom:1px solid var(--ms-border)!important}
[data-testid="stTab"]{padding:.7rem .9rem!important}
[data-testid="stTab"] p{font-size:1.06rem!important;color:var(--ms-muted)!important}
[data-testid="stTab"]:hover p{color:var(--ms-text)!important}
[data-testid="stTab"][aria-selected="true"]{box-shadow:inset 0 -3px 0 var(--ms-primary);border-color:var(--ms-primary)!important}
[data-testid="stTab"][aria-selected="true"] p{color:var(--ms-primary)!important;font-weight:700}
[data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--ms-border)}
[data-baseweb="tab-list"] button[role="tab"]{padding:.7rem .9rem}
[data-baseweb="tab-list"] button[role="tab"], [data-baseweb="tab-list"] button[role="tab"] *{opacity:1!important}
[data-baseweb="tab-list"] button[role="tab"] p{font-size:1.06rem;color:var(--ms-muted)!important}
[data-baseweb="tab-list"] button[role="tab"]:hover p{color:var(--ms-text)!important}
[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p{color:var(--ms-primary)!important;font-weight:700}
[data-baseweb="tab-highlight"]{background:var(--ms-primary)!important;height:3px}
[data-baseweb="tab-border"]{background:transparent!important}

/* ---------- file uploader & audio input ---------- */
[data-testid="stFileUploaderDropzone"], [data-testid="stAudioInput"] > div{background:var(--ms-surface)!important;border:1.5px dashed var(--ms-border)!important;border-radius:12px!important;padding:2.1rem 1.6rem!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ms-primary)!important}
[data-testid="stFileUploaderDropzoneInstructions"] *{color:var(--ms-muted)!important}
[data-testid="stFileChip"]{background:var(--ms-bg)!important;border:1px solid var(--ms-border)!important}
[data-testid="stFileChip"] *{color:var(--ms-text)!important}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] > div{background:var(--ms-bg)!important;border:1px solid var(--ms-border);border-radius:12px}
[data-testid="stFileUploaderFile"] *{color:var(--ms-text)!important}
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg{color:var(--ms-muted)!important;stroke:var(--ms-muted)}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--ms-border)!important}
[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]{border-color:var(--ms-border)!important}

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div, [data-baseweb="base-input"]{background:var(--ms-bg)!important;border-color:var(--ms-border)!important;border-radius:12px!important}
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] *{color:var(--ms-text)!important;-webkit-text-fill-color:var(--ms-text)}
[data-baseweb="input"] > div, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="base-input"] > *{background:var(--ms-bg)!important}
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], [data-testid="stNumberInputContainer"]{background:var(--ms-bg)!important;border-color:var(--ms-border)!important;border-radius:12px!important}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within{border-color:var(--ms-accent)!important}
[data-testid="stTextInputRootElement"] input, [data-testid="stTextAreaRootElement"] textarea{color:var(--ms-text)!important;-webkit-text-fill-color:var(--ms-text)}
input::placeholder, textarea::placeholder{color:var(--ms-muted)!important;-webkit-text-fill-color:var(--ms-muted)!important;opacity:.85}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"]{background:var(--ms-bg)!important;border:1px solid var(--ms-border)!important;border-radius:12px!important}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input{color:var(--ms-text)!important;-webkit-text-fill-color:var(--ms-text)!important;background:transparent!important}
[data-testid="stSelectbox"] button, [data-testid="stMultiSelect"] button{color:var(--ms-muted)!important;background:transparent!important}
[data-testid="stSelectbox"] [role="group"]:focus-within{border-color:var(--ms-accent)!important;box-shadow:0 0 0 2px color-mix(in srgb,var(--ms-accent) 30%,transparent)!important}
[role="listbox"] [role="option"]{color:var(--ms-text)!important;background:var(--ms-surface)!important}
[role="listbox"] [role="option"][data-focused], [role="listbox"] [role="option"]:hover{background:color-mix(in srgb,var(--ms-primary) 16%,var(--ms-surface))!important}
[role="listbox"] [role="option"][aria-selected="true"]{color:var(--ms-primary)!important;font-weight:600}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within{border-color:var(--ms-accent)!important;box-shadow:0 0 0 2px color-mix(in srgb,var(--ms-accent) 30%,transparent)!important}
[data-baseweb="tag"]{background:color-mix(in srgb,var(--ms-primary) 22%,transparent)!important}
[data-baseweb="tag"] *{color:var(--ms-text)!important}
[data-testid="stSlider"] [role="slider"]{background:var(--ms-primary)!important}

/* ---------- buttons ---------- */
button[kind], [data-testid^="stBaseButton"]{border-radius:12px!important;transition:background .15s,border-color .15s,color .15s}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"], [data-testid="stPopoverButton"], [data-testid="stBaseLinkButton-secondary"]{background:var(--ms-surface)!important;color:var(--ms-text)!important;border:1px solid var(--ms-border)!important}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover, [data-testid="stPopoverButton"]:hover, [data-testid="stBaseLinkButton-secondary"]:hover{border-color:var(--ms-accent)!important;color:var(--ms-accent)!important}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]{background:var(--ms-primary)!important;border:1px solid var(--ms-primary)!important;color:var(--ms-on-primary)!important;font-weight:700}
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] *{color:var(--ms-on-primary)!important}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:focus-visible, [data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:focus-visible{background:var(--ms-accent)!important;border-color:var(--ms-accent)!important;color:var(--ms-on-accent)!important}
[data-testid="stBaseButton-primary"]:hover *, [data-testid="stBaseButton-primary"]:focus-visible *, [data-testid="stBaseButton-primaryFormSubmit"]:hover *{color:var(--ms-on-accent)!important}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p{font-weight:700}
[data-testid="stBaseButton-tertiary"]{color:var(--ms-primary)!important}
[data-testid="stBaseButton-tertiary"] *{color:var(--ms-primary)!important}
[data-testid="stBaseButton-tertiary"]:hover *{color:var(--ms-accent)!important}

.st-key-ms_process button{min-height:3.3rem;font-size:1.15rem!important;letter-spacing:.01em}
.st-key-ms_process button p{font-size:1.15rem!important;font-weight:700}
/* disabled buttons: must win over hover/primary styles */
[data-testid="stApp"] button:disabled, [data-testid="stApp"] button:disabled:hover, [data-testid="stApp"] button[disabled]:focus-visible{
  background:color-mix(in srgb,var(--ms-surface) 65%,var(--ms-bg))!important;border:1px solid var(--ms-border)!important;
  opacity:1!important;cursor:not-allowed!important;box-shadow:none!important}
[data-testid="stApp"] button:disabled *, [data-testid="stApp"] button:disabled:hover *{color:var(--ms-muted)!important}
/* segmented control / pills */
[data-testid="stButtonGroup"] button{background:var(--ms-surface)!important;border-color:var(--ms-border)!important;color:var(--ms-text)!important;padding:.5rem 1rem}
[data-testid="stButtonGroup"] button *{color:var(--ms-text)!important;font-size:1rem}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind$="Active"]{background:color-mix(in srgb,var(--ms-primary) 20%,var(--ms-surface))!important;border-color:var(--ms-primary)!important}
[data-testid="stButtonGroup"] button[aria-checked="true"] *, [data-testid="stButtonGroup"] button[kind$="Active"] *{color:var(--ms-primary)!important;font-weight:600}

/* ---------- alerts, status, chat, tables ---------- */
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"]{border-radius:12px!important}
[data-testid="stAlertContentInfo"] *{color:var(--ms-text)!important}
[data-testid="stChatMessage"]{background:var(--ms-surface)!important;border:1px solid var(--ms-border);border-radius:12px}
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td{color:var(--ms-text)!important;border-color:var(--ms-border)!important;background:var(--ms-surface)!important}
[data-testid="stTable"] th{color:var(--ms-muted)!important;font-weight:600}
[data-testid="stTable"]{border-radius:12px;overflow:hidden;border:1px solid var(--ms-border)}
[data-testid="stCode"] pre, code{background:var(--ms-surface)!important;color:var(--ms-text)!important}
[data-testid="stProgress"] > div > div > div > div{background:var(--ms-primary)!important}
[data-testid="stAudio"] audio, audio{border-radius:12px}
/* ---------- loading animation ---------- */
.ms-loader{display:flex;align-items:center;gap:1.4rem;background:var(--ms-surface);border:1px solid var(--ms-border);border-radius:12px;
  padding:1.4rem 1.6rem;margin:.4rem 0 .6rem;box-shadow:0 4px 18px rgba(0,0,0,.10)}
.ms-loader.small{padding:.8rem 1.1rem;gap:1rem}
.ms-wave{display:flex;align-items:center;gap:5px;height:46px;flex:none}
.ms-loader.small .ms-wave{height:28px;gap:4px}
.ms-wave i{display:block;width:6px;height:100%;border-radius:6px;background:var(--ms-primary);animation:ms-wave 1.1s ease-in-out infinite;transform-origin:center}
.ms-loader.small .ms-wave i{width:4px}
.ms-wave i:nth-child(2){animation-delay:-.95s}.ms-wave i:nth-child(3){animation-delay:-.8s}
.ms-wave i:nth-child(4){animation-delay:-.65s;background:var(--ms-accent)}.ms-wave i:nth-child(5){animation-delay:-.5s}
.ms-wave i:nth-child(6){animation-delay:-.35s}.ms-wave i:nth-child(7){animation-delay:-.2s}
@keyframes ms-wave{0%,100%{transform:scaleY(.25);opacity:.55}50%{transform:scaleY(1);opacity:1}}
.ms-load-label{font-size:1.25rem;font-weight:700;color:var(--ms-text)}
.ms-loader.small .ms-load-label{font-size:1.02rem}
.ms-load-steps{display:flex;gap:6px;margin-top:.5rem}
.ms-load-steps span{width:34px;height:5px;border-radius:5px;background:var(--ms-border)}
.ms-load-steps span.on{background:var(--ms-primary)}
.ms-load-steps span.now{background:var(--ms-primary);animation:ms-blink 1.2s ease-in-out infinite}
@keyframes ms-blink{0%,100%{opacity:1}50%{opacity:.35}}
.ms-load-sub{color:var(--ms-muted);font-size:.9rem;margin-top:.35rem}
@media (prefers-reduced-motion: reduce){.ms-wave i,.ms-load-steps span.now{animation:none}}
/* ---------- full loading page ---------- */
.ms-lp{max-width:720px;margin:6vh auto 2rem;text-align:center;padding:2.6rem 2rem 2.2rem;background:var(--ms-surface);
  border:1px solid var(--ms-border);border-radius:22px;box-shadow:0 18px 50px rgba(0,0,0,.18);position:relative;overflow:hidden}
.ms-lp:before{content:"";position:absolute;inset:-40% -20% auto;height:70%;pointer-events:none;
  background:radial-gradient(closest-side,color-mix(in srgb,var(--ms-primary) 22%,transparent),transparent);animation:ms-glow 4s ease-in-out infinite}
@keyframes ms-glow{0%,100%{opacity:.55;transform:scale(1)}50%{opacity:1;transform:scale(1.08)}}
.ms-lp > *{position:relative}
.ms-lp-logo{width:84px;height:84px;border-radius:50%;object-fit:cover;box-shadow:0 0 0 4px color-mix(in srgb,var(--ms-primary) 35%,transparent);
  animation:ms-pulse 2.2s ease-in-out infinite}
@keyframes ms-pulse{0%,100%{box-shadow:0 0 0 4px color-mix(in srgb,var(--ms-primary) 35%,transparent)}
  50%{box-shadow:0 0 0 14px color-mix(in srgb,var(--ms-primary) 0%,transparent)}}
.ms-lp-wave{display:flex;justify-content:center;align-items:center;gap:6px;height:70px;margin:1.4rem 0 1rem}
.ms-lp-wave i{display:block;width:7px;height:100%;border-radius:7px;background:var(--ms-primary);animation:ms-wave 1.1s ease-in-out infinite}
.ms-lp-wave i:nth-child(3n){background:var(--ms-accent)}
.ms-lp-wave i:nth-child(1){animation-delay:-1.05s}.ms-lp-wave i:nth-child(2){animation-delay:-.9s}.ms-lp-wave i:nth-child(3){animation-delay:-.75s}
.ms-lp-wave i:nth-child(4){animation-delay:-.6s}.ms-lp-wave i:nth-child(5){animation-delay:-.45s}.ms-lp-wave i:nth-child(6){animation-delay:-.3s}
.ms-lp-wave i:nth-child(7){animation-delay:-.15s}.ms-lp-wave i:nth-child(8){animation-delay:-.3s}.ms-lp-wave i:nth-child(9){animation-delay:-.45s}
.ms-lp-wave i:nth-child(10){animation-delay:-.6s}.ms-lp-wave i:nth-child(11){animation-delay:-.75s}
.ms-lp-title{font-size:1.75rem;font-weight:800;color:var(--ms-text);margin:0}
.ms-lp-file{color:var(--ms-muted);font-size:.95rem;margin-top:.3rem;word-break:break-all}
.ms-lp-track{display:flex;align-items:flex-start;justify-content:center;margin:1.8rem auto 1.2rem;max-width:520px}
.ms-lp-st{flex:1;display:flex;flex-direction:column;align-items:center;gap:.45rem;position:relative}
.ms-lp-st:not(:last-child):after{content:"";position:absolute;top:17px;left:calc(50% + 22px);right:calc(-50% + 22px);height:3px;border-radius:3px;background:var(--ms-border)}
.ms-lp-st.done:not(:last-child):after{background:var(--ms-primary)}
.ms-lp-dot{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.95rem;
  border:2px solid var(--ms-border);color:var(--ms-muted);background:var(--ms-bg)}
.ms-lp-st.done .ms-lp-dot{background:var(--ms-primary);border-color:var(--ms-primary);color:var(--ms-bg)}
.ms-lp-st.now .ms-lp-dot{border-color:var(--ms-primary);color:var(--ms-primary);animation:ms-pulse 1.6s ease-in-out infinite}
.ms-lp-name{font-size:.85rem;color:var(--ms-muted)}
.ms-lp-st.now .ms-lp-name,.ms-lp-st.done .ms-lp-name{color:var(--ms-text);font-weight:600}
.ms-lp-meta{display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap;margin-top:.4rem}
.ms-lp-chip{font-size:.85rem;color:var(--ms-muted);border:1px solid var(--ms-border);border-radius:999px;padding:.25rem .8rem;background:var(--ms-bg)}
.ms-lp-sub{margin:1rem auto 0;max-width:520px;font-size:.9rem;color:var(--ms-text);background:color-mix(in srgb,var(--ms-accent) 14%,transparent);
  border:1px solid color-mix(in srgb,var(--ms-accent) 40%,transparent);border-radius:10px;padding:.5rem .8rem}
.ms-lp-tip{margin-top:1.4rem;color:var(--ms-muted);font-size:.9rem;font-style:italic}
@media (max-width:640px){.ms-lp{padding:2rem 1rem}.ms-lp-title{font-size:1.4rem}.ms-lp-name{font-size:.75rem}}
@media (prefers-reduced-motion: reduce){.ms-lp-wave i,.ms-lp-logo,.ms-lp:before,.ms-lp-st.now .ms-lp-dot{animation:none}}
/* ---------- Q&A ---------- */
.ms-qa-q{font-weight:700;font-size:1.08rem;color:var(--ms-text);margin-bottom:.35rem}
.ms-qa-a{color:var(--ms-text);line-height:1.6}
.ms-qa-a.nf{color:var(--ms-muted);font-style:italic}
.ms-qa-old .ms-qa-q{font-size:.98rem}
/* transcript boxes (side by side) */
.mss{color:var(--ms-text)}
.mss .ts{color:var(--ms-muted)!important;opacity:1!important}
.mss ins{background:var(--ms-ins)!important}
.mss del{background:var(--ms-del)!important}
.mss .row{border-bottom-color:color-mix(in srgb,var(--ms-border) 70%,transparent)!important}
</style>""",
        unsafe_allow_html=True,
    )


DEMO_AUDIO = SAMPLES_DIR / "demo_meeting.mp3"
DEMO_CONTEXT = {
    "ctx_title": "Payments platform weekly sync",
    "ctx_desc": "Weekly engineering sync of the payments platform team",
    "ctx_people": "Priya, Arjun, Meera",
    "ctx_terms": "Redis, PostgreSQL, Grafana, Kafka, OAuth, TTL, p99, p50, runbook, feature flag",
    "ctx_domains": ["Software engineering"],
}

ss = st.session_state
ss.setdefault("run_dir", None)
ss.setdefault("seek", 0.0)
ss.setdefault("autoplay", False)
ss.setdefault("qa", {})
ss.setdefault("show_new", True)


# --------------------------------------------------------------------------- helpers


def seek_to(t: float) -> None:
    ss.seek = float(t)
    ss.autoplay = True


def jump_button(t: float | None, key: str, hours: bool = False) -> None:
    if t is None:
        return
    st.button(f"▶ {fmt_ts(t, hours)}", key=key, on_click=seek_to, args=(t,), type="tertiary", help="Play the recording from here")


def first_time(run: Run, ids: list[int], start: float | None = None) -> float | None:
    if start is not None:
        return start
    segs = run.refined.segments if run.refined else (run.transcript.segments if run.transcript else [])
    times = [s.start for s in segs if s.id in set(ids)]
    return min(times) if times else None


@st.cache_data(show_spinner=False, max_entries=4)
def b64_file(path: str, mtime: float) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def player_theme_css() -> str:
    t = theme()
    return (f":root{{--bg:{t['bg']};--fg:{t['text']};--muted:{t['muted']};--line:{t['border']};--card:{t['surface']};"
            f"--accent:{t['primary']};--warn:{t['warn']};--ins:{t['ins']};--del:{t['del']}}}"
            f"body{{margin:0;background:{t['bg']}}}")


def current_run() -> Run | None:
    if not ss.run_dir:
        return None
    p = Path(ss.run_dir)
    if not (p / "run.json").exists():
        ss.run_dir = None
        return None
    return Run.load(p)


# --------------------------------------------------------------------------- background jobs
# Long work (processing, retries, Q&A) runs in a background thread, not inside the Streamlit script run.
# Streamlit stops the current script run whenever the user clicks anything, so work done inside it gets
# killed and restarted by any click. In a thread it keeps going; the page just polls it.
# Only one job runs per session: every work button is disabled while ss.busy is set, and requests that
# still slip through are ignored.

STAGE_STEP = {"transcribe": 1, "refine": 2, "minutes": 3}
STAGE_TEXT = {"transcribe": "Transcribing…", "refine": "Cleaning up the transcript…", "minutes": "Writing the minutes…"}
_WAIT_WORDS = ("wait", "rate limit", "pacing", "retrying", "hiccup", "network problem")


@dataclass
class Job:
    kind: str  # "process" | "qa"
    run_dir: str = ""
    label: str = "Getting ready…"
    step: int = 0
    sub: str = ""
    done: bool = False
    error: Optional[str] = None
    result: Any = None
    meta: dict = field(default_factory=dict)
    thread: Optional[threading.Thread] = None
    started: float = field(default_factory=time.time)


def job_running() -> bool:
    j = ss.get("job")
    return j is not None and not j.done and j.thread is not None and j.thread.is_alive()


def busy() -> bool:
    return bool(ss.get("busy")) or job_running()


def request(kind: str, **kw) -> None:
    """on_click callback for every button that starts work. Runs before the rerun, so the rerun
    already draws the buttons disabled. Extra clicks while busy are ignored."""
    if busy():
        return
    ss.busy = True
    ss.pending = {"kind": kind, **kw}


def _launch(job: Job, fn, *args) -> None:
    ss.job = job
    t = threading.Thread(target=fn, args=(job, *args), daemon=True)
    job.thread = t
    t.start()


def _process_worker(job: Job, run: Run, s: Settings, start: str) -> None:
    def on_event(stage: str, kind: str, msg: str) -> None:
        if kind == "start":
            job.step, job.label, job.sub = STAGE_STEP[stage], STAGE_TEXT[stage], ""
        elif kind == "info":
            low = msg.lower()
            if any(w in low for w in _WAIT_WORDS):
                job.sub = msg
            else:
                job.sub = ""
                if "speakers" in low:
                    job.label = "Identifying speakers…"
                elif "checking every" in low:
                    job.label = "Double-checking owners and deadlines…"
                elif "part " in low and " of " in low and stage in STAGE_TEXT:
                    part = msg[msg.find("part "):].split(")")[0]
                    job.label = STAGE_TEXT[stage].rstrip("…") + f" ({part})…"
        elif kind == "error":
            job.error = msg

    try:
        run_pipeline(run, s, on_event, start=start)
    except StageError as e:
        job.error = e.message
    except Exception as e:  # never leave the session stuck
        job.error = str(e)
    finally:
        # count the audio toward the free daily allowance once it has been transcribed
        user, store = job.meta.get("charge"), job.meta.get("store")
        if user and store is not None and start == "transcribe":
            try:
                r = Run.load(Path(job.run_dir))
                if r.transcript is not None:
                    store.charge(user, float(r.duration or job.meta.get("duration") or 0.0))
            except Exception:
                pass
        job.done = True


def _qa_worker(job: Job, question: str, segs, names: dict, s: Settings) -> None:
    try:
        ans = ask(question, segs, names, s, ModelClient(s))
        job.result = {"q": question, "a": ans.text, "found": ans.found, "ids": ans.segment_ids}
    except MeetScribeError as e:
        job.error = e.message + (f" {e.hint}" if e.hint else "")
    except Exception as e:
        job.error = str(e)
    finally:
        job.done = True


def _ctx_from_state() -> MeetingContext:
    return MeetingContext(
        title=(ss.get("ctx_title") or "").strip(), date=(ss.get("ctx_date") or "").strip(),
        description=(ss.get("ctx_desc") or "").strip(), participants=parse_terms(ss.get("ctx_people") or ""),
        glossary=parse_terms(ss.get("ctx_terms") or ""), domains=list(ss.get("ctx_domains") or []),
    )


def _start_new_process(s: Settings, force: bool) -> None:
    src = ss.get("source") or "Upload a file"
    if src == "Try the demo meeting":
        name, data = DEMO_AUDIO.name, DEMO_AUDIO.read_bytes()
    else:
        up = ss.get("upload_rec") if src == "Record now" else ss.get("upload_file")
        if not up:
            ss.busy = False
            return
        name, data = getattr(up, "name", None) or "recording.wav", up.getvalue()
    try:
        validate_upload(name, len(data), s.max_upload_mb)
    except MeetScribeError as e:
        ss.flash = (e.message, e.hint)
        ss.busy = False
        return
    digest = hashlib.sha256(data).hexdigest()
    seen = ss.hashes.get(digest)
    if seen and not force and (Path(seen) / "run.json").exists():
        ss.dup = {"run_dir": seen, "name": name}
        ss.busy = False
        return
    ss.dup = None
    run = new_run(name, s, _ctx_from_state())
    run.source_path.write_bytes(data)
    meta = {"hash": digest, "new": True}
    if ob.uses_free_key(s):
        store = usage_store(s)
        if store.enabled:
            user = (ss.get("user") or {}).get("email", "local")
            dur = probe_duration(run.source_path)
            left = store.remaining(user)
            if dur is not None and not store.allows(user, dur):
                shutil.rmtree(run.dir, ignore_errors=True)
                ss.flash = (f"This recording is {fmt_minutes(dur)} long, but you have {fmt_minutes(left)} of free audio left today.",
                            "Try a shorter clip, come back tomorrow (the allowance resets at midnight, India time), "
                            "or use your own API key: Change setup in the sidebar.")
                ss.busy = False
                return
            meta.update(charge=user, store=store, duration=dur or 0.0)
    run.save()
    ss.qa = {}
    ss.seek = 0.0
    _launch(Job(kind="process", run_dir=str(run.dir), meta=meta), _process_worker, run, s, "transcribe")


def handle_pending(s: Settings) -> None:
    p = ss.get("pending")
    ss.pending = None
    if not p:
        return
    if job_running():
        return
    try:
        if p["kind"] == "process_new":
            _start_new_process(s, force=bool(p.get("force")))
        elif p["kind"] == "process":
            run = Run.load(Path(p["run_dir"]))
            if p.get("names") is not None:
                run.speaker_names = p["names"]
                run.save()
            meta = {"new": False}
            if p["start"] == "transcribe" and ob.uses_free_key(s) and usage_store(s).enabled:
                meta.update(charge=(ss.get("user") or {}).get("email", "local"), store=usage_store(s))
            _launch(Job(kind="process", run_dir=p["run_dir"], meta=meta), _process_worker, run, s, p["start"])
        elif p["kind"] == "qa":
            run = Run.load(Path(p["run_dir"]))
            segs = run.refined.segments if run.refined else run.transcript.segments
            _launch(Job(kind="qa", run_dir=p["run_dir"], label="Reading the transcript…",
                        meta={"run_id": p["run_id"], "replace": p.get("replace", False), "qkey": p["qkey"]}),
                    _qa_worker, p["q"], segs, run.speaker_names, s)
        else:
            ss.busy = False
    except Exception as e:
        ss.flash = (f"Couldn't start: {e}", "")
        ss.busy = False


def _norm_q(q: str) -> str:
    return " ".join(q.lower().strip().rstrip("?.! ").split())


def finish_job() -> None:
    """Called at the top of every run: if the background job is finished, apply its result and
    clear the busy flag (it is always cleared here, even if the job failed)."""
    j = ss.get("job")
    if j is None:
        if ss.get("busy") and not ss.get("pending"):
            ss.busy = False  # stale flag, nothing is running
        return
    if j.thread is not None and j.thread.is_alive() and not j.done:
        return
    try:
        if j.kind == "process":
            ss.run_dir = j.run_dir
            ss.show_new = False
            if j.meta.get("new"):
                ss.seek = 0.0
            try:
                r = Run.load(Path(j.run_dir))
                if r.record and not r.errors and j.meta.get("hash"):
                    ss.hashes[j.meta["hash"]] = j.run_dir
            except Exception:
                pass
        elif j.kind == "qa":
            rid = j.meta["run_id"]
            hist = ss.qa.setdefault(rid, [])
            if j.result:
                if j.meta.get("replace"):
                    hist[:] = [h for h in hist if _norm_q(h["q"]) != _norm_q(j.result["q"])]
                hist.insert(0, j.result)
                ss[j.meta["qkey"]] = ""
            elif j.error:
                ss.qa_error = (rid, j.error)
    finally:
        ss.job = None
        ss.busy = False


@st.fragment(run_every=0.7)
def job_monitor(kinds: tuple, small: bool = False) -> None:
    """Live loader for the running job. Re-runs every 0.7 s on its own; when the job is done it
    triggers one full rerun so the results replace the loader."""
    j = ss.get("job")
    if j is None or j.kind not in kinds:
        return
    if j.done or not (j.thread and j.thread.is_alive()):
        st.rerun()
        return
    render_loader(j, small)


LOAD_TIPS = [
    "Owners and deadlines are only filled in when someone actually said them.",
    "Every decision and task comes with a quote you can click to hear.",
    "Adding participant names and terms next time helps the transcript a lot.",
    "Long meetings are handled in parts, so the free tier may pause for a few seconds.",
    "Numbers, negations and names are protected: refinement can't change them.",
    "Ideas nobody agreed to go under 'Proposals not agreed', not decisions.",
]
LOAD_STEPS = ["Transcribe", "Refine", "Minutes"]


@st.fragment(run_every=0.7)
def loading_page() -> None:
    """Full-screen loader shown instead of the normal page while a meeting is being processed."""
    j = ss.get("job")
    if j is None or j.done or not (j.thread and j.thread.is_alive()):
        st.rerun()
        return
    el = int(time.time() - j.started)
    logo = f"<img class='ms-lp-logo' src='data:image/png;base64,{LOGO_B64}' alt=''>" if LOGO_B64 else ""
    track = []
    for i, name in enumerate(LOAD_STEPS, start=1):
        cls = "done" if i < j.step else ("now" if i == j.step else "")
        mark = "✓" if cls == "done" else str(i)
        track.append(f"<div class='ms-lp-st {cls}'><div class='ms-lp-dot'>{mark}</div><div class='ms-lp-name'>{name}</div></div>")
    fname = ""
    try:
        fname = Run.load(Path(j.run_dir)).audio_name if j.run_dir else ""
    except Exception:
        pass
    chips = [f"⏱ {el // 60}:{el % 60:02d} elapsed", f"Step {max(j.step, 1)} of 3"]
    sub = f"<div class='ms-lp-sub'>{html.escape(j.sub)}</div>" if j.sub else ""
    tip = LOAD_TIPS[(el // 8) % len(LOAD_TIPS)]
    st.markdown(
        f"<div class='ms-lp' role='status' aria-live='polite'>{logo}"
        f"<div class='ms-lp-wave'>{'<i></i>' * 11}</div>"
        f"<h2 class='ms-lp-title'>{html.escape(j.label)}</h2>"
        + (f"<div class='ms-lp-file'>{html.escape(fname)}</div>" if fname else "")
        + f"<div class='ms-lp-track'>{''.join(track)}</div>"
        f"<div class='ms-lp-meta'>{''.join(f'<span class=ms-lp-chip>{c}</span>' for c in chips)}</div>"
        f"{sub}<div class='ms-lp-tip'>💡 {html.escape(tip)}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='text-align:center;color:var(--ms-muted);font-size:.85rem'>You can leave this tab open and come back. "
                "The results open here by themselves when they're ready.</p>", unsafe_allow_html=True)


def render_loader(j: Job, small: bool = False) -> None:
    steps = ""
    if j.kind == "process":
        cells = []
        for i in range(1, 4):
            cls = "on" if i < j.step else ("now" if i == j.step else "")
            cells.append(f"<span class='{cls}'></span>")
        steps = f"<div class='ms-load-steps'>{''.join(cells)}</div>"
    sub = f"<div class='ms-load-sub'>{html.escape(j.sub)}</div>" if j.sub else ""
    st.markdown(
        f"<div class='ms-loader{' small' if small else ''}' role='status' aria-live='polite'>"
        f"<div class='ms-wave'>{'<i></i>' * 7}</div>"
        f"<div><div class='ms-load-label'>{html.escape(j.label)}</div>{steps}{sub}</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- sidebar: settings


def server_groq_key() -> str:
    """The app's own (free, preloaded) Groq key from .env or Streamlit secrets."""
    k = Settings.from_env().api_key
    if not k:
        try:
            k = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            k = ""
    return k


def usage_store(s: Settings) -> UsageStore:
    return UsageStore(s.usage_file, s.free_daily_minutes * 60, s.quota_tz)


def account_sidebar(s: Settings) -> None:
    user = ss.get("user") or {}
    with st.container(key="ms_account"):
        if not user.get("local"):
            st.markdown(f"**{html.escape(user.get('name', ''))}**  \n<span style='color:var(--ms-muted);font-size:.85rem'>"
                        f"{html.escape(user.get('email', ''))}</span>", unsafe_allow_html=True)
        st.caption(ob.mode_label())
        if ob.uses_free_key(s):
            store = usage_store(s)
            if store.enabled:
                left = store.remaining(user.get("email", "local"))
                st.progress(min(1.0, left / store.limit), text=f"Free key: {fmt_minutes(left)} of audio left today")
        c = st.columns(2)
        if c[0].button("Change setup", width="stretch", disabled=busy(), key="ms_change_setup"):
            ss.mode = None
            ss.own_ready = False
            st.rerun()
        if not user.get("local") and c[1].button("Sign out", width="stretch", disabled=busy(), key="ms_signout"):
            ob.sign_out()
    st.divider()


def build_settings() -> Settings:
    s = Settings.from_env()
    user = ss.get("user") or {"email": "local", "local": True}
    if not user.get("local"):
        s.runs_dir = s.runs_dir / ob.user_folder(user)  # each person only sees their own meetings
    s = ob.apply_mode(s, server_groq_key())
    own = ss.get("mode") == "own"
    with st.sidebar:
        if LOGO_B64:
            st.markdown(f"<div class='ms-side-logo'><img src='data:image/png;base64,{LOGO_B64}' alt=''><b>LazyMeets</b></div>",
                        unsafe_allow_html=True)
        account_sidebar(s)
        run = current_run()
        if run and run.playback_path.exists() and not ss.show_new:
            st.markdown("#### 🎧 Recording")
            st.caption(run.audio_name)
            st.audio(run.playback_path.read_bytes(), format="audio/mpeg", start_time=int(ss.seek), autoplay=ss.autoplay)
            ss.autoplay = False
            st.divider()

        st.markdown("#### Settings")
        with st.expander("Models"):
            if own:
                o = ob._own()
                ms = o["models"] or [s.refine_model, s.minutes_model]
                st.caption(f"Language models on {PROVIDERS[o['provider']].name}")
                s.refine_model = st.selectbox("Language model #1: refinement", ms, index=_idx(ms, s.refine_model), key="sb_refine")
                s.minutes_model = st.selectbox("Language model #2: minutes & tasks", ms, index=_idx(ms, s.minutes_model), key="sb_minutes")
                o["refine"], o["minutes"] = s.refine_model, s.minutes_model
                if s.stt_backend == "api":
                    st.caption(f"Speech-to-text: {s.stt_model}")
                else:
                    st.caption("Speech-to-text: offline (Moonshine)")
            else:
                stt_mode = st.radio(
                    "Speech-to-text",
                    ["Whisper via API (most accurate)", "Offline on this machine (Moonshine)"],
                    index=0 if s.stt_backend == "api" else 1, key="sb_stt_mode",
                    help="Offline mode runs a small model on the CPU. It's slower and less accurate, but no audio leaves the machine.",
                )
                s.stt_backend = "api" if stt_mode.startswith("Whisper") else "local"
                if s.stt_backend == "api":
                    s.stt_model = st.selectbox("Whisper model", STT_MODEL_CHOICES, index=_idx(STT_MODEL_CHOICES, s.stt_model), key="sb_stt")
                s.refine_model = st.selectbox("Language model #1: refinement", LLM_MODEL_CHOICES,
                                              index=_idx(LLM_MODEL_CHOICES, s.refine_model), key="sb_refine_free")
                s.minutes_model = st.selectbox("Language model #2: minutes & tasks", LLM_MODEL_CHOICES,
                                               index=_idx(LLM_MODEL_CHOICES, s.minutes_model), key="sb_minutes_free")
            if s.refine_model == s.minutes_model:
                st.warning("Both stages use the same model, so they share one rate limit. Pick two different models.")
            if st.button("Check keys & models", width="stretch", disabled=busy()):
                check_key(s)

        with st.expander("Speakers"):
            s.diarize = st.toggle("Detect who is speaking", value=s.diarize,
                                  help="Runs offline speaker detection (pyannote + TitaNet). Lets the minutes attribute \"I'll do it\" to a person. Adds some processing time.")
            n = st.number_input("Number of speakers (0 = auto)", min_value=0, max_value=12, value=0, disabled=not s.diarize)
            s.num_speakers = int(n)

        with st.expander("Advanced"):
            s.refine_reasoning = st.select_slider("Refinement reasoning effort", REASONING_CHOICES, value=s.refine_reasoning)
            s.minutes_reasoning = st.select_slider("Minutes reasoning effort", REASONING_CHOICES, value=s.minutes_reasoning)
            s.temperature = st.slider("Temperature", 0.0, 1.0, float(s.temperature), 0.05)
            tpm = st.number_input("Tokens per request budget (0 = auto-detect)", min_value=0, value=int(s.tpm_budget), step=1000,
                                  help="Groq's free tier allows 8,000 tokens per minute. Long meetings are then processed in parts.")
            s.tpm_budget = int(tpm)

        history_sidebar(s)
        st.divider()
        st.caption(f"LazyMeets {__version__} · speech-to-text → refinement → minutes")
    return s


def _idx(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def check_key(s: Settings) -> None:
    try:
        client = ModelClient(s)
        need = {"refinement": s.refine_model, "minutes": s.minutes_model}
        if s.stt_backend == "api":
            need["speech-to-text"] = s.stt_model
        client.preflight(need)
        st.success("Keys work. All selected models are available.")
    except MeetScribeError as e:
        st.error(e.message)
        if e.hint:
            st.caption(e.hint)


def history_sidebar(s: Settings) -> None:
    runs = list_runs(s) if s.persist_runs else []
    if not runs:
        return
    with st.expander(f"Past meetings ({len(runs)})"):
        for p in runs[:30]:
            try:
                d = json.loads((p / "run.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            title = (d.get("record") or {}).get("title") or d.get("audio_name")
            status = "✅" if d.get("record") else ("⚠️" if d.get("errors") else "⏳")
            if st.button(f"{status} {title}", key=f"hist-{p.name}", help=f"{d.get('audio_name')} · {d.get('created', '')}", width="stretch",
                         disabled=busy()):
                ss.run_dir = str(p)
                ss.show_new = False
                ss.seek = 0.0
                st.rerun()


# --------------------------------------------------------------------------- new meeting form


CTX_KEYS = ("ctx_title", "ctx_date", "ctx_desc", "ctx_people", "ctx_terms", "ctx_domains")


def _keep_ctx() -> None:
    """Streamlit forgets a widget's value when the widget isn't drawn (results or loading page).
    Keep a copy so the names and terms are still filled in for the next meeting."""
    for k in CTX_KEYS:
        if k in ss:
            ss["_keep_" + k] = ss[k]


def context_form(disabled: bool = False) -> None:
    for k in CTX_KEYS:
        if k not in ss and ("_keep_" + k) in ss:
            ss[k] = ss["_keep_" + k]
    with st.expander("Help the AI with names and terms (optional, but it improves accuracy)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Meeting title", key="ctx_title", placeholder="e.g. Sprint 14 planning", disabled=disabled)
            st.text_input("Date", key="ctx_date", placeholder="e.g. 7 Oct 2026", disabled=disabled)
            st.text_input("What is the meeting about?", key="ctx_desc", placeholder="e.g. Weekly sync of the payments team", disabled=disabled)
        with c2:
            st.text_input("Participants", key="ctx_people", placeholder="Priya, Arjun, Meera", disabled=disabled,
                          help="Used to spell names right. Names are never assigned as owners unless the recording says so.")
            st.text_area("Glossary: terms, acronyms, product names", key="ctx_terms", height=68, disabled=disabled,
                         placeholder="Kubernetes, PostgreSQL, OKR, Project Falcon…")
            st.multiselect("Add common terms for a domain", list(PRESETS), key="ctx_domains", disabled=disabled)
    _keep_ctx()


def load_demo_context() -> None:
    for k, v in DEMO_CONTEXT.items():
        ss[k] = v


def new_meeting_view(s: Settings) -> None:
    logo = f"<img class='ms-logo' src='data:image/png;base64,{LOGO_B64}' alt='LazyMeets logo'>" if LOGO_B64 else ""
    st.markdown(
        f"<div class='ms-hero'>{logo}<div><h1 class='ms-brand'>Lazy<span>Meets</span></h1>"
        "<p class='ms-tag'>Upload a meeting recording and get a transcript, a corrected transcript, and minutes with decisions "
        "and action items. Owners and deadlines are only filled in when someone actually said them.</p></div></div>",
        unsafe_allow_html=True,
    )
    b = busy()
    if ss.get("source") not in (None, "Upload a file", "Record now"):
        ss.source = "Upload a file"
    source = st.segmented_control("Audio source", ["Upload a file", "Record now"], default="Upload a file",
                                  label_visibility="collapsed", key="source", disabled=b)
    upload = None
    if source == "Record now":
        upload = st.audio_input("Record the meeting with your microphone", key="upload_rec", disabled=b)
    else:
        upload = st.file_uploader("Meeting recording", type=None, key="upload_file", disabled=b,
                                  help="mp3, wav, m4a, mp4, ogg, webm, flac, aac… up to a few hours long.")
    context_form(disabled=b)

    flash = ss.get("flash")
    if flash:
        st.error(flash[0])
        if flash[1]:
            st.caption(flash[1])
        ss.flash = None
    dup = ss.get("dup")
    if dup and not b:
        st.info(f"You already processed this exact recording ({dup['name']}) in this session.")
        c = st.columns(2)
        c[0].button("Open the existing result", on_click=_open_existing, args=(dup["run_dir"],), width="stretch")
        c[1].button("Process it again anyway", on_click=request, args=("process_new",), kwargs={"force": True}, width="stretch")

    ready = bool(upload)
    no_key = not s.api_key
    with st.container(key="ms_process"):
        st.button("Processing…" if b else "Process meeting", type="primary", width="stretch",
                  disabled=b or not ready or no_key, on_click=request, args=("process_new",))
    if no_key:
        st.caption("No API key is set up. Use Change setup in the sidebar.")
    elif ob.uses_free_key(s) and usage_store(s).enabled:
        left = usage_store(s).remaining((ss.get("user") or {}).get("email", "local"))
        st.caption(f"Free key: {fmt_minutes(left)} of audio left today.")
    job_monitor(("process",))


def _open_existing(run_dir: str) -> None:
    ss.run_dir = run_dir
    ss.show_new = False
    ss.dup = None
    ss.seek = 0.0


# --------------------------------------------------------------------------- results


def stage_row(run: Run) -> None:
    cols = st.columns(3)
    for col, stage in zip(cols, STAGES):
        with col:
            err = run.errors.get(stage)
            if err:
                icon, msg = "❌", err["message"]
            elif run.done(stage):
                icon, msg = "✅", _summary(run, stage)
            else:
                icon, msg = "⏸️", "not run yet"
            model = {"transcribe": run.models.get("speech_to_text"), "refine": run.models.get("refinement"),
                     "minutes": run.models.get("minutes")}[stage]
            t = run.timings.get(stage)
            st.markdown(
                f"<div class='ms-stage'><div class='t'>{icon} {STAGE_LABELS[stage]}</div>"
                f"<div class='m'>{model or ''}{f' · {t:.0f}s' if t else ''}</div><div class='m'>{msg}</div></div>",
                unsafe_allow_html=True,
            )


def _summary(run: Run, stage: str) -> str:
    if stage == "transcribe":
        t = run.transcript
        sp = f" · {len(t.speakers())} speakers" if t.diarized else ""
        return f"{len(t.segments)} segments · {len(t.plain_text().split())} words{sp}"
    if stage == "refine":
        a = len(run.refined.applied())
        b = sum(1 for c in run.refined.corrections if c.status == "blocked")
        return f"{a} corrections" + (f" · {b} blocked" if b else "")
    r = run.record
    return f"{len(r.decisions)} decisions · {len(r.action_items)} action items"


def error_panel(run: Run, s: Settings) -> None:
    for stage in STAGES:
        err = run.errors.get(stage)
        if not err:
            continue
        st.error(f"**{STAGE_LABELS[stage]} failed:** {err['message']}")
        if err.get("hint"):
            st.caption("💡 " + err["hint"])
        if err.get("detail"):
            with st.expander("Technical details"):
                st.code(err["detail"][-2500:])
        if err.get("kind") in ("AudioError", "NoSpeechError"):
            if st.button("Try a different file", type="primary", key=f"newfile-{stage}", disabled=busy()):
                ss.show_new = True
                st.rerun()
            break
        can_retry = stage == "transcribe" or run.done(STAGES[STAGES.index(stage) - 1])
        if can_retry:
            st.button(f"Retry from {STAGE_LABELS[stage].lower()}", type="primary", key=f"retry-{stage}",
                      disabled=busy() or (not s.api_key and s.stt_backend == "api"),
                      on_click=request, args=("process",), kwargs={"run_dir": str(run.dir), "start": stage})
        if run.done("transcribe"):
            st.caption("Results from the steps that finished are still shown below and can be downloaded.")
        break


def results_view(run: Run, s: Settings) -> None:
    top = st.columns([6, 1.3, 1.3])
    with top[0]:
        title = run.record.title if run.record else (run.context.title or Path(run.audio_name).stem)
        st.markdown(f"<h1 class='ms-title'>{title}</h1>", unsafe_allow_html=True)
        bits = [run.audio_name, f"{fmt_ts(run.duration)} long" if run.duration else None, run.context.date or None]
        st.caption(" · ".join(b for b in bits if b))
    b = busy()
    with top[1]:
        if st.button("➕ New meeting", width="stretch", disabled=b):
            ss.show_new = True
            ss.seek = 0.0
            st.rerun()
    with top[2]:
        if run.transcript and not b:
            st.download_button("⬇ Download all", run.zip_bytes(), file_name=f"{run.run_id}.zip", mime="application/zip",
                               width="stretch", type="primary", on_click="ignore")
    j = ss.get("job")
    if j is not None and j.kind == "process":
        # retry / regenerate in progress: show only the loader, no half-updated results
        job_monitor(("process",))
        return
    stage_row(run)
    st.write("")
    error_panel(run, s)
    if not run.transcript:
        return

    names = run.speaker_names
    tabs = ["📋 Meeting record", "📝 Transcripts", "🔍 Refinement changes", "💬 Ask the meeting"]
    if run.transcript.diarized:
        tabs.append("🗣️ Speakers")
    tabs += ["⬇️ Downloads", "⚙️ Run details"]
    t = dict(zip(tabs, st.tabs(tabs)))
    with t["📋 Meeting record"]:
        tab_record(run)
    with t["📝 Transcripts"]:
        tab_transcripts(run, names)
    with t["🔍 Refinement changes"]:
        tab_changes(run)
    with t["💬 Ask the meeting"]:
        tab_ask(run, s, names)
    if "🗣️ Speakers" in t:
        with t["🗣️ Speakers"]:
            tab_speakers(run, s)
    with t["⬇️ Downloads"]:
        tab_downloads(run)
    with t["⚙️ Run details"]:
        tab_details(run)


def _unspec(v: str | None) -> str:
    return v if v else "<span class='ms-unspec'>Unspecified</span>"


def tab_record(run: Run) -> None:
    r = run.record
    if not r:
        st.info("The meeting record hasn't been generated yet. See the status above.")
        return
    hours = run.duration >= 3600
    if r.participants:
        st.caption("Participants: " + ", ".join(r.participants))
    st.markdown(f"**Summary.** {r.summary}")
    m = st.columns(4)
    m[0].metric("Decisions", len(r.decisions))
    m[1].metric("Action items", len(r.action_items))
    m[2].metric("Open questions", len(r.open_questions))
    m[3].metric("Terms corrected", len(run.refined.applied()) if run.refined else 0)

    st.subheader("Key decisions", anchor=False)
    if not r.decisions:
        st.caption("No decisions were reached in this meeting.")
    for d in r.decisions:
        c = st.container(border=True).columns([12, 1.4])
        with c[0]:
            st.markdown(f"**{d.id}.** {d.decision}")
            if d.evidence_quote:
                st.markdown(f"<div class='ms-quote'>“{d.evidence_quote}”</div>", unsafe_allow_html=True)
            for f in d.flags:
                st.markdown(f"<div class='ms-flag'>⚠ {f}</div>", unsafe_allow_html=True)
        with c[1]:
            jump_button(first_time(run, d.segment_ids, d.start), f"d-{d.id}", hours)

    st.subheader("Action items", anchor=False)
    if not r.action_items:
        st.caption("No action items were assigned.")
    else:
        widths = [0.8, 5.4, 1.8, 1.8, 1.8, 1.4]
        h = st.columns(widths)
        for col, lab in zip(h, ["#", "Task", "Target", "Owner", "Deadline", "Said at"]):
            col.markdown(f"<span style='opacity:.6;font-size:.8rem;text-transform:uppercase'>{lab}</span>", unsafe_allow_html=True)
        for a in r.action_items:
            c = st.container(border=True).columns(widths)
            c[0].markdown(f"**{a.id}**")
            with c[1]:
                st.markdown(a.task)
                if a.evidence_quote:
                    st.markdown(f"<div class='ms-quote'>“{a.evidence_quote}”</div>", unsafe_allow_html=True)
                for f in a.flags:
                    st.markdown(f"<div class='ms-flag'>⚠ {f}</div>", unsafe_allow_html=True)
            c[2].markdown(html.escape(a.target) if a.target else "<span class='ms-unspec'>—</span>", unsafe_allow_html=True)
            c[3].markdown(_unspec(a.owner), unsafe_allow_html=True)
            c[4].markdown(_unspec(a.deadline), unsafe_allow_html=True)
            with c[5]:
                jump_button(first_time(run, a.segment_ids, a.start), f"a-{a.id}", hours)

    if r.suggested_followups:
        st.subheader("Suggested follow-ups (not confirmed)", anchor=False)
        st.caption("These came up, but nobody committed to them, so they aren't in the action items.")
        for a in r.suggested_followups:
            c = st.columns([12, 1.4])
            c[0].markdown(f"- {a.task}" + "".join(f"  \n  <span class='ms-flag'>{f}</span>" for f in a.flags), unsafe_allow_html=True)
            with c[1]:
                jump_button(first_time(run, a.segment_ids, a.start), f"s-{a.id}", hours)

    if r.proposals:
        st.subheader("Proposals not agreed", anchor=False)
        for p in r.proposals:
            c = st.columns([12, 1.4])
            c[0].markdown(f"- {p.proposal} · *{p.status}*")
            with c[1]:
                jump_button(first_time(run, p.segment_ids, p.start), f"p-{p.id}", hours)

    if r.open_questions:
        st.subheader("Open questions", anchor=False)
        for i, q in enumerate(r.open_questions):
            c = st.columns([12, 1.4])
            c[0].markdown(f"- {q.question}")
            with c[1]:
                jump_button(first_time(run, q.segment_ids), f"q-{i}", hours)

    st.subheader("Minutes", anchor=False)
    for i, sec in enumerate(r.minutes):
        c = st.columns([12, 1.4])
        with c[0]:
            st.markdown(f"**{sec.topic}**\n" + "\n".join(f"- {p}" for p in sec.points))
        with c[1]:
            jump_button(first_time(run, sec.segment_ids), f"m-{i}", hours)

    with st.expander(f"Automatic checks ({len(r.verification_notes)})"):
        st.caption("After the model writes the record, the app checks every quote, owner and deadline against the transcript. "
                   "This is what it changed:")
        if r.verification_notes:
            for n in r.verification_notes:
                st.markdown(f"- {n}")
        else:
            st.markdown("Nothing needed changing. Every item was backed by the transcript.")


def tab_transcripts(run: Run, names: dict) -> None:
    view = st.segmented_control("View", ["Side by side", "Follow along with audio"], default="Side by side",
                                key="tview", label_visibility="collapsed")
    raw = run.transcript.segments
    refined = run.refined.segments if run.refined else raw
    if view == "Follow along with audio":
        audio = b64_file(str(run.playback_path), run.playback_path.stat().st_mtime) if run.playback_path.exists() else None
        components.html(player_html(refined, raw, names, audio, theme_css=player_theme_css()), height=720, scrolling=True)
        return
    left, right = side_by_side_html(raw, refined, names)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Raw transcript** · {run.transcript.model}")
        st.caption("Straight from speech-to-text. Dotted underline means low recognizer confidence. Red marks words the refinement replaced.")
        with st.container(height=620):
            st.html(left)
    with c2:
        st.markdown(f"**Refined transcript** · {run.refined.model if run.refined else '(not refined yet)'}")
        st.caption("After domain-term correction. Green marks the new words. Timestamps and wording are otherwise unchanged.")
        with st.container(height=620):
            st.html(right)
    if run.transcript.dropped:
        with st.expander(f"Removed as noise ({len(run.transcript.dropped)})"):
            st.caption("Whisper sometimes invents text over silence or noise. These segments were left out of the transcript:")
            for d in run.transcript.dropped:
                st.markdown(f"- `{fmt_ts(d.start)}` “{d.text}”: {d.reason}")


def tab_changes(run: Run) -> None:
    if not run.refined:
        st.info("Refinement hasn't run yet.")
        return
    cs = run.refined.corrections
    counts = {k: sum(1 for c in cs if c.status == k) for k in ("applied", "propagated", "blocked", "not_found")}
    m = st.columns(4)
    m[0].metric("Applied", counts["applied"])
    m[1].metric("Same fix elsewhere", counts["propagated"])
    m[2].metric("Blocked by checks", counts["blocked"])
    m[3].metric("Couldn't locate", counts["not_found"])
    st.caption(
        "Language model #1 only proposes small replacements, for example a misheard term. Before applying one, the app blocks it "
        "if it would change a number, a negation (not/never…), a commitment word (will/might/should…) or a participant's name, "
        "or if the new words don't sound like the old ones."
    )
    if not cs:
        st.success("No corrections were needed. The raw transcript already had the terms right.")
        return
    idx = {s.id: s for s in run.refined.segments}
    icon = {"applied": "✅ applied", "propagated": "✅ same fix", "blocked": "⛔ blocked", "not_found": "❔ not found", "unchanged": "·"}
    df = pd.DataFrame(
        [
            {
                "When": fmt_ts(idx[c.segment_id].start) if c.segment_id in idx else "",
                "Status": icon.get(c.status, c.status),
                "Heard as": c.original,
                "Corrected to": c.corrected,
                "Type": c.category.replace("_", " "),
                "Why": c.block_reason if c.status in ("blocked", "not_found") else c.reason,
            }
            for c in cs
        ]
    )
    st.table(df.set_index("When"))


QA_EXAMPLES = ["What was decided?", "Who is doing what, and by when?", "What numbers or metrics were mentioned?",
               "What was left unresolved?"]


def _fill_question(pkey: str, qkey: str) -> None:
    """A suggestion chip only fills the question box; it never submits."""
    v = ss.get(pkey)
    if v:
        ss[qkey] = v
    ss[pkey] = None


def _request_ask(run_id: str, run_dir: str, qkey: str) -> None:
    if busy():
        return
    q = (ss.get(qkey) or "").strip()
    if not q:
        return
    hist = ss.qa.setdefault(run_id, [])
    n = _norm_q(q)
    idx = next((i for i, h in enumerate(hist) if _norm_q(h["q"]) == n), None)
    notes = ss.setdefault("qa_note", {})
    if idx is not None and ss.get("qa_repeat") != (run_id, n):
        # already answered: bring the earlier answer to the top instead of asking again
        hist.insert(0, hist.pop(idx))
        ss.qa_repeat = (run_id, n)
        notes[run_id] = "You already asked this, so here's the earlier answer. Click Ask again for a fresh one."
        return
    ss.qa_repeat = None
    notes.pop(run_id, None)
    request("qa", run_id=run_id, run_dir=run_dir, q=q, qkey=qkey, replace=idx is not None)


def _clear_qa(run_id: str) -> None:
    ss.qa[run_id] = []
    ss.setdefault("qa_note", {}).pop(run_id, None)
    ss.qa_repeat = None


def _render_answer(run: Run, h: dict, key: str, old: bool = False) -> None:
    cls = "ms-qa-a" + ("" if h["found"] else " nf")
    st.markdown(f"<div class='{'ms-qa-old' if old else ''}'><div class='ms-qa-q'>{html.escape(h['q'])}</div>"
                f"<div class='{cls}'>{html.escape(h['a'])}</div></div>", unsafe_allow_html=True)
    if h["ids"]:
        cols = st.columns(min(len(h["ids"]), 6) + 2)
        for j, sid in enumerate(h["ids"][:6]):
            with cols[j]:
                jump_button(first_time(run, [sid]), f"{key}-{j}")


def tab_ask(run: Run, s: Settings, names: dict) -> None:
    rid = run.run_id
    qkey, pkey = f"qa-q-{rid}", f"qa-ex-{rid}"
    hist = ss.qa.setdefault(rid, [])
    b = busy()
    asking = b and ss.get("job") is not None and ss.job.kind == "qa"
    st.caption(f"Ask anything about the meeting. Answers come only from the transcript ({s.minutes_model}), with timestamps. "
               "Pick a suggestion to fill the box, then press Ask.")
    st.pills("Suggestions", QA_EXAMPLES, label_visibility="collapsed", key=pkey, on_change=_fill_question, args=(pkey, qkey),
             disabled=b)
    st.text_input("Your question", key=qkey, placeholder="e.g. What did we decide about the cache?", disabled=b)
    c = st.columns([1.3, 1.1, 7])
    with c[0]:
        st.button("Asking…" if asking else "Ask", type="primary", key=f"qa-ask-{rid}", width="stretch",
                  disabled=b or not s.api_key, on_click=_request_ask, args=(rid, str(run.dir), qkey))
    with c[1]:
        st.button("Clear", key=f"qa-clear-{rid}", width="stretch", disabled=b or not hist, on_click=_clear_qa, args=(rid,))
    if not s.api_key:
        st.caption("Add your API key in the sidebar first.")
    note = ss.get("qa_note", {}).get(rid)
    if note:
        st.caption(note)
    err = ss.get("qa_error")
    if err and err[0] == rid:
        st.error(err[1])
        ss.qa_error = None
    job_monitor(("qa",), small=True)
    if hist:
        with st.container(border=True):
            _render_answer(run, hist[0], f"qa-now-{rid}")
    if len(hist) > 1:
        with st.expander(f"Previous questions ({len(hist) - 1})", expanded=False):
            for i, h in enumerate(hist[1:], start=1):
                _render_answer(run, h, f"qa-old-{rid}-{i}", old=True)
                if i < len(hist) - 1:
                    st.divider()


def tab_speakers(run: Run, s: Settings) -> None:
    segs = run.transcript.segments
    stats = {}
    for x in segs:
        if not x.speaker:
            continue
        d = stats.setdefault(x.speaker, {"time": 0.0, "turns": 0, "first": x.text, "start": x.start})
        d["time"] += max(0.0, x.end - x.start)
        d["turns"] += 1
    total = sum(v["time"] for v in stats.values()) or 1
    st.caption("Speakers were detected from voices, so the labels are a best guess. Give them real names to use in the "
               "transcripts and exports. Then regenerate the minutes so \"I'll do it\" commitments get the right owner.")
    with st.form(f"names-{run.run_id}"):
        new_names = {}
        for spk, d in sorted(stats.items(), key=lambda kv: kv[1]["start"]):
            c = st.columns([2, 3, 5])
            with c[0]:
                new_names[spk] = st.text_input(spk, value=run.speaker_names.get(spk, ""), placeholder="Name", key=f"nm-{run.run_id}-{spk}")
            c[1].progress(d["time"] / total, text=f"{d['time'] / total:.0%} of talk time · {d['turns']} turns")
            c[2].caption(f"First said: “{d['first'][:110]}”")
        b1, b2 = st.columns(2)
        save = b1.form_submit_button("Save names", disabled=busy())
        b2.form_submit_button("Save names & regenerate minutes", type="primary", disabled=busy() or not run.refined,
                              on_click=_regen_with_names, args=(str(run.dir), run.run_id, list(new_names)))
    if save and not busy():
        run.speaker_names = {k: v.strip() for k, v in new_names.items() if v.strip()}
        run.save()
        st.rerun()


def _regen_with_names(run_dir: str, run_id: str, speakers: list[str]) -> None:
    names = {spk: (ss.get(f"nm-{run_id}-{spk}") or "").strip() for spk in speakers}
    request("process", run_dir=run_dir, start="minutes", names={k: v for k, v in names.items() if v})


def tab_downloads(run: Run) -> None:
    files = run.export_files()
    desc = {
        "meeting_record.md": "Meeting record, human-readable (Markdown)",
        "meeting_record.json": "Meeting record, machine-readable (JSON, same decisions & tasks)",
        "meeting_report.html": "Interactive report with the audio. Click any timestamp to listen",
        "action_items.csv": "Action items for a spreadsheet or tracker",
        "raw_transcript.txt": "Raw transcript (speech-to-text output)",
        "refined_transcript.txt": "Refined transcript (after terminology correction)",
        "raw_transcript.srt": "Raw transcript as subtitles",
        "refined_transcript.srt": "Refined transcript as subtitles",
        "refinement_changes.csv": "Every correction proposed, applied or blocked",
    }
    mime = {".md": "text/markdown", ".json": "application/json", ".html": "text/html", ".csv": "text/csv",
            ".txt": "text/plain", ".srt": "application/x-subrip"}
    b = busy()
    if b:
        st.caption("Downloads are paused while something is being processed.")
    st.download_button("⬇ Everything as a .zip", run.zip_bytes(), file_name=f"{run.run_id}.zip", mime="application/zip",
                       type="primary", on_click="ignore", disabled=b)
    for name in desc:
        if name in files:
            c = st.columns([3, 5])
            with c[0]:
                st.download_button(name, files[name], file_name=name, mime=mime[Path(name).suffix], on_click="ignore",
                                   key=f"dl-{name}", width="stretch", disabled=b)
            c[1].caption(desc[name])
    if run.record:
        st.markdown("#### Email draft")
        subject, body = email_draft(run.record)
        st.text_input("Subject", subject, key=f"em-s-{run.run_id}")
        st.text_area("Body", body, height=260, key=f"em-b-{run.run_id}")
        url = "mailto:?subject=" + urllib.parse.quote(subject) + "&body=" + urllib.parse.quote(body[:1800])
        st.link_button("Open in my email app", url)
        with st.expander("Preview meeting_record.md"):
            st.markdown(files["meeting_record.md"].decode("utf-8"))


def tab_details(run: Run) -> None:
    st.markdown("#### Models and their roles")
    m = run.models
    st.table(pd.DataFrame([
        {"Stage": "1 · Speech-to-text", "Model": m.get("speech_to_text"), "Role": "Audio → raw transcript with timestamps"},
        {"Stage": "2 · Refinement (LLM #1)", "Model": m.get("refinement"), "Role": "Fix misheard domain terms; guarded by code checks"},
        {"Stage": "3 · Documentation (LLM #2)", "Model": m.get("minutes"), "Role": "Summary, minutes, decisions, action items; verified by code"},
        *([{"Stage": "Speaker detection", "Model": m.get("speaker_detection"), "Role": "Who spoke when (offline)"}] if m.get("speaker_detection") else []),
    ]))
    st.markdown("#### Timing")
    tc = st.columns(len(run.timings) or 1)
    for col, (k, v) in zip(tc, run.timings.items()):
        col.metric(STAGE_LABELS[k], f"{v:.1f} s")
    if run.usage:
        st.markdown("#### Model calls")
        st.table(pd.DataFrame(run.usage))
    if run.record:
        st.caption(f"Minutes mode: {run.record.mode}")
    if run.transcript and run.transcript.whisper_prompt:
        st.markdown("#### Spelling hint sent to Whisper")
        st.code(run.transcript.whisper_prompt, language=None)
    if run.audio_stats:
        a = run.audio_stats
        st.markdown("#### Audio")
        st.caption(f"Peak level {a.get('peak_dbfs')} dBFS · average {a.get('rms_dbfs')} dBFS · sent to speech-to-text in {a.get('chunks')} part(s)")
    with st.expander("Settings used"):
        st.json(run.settings)
    st.caption(f"Run folder: {run.dir}")


# --------------------------------------------------------------------------- main

ss.setdefault("theme_name", "Midnight Teal")
for _k, _v in {"busy": False, "job": None, "pending": None, "hashes": {}, "dup": None, "flash": None,
               "mode": None, "own_ready": False}.items():
    ss.setdefault(_k, _v)
finish_job()
inject_css(theme())
_top = st.columns([10, 1.7])
with _top[1]:
    with st.popover("🎨 Theme", width="stretch"):
        st.radio("Colour theme", list(THEMES), key="theme_name")

# ---- onboarding: sign in -> free key or own keys -> (own keys) setup
_user = ob.current_user()
if _user is None:
    ob.login_page(LOGO_B64)
    st.stop()
ss.user = _user
_server_key = server_groq_key()
if ss.mode == "free" and not _server_key:
    ss.mode = None
if ss.mode is None:
    _base = Settings.from_env()  # the usage file is shared by all users, next to the runs folder
    ob.choose_page(LOGO_B64, _user, usage_store(_base), bool(_server_key))
    st.stop()
if ss.mode == "own" and not ss.own_ready:
    _base = Settings.from_env()
    ob.keys_page(_base, bool(_server_key), fmt_minutes(_base.free_daily_minutes * 60), _user)
    st.stop()

settings = build_settings()
handle_pending(settings)
run = current_run()
_j = ss.get("job")
if _j is not None and _j.kind == "process" and job_running():
    loading_page()  # processing gets its own page
elif ss.show_new or run is None:
    new_meeting_view(settings)
else:
    results_view(run, settings)
