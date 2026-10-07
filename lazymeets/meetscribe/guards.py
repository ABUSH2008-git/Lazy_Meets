"""Meaning guards for transcript refinement.

The refinement model proposes small span replacements. Before any of them is applied, code
checks that it doesn't change numbers, negation, commitments or people's names, and that the
new words actually sound like the old ones (a mishearing), not like a rewrite.
Blocked corrections are kept and shown to the user with the reason.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import jellyfish
from rapidfuzz import fuzz

# ---------------------------------------------------------------- numbers

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9, "lakh": 10**5, "lakhs": 10**5,
           "crore": 10**7, "crores": 10**7, "k": 1000}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_NUMWORDS = set(_UNITS) | set(_TENS) | {"hundred", "thousand", "million", "billion", "lakh", "lakhs", "crore", "crores"} | set(_ORDINALS)

_DIGITS = re.compile(r"(?<![\d.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k|m|bn|b|million|billion|thousand|lakh|lakhs|crore|crores)?(?![a-z\d])", re.I)
_DIGITS_ANY = re.compile(r"(?<![\d.])(\d+(?:,\d{3})*(?:\.\d+)?)")


def _parse_word_number(tokens: list[str]) -> Optional[float]:
    total, current, seen = 0.0, 0.0, False
    decimal: list[int] = []
    in_decimal = False
    for t in tokens:
        if t == "and":
            continue
        if t == "point":
            in_decimal = True
            continue
        if in_decimal:
            if t in _UNITS and _UNITS[t] < 10:
                decimal.append(_UNITS[t])
            continue
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t in _ORDINALS:
            current += _ORDINALS[t]
        elif t == "hundred":
            current = max(current, 1) * 100
        elif t in _SCALES:
            total += max(current, 1) * _SCALES[t]
            current = 0
        else:
            continue
        seen = True
    if not seen:
        return None
    value = total + current
    if decimal:
        value += float("0." + "".join(map(str, decimal)))
    return value


def numbers_in(text: str) -> Counter:
    """All numeric values in a piece of text, whether written as digits or words."""
    vals: list[float] = []
    low = text.lower().replace("’", "'")
    for m in _DIGITS.finditer(low):
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9,
                "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}.get(suf, 1)
        vals.append(v * mult)
    # digits glued to letters (p99, Q3, GPT-4o) that the first pattern skipped
    covered = {m.span(1) for m in _DIGITS.finditer(low)}
    for m in _DIGITS_ANY.finditer(low):
        if m.span(1) not in covered:
            vals.append(float(m.group(1).replace(",", "")))
    tokens = re.findall(r"[a-z]+", re.sub(r"(?<=[a-z])-(?=[a-z])", " ", low))
    run: list[str] = []
    for t in tokens + ["<end>"]:
        if t in _NUMWORDS or (run and t in {"and", "point"}):
            run.append(t)
        else:
            while run and run[-1] in {"and", "point"}:
                run.pop()
            if run:
                v = _parse_word_number(run)
                if v is not None:
                    vals.append(v)
            run = []
    return Counter(round(v, 6) for v in vals)


# ---------------------------------------------------------------- negation & commitments

_NEG = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "cannot", "without", "nope", "nah"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def negation_count(text: str) -> int:
    return sum(1 for w in _words(text) if w in _NEG or w.endswith("n't"))


_MODAL_CLASSES = {
    "will": {"will", "shall", "gonna", "won't", "wont"},
    "must": {"must", "gotta", "mustn't"},
    "should": {"should", "shouldn't", "ought"},
    "maybe": {"may", "might", "maybe", "perhaps", "possibly", "probably", "could", "couldn't"},
    "can": {"can", "can't", "cannot"},
    "agree": {"agree", "agreed", "approve", "approved", "accept", "accepted", "confirm", "confirmed", "decide",
              "decided", "reject", "rejected", "deny", "denied", "yes", "yeah", "yep"},
}


def modality(text: str) -> Counter:
    c: Counter = Counter()
    ws = _words(text)
    for i, w in enumerate(ws):
        if w.endswith("'ll"):
            c["will"] += 1
            continue
        for cls, words in _MODAL_CLASSES.items():
            if w in words:
                c[cls] += 1
        if w in {"need", "needs", "have", "has"} and i + 1 < len(ws) and ws[i + 1] == "to":
            c["must"] += 1
        if w == "going" and i + 1 < len(ws) and ws[i + 1] == "to":
            c["will"] += 1
    return c


# ---------------------------------------------------------------- sound-alike check

_LETTER_NAMES = {
    "a": "a", "ay": "a", "eh": "a", "b": "b", "bee": "b", "be": "b", "c": "c", "see": "c", "sea": "c", "d": "d",
    "dee": "d", "e": "e", "ee": "e", "f": "f", "ef": "f", "eff": "f", "g": "g", "gee": "g", "h": "h", "aitch": "h",
    "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k", "kay": "k", "l": "l", "el": "l", "ell": "l", "m": "m",
    "em": "m", "n": "n", "en": "n", "o": "o", "oh": "o", "p": "p", "pee": "p", "pea": "p", "q": "q", "cue": "q",
    "queue": "q", "r": "r", "are": "r", "ar": "r", "s": "s", "es": "s", "ess": "s", "t": "t", "tee": "t", "tea": "t",
    "u": "u", "you": "u", "v": "v", "vee": "v", "w": "w", "x": "x", "ex": "x", "y": "y", "why": "y", "z": "z",
    "zee": "z", "zed": "z",
}


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _spelled_letters(s: str) -> str:
    toks = re.findall(r"[a-z]+|\d+", s.lower().replace("double you", "w").replace("double u", "w"))
    if toks and all(t in _LETTER_NAMES or t.isdigit() for t in toks):
        return "".join(_LETTER_NAMES.get(t, t) for t in toks)
    return ""


def _phonetic(s: str) -> str:
    a = _alnum(s)
    try:
        return jellyfish.metaphone(a) if a else ""
    except Exception:  # pragma: no cover
        return a


def _changed_part(original: str, corrected: str) -> tuple[str, str]:
    """Drop the words both sides share at the start and end, keep the part that changed."""
    a = re.findall(r"[a-z0-9]+", original.lower())
    b = re.findall(r"[a-z0-9]+", corrected.lower())
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return " ".join(a[i: len(a) - j]), " ".join(b[i: len(b) - j])


def sound_similarity(original: str, corrected: str) -> float:
    """0..1, how much the changed words in `corrected` look or sound like the ones they replace."""
    o, c = _changed_part(original, corrected)
    if not o and not c:
        return 1.0  # only casing / punctuation / spacing changed
    b = _alnum(c)
    if not b or not _alnum(o):
        return 0.0
    cands = [x for x in (_alnum(o), _spelled_letters(o)) if x]
    char = max((fuzz.ratio(x, b) for x in cands), default=0) / 100
    pa, pb = _phonetic(o), _phonetic(c)
    phon = fuzz.ratio(pa, pb) / 100 if pa and pb else 0.0
    # metaphone codes are short, so only trust a phonetic match when it is a strong one
    return max(char, phon) if phon >= 0.8 else char


def _only_number_format(original: str, corrected: str) -> bool:
    """True if the only difference is how a number is written ("twenty-fourth" -> "24th")."""
    nums = numbers_in(original)
    if not nums or nums != numbers_in(corrected):
        return False
    strip = lambda t: re.sub(r"\d+(?:[.,]\d+)*(?:st|nd|rd|th|k)?", " ", " ".join(w for w in re.findall(r"[a-z0-9.,]+", t.lower()) if w not in _NUMWORDS and w not in {"and", "point"}))
    return fuzz.ratio(_alnum(strip(original)), _alnum(strip(corrected))) >= 80


# ---------------------------------------------------------------- the check


def _known(term: str, known: list[str], cutoff: int = 88) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    for k in known:
        kl = k.lower()
        if t == kl or fuzz.ratio(t, kl) >= cutoff:
            return True
        # the corrected span may wrap the term ("the Kubernetes cluster")
        if len(kl) >= 3 and re.search(r"(?<!\w)" + re.escape(kl) + r"(?!\w)", t):
            return True
    return False


def _name_tokens(names: list[str]) -> set[str]:
    out = set()
    for n in names:
        for t in re.findall(r"[A-Za-z][A-Za-z'.-]+", n):
            if len(t) >= 3:
                out.add(t.lower())
    return out


def check_correction(original: str, corrected: str, category: str, terms: list[str], participants: list[str]) -> Optional[str]:
    """Return None if the correction is safe to apply, else a short reason it was blocked."""
    o, c = original.strip(), corrected.strip()
    if not c:
        return "would delete words from the transcript"
    ow, cw = _words(o), _words(c)
    if len(o) > 90 or len(ow) > 8:
        return "edit is too long; refinement may only fix short terms, not rewrite sentences"
    if len(cw) > len(ow) + 3:
        return "adds too many words"
    if numbers_in(o) != numbers_in(c):
        a = ", ".join(_fmt(v) for v in sorted(numbers_in(o).elements())) or "none"
        b = ", ".join(_fmt(v) for v in sorted(numbers_in(c).elements())) or "none"
        return f"would change numbers ({a} → {b})"
    if negation_count(o) != negation_count(c):
        return "would change negation (not/no/never…)"
    if modality(o) != modality(c):
        return "would change commitment or certainty wording (will/might/should…)"
    names = _name_tokens(participants)
    if category == "person_name":
        if not (_known(c, participants) or _known(c, terms)):
            return "name changed without the correct spelling in the participants list or glossary"
    elif names:
        o_names = {t for t in names if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", o.lower())}
        c_names = {t for t in names if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", c.lower())}
        if o_names - c_names:
            return "would change a participant's name"
    if _only_number_format(o, c):
        return None
    trusted = _known(c, terms) or _known(c, participants)
    sim = sound_similarity(o, c)
    if sim < (0.45 if trusted else 0.6):
        return f"replacement doesn't sound like the original words (similarity {sim:.2f}), so it isn't a likely mishearing"
    return None


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)
