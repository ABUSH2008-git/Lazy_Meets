"""Measure transcript quality against a reference you typed by hand (or the sample meeting script).

It reports word error rate (WER) for the raw and the refined transcript, plus how many glossary
terms each one got right, so you can see what the refinement stage actually fixed.

Usage:
  python scripts/eval_wer.py --reference samples/demo_reference.txt --run outputs/<run folder>
  python scripts/eval_wer.py --reference ref.txt --raw raw_transcript.txt --refined refined_transcript.txt --terms "Redis, Kafka"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetscribe.guards import _NUMWORDS, _parse_word_number, numbers_in  # noqa: E402


def load_transcript(path: Path) -> str:
    """Our .txt exports have a short header and '[mm:ss] Speaker: ' prefixes; strip them."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        m = re.match(r"\[(?:\d{2}:)?\d{2}:\d{2}\]\s*(?:[^:]{1,30}:\s)?(.*)", line)
        if m:
            out.append(m.group(1))
    return " ".join(out) if out else path.read_text(encoding="utf-8")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, and write every number as digits on both sides
    ("four hundred and twenty" == "420", "twenty-fourth" == "24th" == "24")."""
    text = text.lower().replace("’", "'").replace("$", " ").replace("%", " percent ")
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    words = re.findall(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?|[a-z']+|[,.;:?!]", text)
    words = [re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", w) for w in words]
    out: list[str] = []
    run: list[str] = []
    for w in words + ["<end>"]:
        if w in _NUMWORDS or (run and w == "and" and run[-1] in {"hundred", "thousand", "million", "billion"}):
            run.append(w)
            continue
        trailing = 0
        while run and run[-1] == "and":
            run.pop()
            trailing += 1
        if run:
            v = _parse_word_number(run)
            out.append(str(int(v)) if v is not None and float(v).is_integer() else " ".join(run))
            run = []
        out.extend(["and"] * trailing)
        if w != "<end>" and w not in ",.;:?!":
            out.append(w)
    return " ".join(out)


def term_hits(text: str, terms: list[str]) -> tuple[int, int]:
    low = text.lower()
    hit = sum(1 for t in terms if re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", low))
    return hit, len(terms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="plain text of what was actually said")
    ap.add_argument("--run", help="a run folder (contains raw_transcript.txt and refined_transcript.txt)")
    ap.add_argument("--raw")
    ap.add_argument("--refined")
    ap.add_argument("--terms", default="", help="comma-separated terms to check (default: the run's glossary)")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    if ref_path.suffix == ".json":
        ref = " ".join(t["text"] for t in json.loads(ref_path.read_text())["turns"])
    else:
        ref = ref_path.read_text(encoding="utf-8")
    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    if args.run:
        run = Path(args.run)
        raw_p, ref_p = run / "raw_transcript.txt", run / "refined_transcript.txt"
        if not terms and (run / "run.json").exists():
            terms = json.loads((run / "run.json").read_text())["context"].get("glossary", [])
    else:
        raw_p, ref_p = Path(args.raw), Path(args.refined)

    reference = normalize(ref)
    print(f"Reference: {len(reference.split())} words")
    print(f"{'transcript':<10} {'WER':>7}  {'terms right':>12}  {'numbers kept':>13}")
    ref_nums = numbers_in(ref)
    for label, p in (("raw", raw_p), ("refined", ref_p)):
        if not p or not p.exists():
            continue
        hyp_text = load_transcript(p)
        wer = jiwer.wer(reference, normalize(hyp_text))
        h, n = term_hits(hyp_text, terms) if terms else (0, 0)
        nums = numbers_in(hyp_text)
        kept = sum(min(c, nums[k]) for k, c in ref_nums.items())
        total = sum(ref_nums.values())
        print(f"{label:<10} {wer:>7.1%}  {f'{h}/{n}' if n else '-':>12}  {f'{kept}/{total}':>13}")


if __name__ == "__main__":
    main()
