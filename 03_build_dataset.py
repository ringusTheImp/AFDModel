#!/usr/bin/env python3
"""
Pair weather text with AFD text into chat-format JSONL for SFT.

Now joins 1:1 on utc_valid since 02 produces one weather record per AFD.

Usage:
    python 03_build_dataset.py
"""
import argparse
import json
import re
from pathlib import Path

from wx_afd import SYSTEM_PROMPT, REQUIRED_SECTIONS


def clean_afd(raw):
    lines = raw.strip().split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(".") and any(
            k in s.upper()
            for k in ["KEY MESSAGE", "WHAT HAS CHANGED", "SYNOPSIS",
                       "SHORT TERM", "DISCUSSION", "UPDATE"]
        ):
            start = i
            break
        if s == "&&" and i > 3:
            start = i + 1
            break

    body = "\n".join(lines[start:])
    body = re.sub(r"\n\s*\$\$.*", "", body, flags=re.DOTALL)
    body = re.sub(r"\n\s*LMK WATCHES/WARNINGS/ADVISORIES.*", "", body, flags=re.DOTALL)
    return body.strip()


def quality_score(text):
    s = 0.0
    low = text.lower()
    if len(text) > 500:  s += 1
    if len(text) > 1500: s += 1
    if len(text) > 3000: s += 1
    for sec in REQUIRED_SECTIONS:
        if sec in low: s += 1
    for term in ["front", "trough", "ridge", "jet", "moisture",
                  "model", "confidence", "timing", "precip", "thunder"]:
        if term in low: s += 0.3
    if len(text) < 200: s -= 3
    if "no changes" in low and len(text) < 400: s -= 2
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afd-file", default="data/afds_lmk.jsonl")
    ap.add_argument("--weather-file", default="data/weather.jsonl")
    ap.add_argument("--min-length", type=int, default=200)
    ap.add_argument("--min-quality", type=float, default=2.0)
    ap.add_argument("--val-split", type=float, default=0.05)
    args = ap.parse_args()

    # Index weather by utc_valid (1:1 with AFDs now)
    wx = {}
    for line in open(args.weather_file):
        rec = json.loads(line)
        wx[rec["utc_valid"]] = rec["weather_text"]
    print(f"Weather records: {len(wx)}")

    examples = []
    skip = {"no_weather": 0, "too_short": 0, "low_quality": 0}

    for line in open(args.afd_file):
        afd = json.loads(line)
        utc = afd["utc_valid"]
        weather_text = wx.get(utc)
        if not weather_text:
            skip["no_weather"] += 1
            continue
        body = clean_afd(afd["text"])
        if len(body) < args.min_length:
            skip["too_short"] += 1
            continue
        if quality_score(body) < args.min_quality:
            skip["low_quality"] += 1
            continue
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": weather_text},
                {"role": "assistant", "content": body},
            ]
        })

    n = int(len(examples) * (1 - args.val_split))
    train, val = examples[:n], examples[n:]

    Path("data").mkdir(exist_ok=True)
    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        p = Path("data") / name
        with open(p, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(data)} examples → {p}")

    print(f"Skipped: {skip}")
    if examples:
        lens = [len(e["messages"][2]["content"]) for e in examples]
        wxlens = [len(e["messages"][1]["content"]) for e in examples]
        print(f"Weather input: min={min(wxlens)} avg={sum(wxlens)//len(wxlens)} max={max(wxlens)} chars")
        print(f"AFD output:    min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)} chars")


if __name__ == "__main__":
    main()
