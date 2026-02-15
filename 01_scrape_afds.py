#!/usr/bin/env python3
"""
Scrape Area Forecast Discussions for NWS Louisville (LMK) from IEM.

Usage:
    python 01_scrape_afds.py
"""
import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"

START_DATE = "2024-01-01"
END_DATE = "2026-01-01"

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
TZ_OFFSETS = {"EST": 5, "CST": 6, "EDT": 4, "CDT": 5}


def parse_utc(text):
    """Extract UTC timestamp from the NWS date line."""
    for line in text.strip().split("\n")[:10]:
        m = re.match(
            r"(\d{3,4})\s+(AM|PM)\s+(\w+)\s+\w+\s+(\w+)\s+(\d+)\s+(\d{4})",
            line.strip(),
        )
        if m:
            t, ampm, tz, mon, day, yr = m.groups()
            t = t.zfill(4)
            h, mn = int(t[:2]), int(t[2:])
            if ampm == "PM" and h != 12:
                h += 12
            elif ampm == "AM" and h == 12:
                h = 0
            dt = datetime(int(yr), MONTH_MAP.get(mon, 1), int(day), h, mn)
            dt += timedelta(hours=TZ_OFFSETS.get(tz, 5))
            return dt.strftime("%Y-%m-%dT%H:%MZ")
    return ""


def scrape(office, start, end):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    products = []

    chunks = []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=30), e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt

    for cs, ce in tqdm(chunks, desc=f"AFD{office}"):
        try:
            r = requests.get(
                IEM_URL,
                params={"pil": f"AFD{office}", "sdate": cs, "edate": ce,
                        "limit": 9999, "fmt": "text"},
                timeout=120,
            )
            r.raise_for_status()
            for raw in r.text.split(chr(1)):
                raw = raw.strip()
                if not raw or "Area Forecast Discussion" not in raw:
                    continue
                utc = parse_utc(raw)
                if utc:
                    products.append({"utc_valid": utc, "text": raw})
            time.sleep(1)
        except Exception as ex:
            print(f"  {cs}→{ce} failed: {ex}")
            time.sleep(5)

    seen = set()
    out = []
    for p in products:
        if p["utc_valid"] not in seen:
            seen.add(p["utc_valid"])
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--office", default="LMK")
    args = ap.parse_args()

    afds = scrape(args.office, START_DATE, END_DATE)

    Path("data").mkdir(exist_ok=True)
    outpath = Path(f"data/afds_{args.office.lower()}.jsonl")
    with open(outpath, "w") as f:
        for a in afds:
            f.write(json.dumps(a) + "\n")

    print(f"{len(afds)} AFDs → {outpath}")
    if afds:
        lens = [len(a["text"]) for a in afds]
        print(f"Lengths: min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")


if __name__ == "__main__":
    main()
