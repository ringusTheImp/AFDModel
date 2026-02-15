#!/usr/bin/env python3
"""
Validate the wx-afd dataset for correctness.

Checks:
  1. File integrity (valid JSONL, expected fields)
  2. Weather text structure (all sections present, no Nones leaking)
  3. AFD text quality (sections, length, content)
  4. Time alignment (weather valid time matches AFD issuance)
  5. Sounding sanity (physical bounds on T, GHT, winds)
  6. Dataset stats summary

Usage:
    python test_dataset.py
    python test_dataset.py --weather-file data/weather.jsonl --afd-file data/afds_lmk.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime


class Colors:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def ok(msg):
    print(f"  {Colors.OK}✓{Colors.END} {msg}")

def warn(msg):
    print(f"  {Colors.WARN}⚠{Colors.END} {msg}")

def fail(msg):
    print(f"  {Colors.FAIL}✗{Colors.END} {msg}")


def load_jsonl(path):
    records = []
    bad_lines = 0
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines += 1
    return records, bad_lines


def test_files_exist(args):
    print(f"\n{Colors.BOLD}1. File existence{Colors.END}")
    all_ok = True
    for f in [args.afd_file, args.weather_file, args.train_file, args.val_file]:
        if Path(f).exists():
            ok(f"{f} ({Path(f).stat().st_size / 1024:.0f} KB)")
        else:
            fail(f"{f} missing")
            all_ok = False
    return all_ok


def test_weather_records(args):
    print(f"\n{Colors.BOLD}2. Weather records{Colors.END}")
    records, bad = load_jsonl(args.weather_file)
    if bad:
        fail(f"{bad} malformed JSON lines")
    else:
        ok(f"{len(records)} valid records, 0 bad lines")

    errors = 0
    for i, r in enumerate(records):
        # Required fields
        for field in ["utc_valid", "matched_hour", "weather_text"]:
            if field not in r:
                fail(f"Record {i}: missing '{field}'")
                errors += 1

        text = r.get("weather_text", "")

        # Check sections present
        for section in ["CONDITIONS", "SOUNDING", "TRAJECTORY", "DAILY"]:
            if section not in text:
                fail(f"Record {i} ({r.get('utc_valid', '?')}): missing {section} section")
                errors += 1

        # Check for literal "None" leaking into text
        if "None" in text:
            fail(f"Record {i} ({r.get('utc_valid', '?')}): literal 'None' in weather text")
            errors += 1

        # Check sounding has actual numbers, not all dashes
        sounding_lines = [l for l in text.split("\n") if re.match(r"\s+\d{3,4}\s", l)]
        all_dash = sum(1 for l in sounding_lines if l.count("—") >= 5)
        if sounding_lines and all_dash == len(sounding_lines):
            fail(f"Record {i} ({r.get('utc_valid', '?')}): sounding is all dashes (no data)")
            errors += 1

    if errors == 0:
        ok("All weather records structurally valid")
    else:
        fail(f"{errors} structural errors")

    return records, errors == 0


def test_sounding_physics(records):
    print(f"\n{Colors.BOLD}3. Sounding physical sanity{Colors.END}")
    errors = 0
    checked = 0

    for r in records:
        text = r.get("weather_text", "")
        lines = text.split("\n")

        for line in lines:
            m = re.match(
                r"\s+(\d{3,4})\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                line,
            )
            if not m:
                continue
            checked += 1
            lv, t, td, rh, ws, wd, ght = m.groups()
            lv, t, td, rh = int(lv), float(t), float(td), int(rh)
            ws, wd, ght = int(ws), int(wd), int(ght)

            # Temperature bounds (°F)
            if not (-120 < t < 120):
                fail(f"{r['utc_valid']} {lv}mb: T={t}°F out of range")
                errors += 1
            # Dewpoint <= temperature
            if td > t + 1:
                warn(f"{r['utc_valid']} {lv}mb: Td={td} > T={t}")
            # RH 0-100
            if not (0 <= rh <= 100):
                fail(f"{r['utc_valid']} {lv}mb: RH={rh}% out of range")
                errors += 1
            # Wind speed
            if not (0 <= ws < 300):
                fail(f"{r['utc_valid']} {lv}mb: wind={ws}kt out of range")
                errors += 1
            # Wind direction
            if not (0 <= wd <= 360):
                fail(f"{r['utc_valid']} {lv}mb: dir={wd}° out of range")
                errors += 1
            # GHT should decrease with pressure (higher pressure = lower altitude)
            # 250mb ~10km, 1000mb ~0-200m
            if lv == 250 and not (8000 < ght < 40000):
                fail(f"{r['utc_valid']} 250mb: GHT={ght}m out of range")
                errors += 1
            if lv == 1000 and not (-200 < ght < 1000):
                fail(f"{r['utc_valid']} 1000mb: GHT={ght}m out of range")
                errors += 1

    if errors == 0:
        ok(f"All {checked} sounding values within physical bounds")
    else:
        fail(f"{errors} physics violations in {checked} values")

    return errors == 0


def test_time_alignment(wx_records, args):
    print(f"\n{Colors.BOLD}4. Time alignment{Colors.END}")
    afds, _ = load_jsonl(args.afd_file)
    afd_times = {a["utc_valid"] for a in afds}
    wx_times = {r["utc_valid"] for r in wx_records}

    matched = afd_times & wx_times
    missing = afd_times - wx_times

    ok(f"{len(matched)}/{len(afd_times)} AFDs have matching weather")
    if missing:
        warn(f"{len(missing)} AFDs missing weather: {sorted(missing)[:5]}...")

    # Check hour offset between utc_valid and matched_hour
    big_gaps = 0
    for r in wx_records:
        utc = r["utc_valid"].replace("Z", "")
        matched = r["matched_hour"]
        try:
            dt1 = datetime.fromisoformat(utc)
            dt2 = datetime.fromisoformat(matched)
            gap_min = abs((dt1 - dt2).total_seconds()) / 60
            if gap_min > 60:
                warn(f"{r['utc_valid']}: matched hour {matched} is {gap_min:.0f}min off")
                big_gaps += 1
        except Exception:
            pass

    if big_gaps == 0:
        ok("All weather times within 60min of AFD issuance")
    else:
        warn(f"{big_gaps} records with >60min gap")


def test_training_data(args):
    print(f"\n{Colors.BOLD}5. Training JSONL{Colors.END}")
    for name, path in [("train", args.train_file), ("val", args.val_file)]:
        if not Path(path).exists():
            fail(f"{path} missing")
            continue

        records, bad = load_jsonl(path)
        if bad:
            fail(f"{name}: {bad} malformed lines")

        errors = 0
        for i, r in enumerate(records):
            msgs = r.get("messages", [])
            if len(msgs) != 3:
                fail(f"{name}[{i}]: expected 3 messages, got {len(msgs)}")
                errors += 1
                continue

            roles = [m["role"] for m in msgs]
            if roles != ["system", "user", "assistant"]:
                fail(f"{name}[{i}]: wrong roles {roles}")
                errors += 1

            sys_len = len(msgs[0]["content"])
            wx_len = len(msgs[1]["content"])
            afd_len = len(msgs[2]["content"])

            if wx_len < 500:
                warn(f"{name}[{i}]: weather input only {wx_len} chars")
            if afd_len < 200:
                warn(f"{name}[{i}]: AFD output only {afd_len} chars")
            if "WEATHER DATA VALID" not in msgs[1]["content"]:
                fail(f"{name}[{i}]: weather input missing header")
                errors += 1
            if "None" in msgs[1]["content"]:
                fail(f"{name}[{i}]: literal 'None' in weather input")
                errors += 1

        if errors == 0:
            ok(f"{name}.jsonl: {len(records)} examples, all valid")
        else:
            fail(f"{name}.jsonl: {errors} errors in {len(records)} examples")

    return records


def test_stats(args):
    print(f"\n{Colors.BOLD}6. Dataset stats{Colors.END}")
    train, _ = load_jsonl(args.train_file)
    val, _ = load_jsonl(args.val_file)
    all_ex = train + val

    if not all_ex:
        fail("No examples to analyze")
        return

    wx_lens = [len(e["messages"][1]["content"]) for e in all_ex]
    afd_lens = [len(e["messages"][2]["content"]) for e in all_ex]
    total_lens = [w + a for w, a in zip(wx_lens, afd_lens)]

    print(f"  Total examples:  {len(all_ex)} (train={len(train)}, val={len(val)})")
    print(f"  Weather input:   {min(wx_lens):,}–{max(wx_lens):,} chars "
          f"(avg {sum(wx_lens)//len(wx_lens):,}, ~{sum(wx_lens)//len(wx_lens)//4} tokens)")
    print(f"  AFD output:      {min(afd_lens):,}–{max(afd_lens):,} chars "
          f"(avg {sum(afd_lens)//len(afd_lens):,}, ~{sum(afd_lens)//len(afd_lens)//4} tokens)")
    print(f"  Total per example: avg ~{sum(total_lens)//len(total_lens)//4} tokens")

    # Check for duplicates
    wx_texts = [e["messages"][1]["content"] for e in all_ex]
    unique_wx = len(set(wx_texts))
    if unique_wx < len(wx_texts):
        warn(f"{len(wx_texts) - unique_wx} duplicate weather inputs "
             f"(expected if multiple AFDs same hour)")
    else:
        ok(f"All {unique_wx} weather inputs unique")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afd-file", default="data/afds_lmk.jsonl")
    ap.add_argument("--weather-file", default="data/weather.jsonl")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--val-file", default="data/val.jsonl")
    args = ap.parse_args()

    print(f"{Colors.BOLD}═══ WX-AFD Dataset Validation ═══{Colors.END}")

    if not test_files_exist(args):
        print("\nMissing files — run the pipeline first.")
        sys.exit(1)

    wx_records, wx_ok = test_weather_records(args)
    phys_ok = test_sounding_physics(wx_records)
    test_time_alignment(wx_records, args)
    test_training_data(args)
    test_stats(args)

    print(f"\n{Colors.BOLD}═══ Done ═══{Colors.END}")


if __name__ == "__main__":
    main()
