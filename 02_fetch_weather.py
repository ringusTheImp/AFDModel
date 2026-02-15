#!/usr/bin/env python3
"""
Fetch weather for Louisville matched to each AFD issuance time.

One API call per DATE (cached), one weather record per AFD (matched to
its exact issuance hour). Slim output: snapshot + sounding + 3-day
trajectory (12-hourly) + 7-day daily summary.

Usage:
    python 02_fetch_weather.py
    python 02_fetch_weather.py --limit 10
"""
import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

LAT, LON = 38.25, -85.76

ALL_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
KEY_LEVELS = [250, 300, 500, 700, 850, 925, 1000]

SURFACE_VARS = [
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m",
    "wind_speed_120m", "wind_direction_120m",
    "wind_gusts_10m",
    "precipitation", "snowfall", "snow_depth",
    "cape", "lifted_index", "freezing_level_height", "visibility",
]

PRESSURE_VARS = [
    "temperature", "relative_humidity",
    "wind_speed", "wind_direction", "geopotential_height",
]

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
]

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_api(start_date, end_date):
    plev = [f"{v}_{l}hPa" for v in PRESSURE_VARS for l in ALL_LEVELS]
    r = requests.get(API_URL, params={
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(SURFACE_VARS + plev),
        "daily": ",".join(DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "kn",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise ValueError(data.get("reason", str(data)))
    return data


def closest_idx(times, target_utc):
    target = datetime.fromisoformat(target_utc.replace("Z", ""))
    return min(range(len(times)),
               key=lambda i: abs((datetime.fromisoformat(times[i]) - target).total_seconds()))


def dewpoint_f(t_f, rh):
    if t_f is None or rh is None or rh <= 0:
        return None
    tc = (t_f - 32) * 5 / 9
    g = (17.27 * tc) / (237.3 + tc) + math.log(max(rh, 1) / 100)
    return round((237.3 * g) / (17.27 - g) * 9 / 5 + 32, 1)


def v(val):
    return "—" if val is None else str(val)

def ff(val, w=6):
    return f"{val:>{w}.1f}" if val is not None else f"{'—':>{w}}"

def fi(val, w=5):
    return f"{val:>{w}.0f}" if val is not None else f"{'—':>{w}}"


def serialize(hourly, daily, idx, utc_valid):
    s = {var: hourly[var][idx] if idx < len(hourly.get(var, [])) else None
         for var in hourly if var != "time"}

    fzl = s.get("freezing_level_height")
    vis = s.get("visibility")

    L = [
        f"=== WEATHER DATA VALID {utc_valid} ===",
        "=== LOUISVILLE KY (NWS LMK) ===", "",
        "── CONDITIONS ──",
        f"  T:{v(s.get('temperature_2m'))}°F  "
        f"Td:{v(s.get('dew_point_2m'))}°F  "
        f"RH:{v(s.get('relative_humidity_2m'))}%",
        f"  Wind10m:{v(s.get('wind_speed_10m'))}kt"
        f"@{v(s.get('wind_direction_10m'))}°  "
        f"Gust:{v(s.get('wind_gusts_10m'))}kt",
        f"  Wind120m:{v(s.get('wind_speed_120m'))}kt"
        f"@{v(s.get('wind_direction_120m'))}°",
        f"  MSLP:{v(s.get('pressure_msl'))}hPa  "
        f"CAPE:{v(s.get('cape'))}J/kg  "
        f"LI:{v(s.get('lifted_index'))}",
        f"  Sky:{v(s.get('cloud_cover'))}%  "
        f"Pcpn:{v(s.get('precipitation'))}in  "
        f"Snow:{v(s.get('snowfall'))}in",
        f"  FrzLvl:{f'{fzl:.0f}' if fzl is not None else '—'}m  "
        f"Vis:{f'{vis:.0f}' if vis is not None else '—'}m", "",
    ]

    # Sounding
    L.append("── SOUNDING ──")
    L.append(f"  {'mb':>6} {'T°F':>6} {'Td°F':>6} {'RH%':>4} {'kt':>5} {'dir':>4} {'GHT':>7}")
    for lv in KEY_LEVELS:
        t = s.get(f"temperature_{lv}hPa")
        rh = s.get(f"relative_humidity_{lv}hPa")
        td = dewpoint_f(t, rh)
        L.append(
            f"  {lv:>6} {ff(t)} {ff(td)} {fi(rh,4)} "
            f"{fi(s.get(f'wind_speed_{lv}hPa'))} "
            f"{fi(s.get(f'wind_direction_{lv}hPa'),4)} "
            f"{fi(s.get(f'geopotential_height_{lv}hPa'),7)}"
        )

    g5, g10 = s.get("geopotential_height_500hPa"), s.get("geopotential_height_1000hPa")
    t85, t50 = s.get("temperature_850hPa"), s.get("temperature_500hPa")
    g85 = s.get("geopotential_height_850hPa")
    derived = []
    if g5 is not None and g10 is not None:
        derived.append(f"Thickness:{g5 - g10:.0f}m")
    if all(x is not None for x in [t85, t50, g85, g5]):
        dz = (g5 - g85) / 1000
        if abs(dz) > 0.01:
            lr = -(((t50 - 32) * 5 / 9) - ((t85 - 32) * 5 / 9)) / dz
            derived.append(f"LR:{lr:.1f}C/km")
    if derived:
        L.append(f"  {' '.join(derived)}")
    L.append("")

    # 3-day trajectory, 12-hourly
    times = hourly["time"]
    traj_idxs = list(range(idx, min(idx + 72, len(times)), 12))
    if traj_idxs:
        L.append("── 3-DAY TRAJECTORY (12hr) ──")
        for i in traj_idxs:
            parts = []
            for var, lbl, u in [
                ("temperature_2m", "T", "°F"), ("dew_point_2m", "Td", "°F"),
                ("wind_speed_10m", "W", "kt"), ("wind_gusts_10m", "G", "kt"),
                ("precipitation", "Pcpn", "in"), ("snowfall", "Sn", "in"),
                ("cloud_cover", "Sky", "%"), ("cape", "CAPE", ""),
            ]:
                vals = hourly.get(var, [])
                if i < len(vals) and vals[i] is not None:
                    if lbl in ("CAPE", "Sn") and vals[i] == 0:
                        continue
                    parts.append(f"{lbl}={vals[i]}{u}")
            L.append(f"  {times[i]}: {' '.join(parts)}")
        L.append("")

    # Daily summary
    if daily and daily.get("time"):
        L.append("── DAILY ──")
        for i, d in enumerate(daily["time"]):
            parts = []
            for var, lbl, u in [
                ("temperature_2m_max", "Hi", "°F"), ("temperature_2m_min", "Lo", "°F"),
                ("precipitation_sum", "Pcpn", "in"), ("snowfall_sum", "Sn", "in"),
                ("wind_speed_10m_max", "MaxW", "kt"),
            ]:
                vals = daily.get(var, [])
                if i < len(vals) and vals[i] is not None:
                    if lbl == "Sn" and vals[i] == 0:
                        continue
                    parts.append(f"{lbl}={vals[i]}{u}")
            L.append(f"  {d}: {' '.join(parts)}")
        L.append("")

    L.append("=== END ===")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afd-file", default="data/afds_lmk.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    afds = [json.loads(line) for line in open(args.afd_file)]
    if args.limit:
        afds = afds[:args.limit]

    # Cache API responses by date (multiple AFDs same day reuse one call)
    cache = {}

    print(f"{len(afds)} AFDs to process")

    Path("data").mkdir(exist_ok=True)
    outpath = Path("data/weather.jsonl")
    ok, fail = 0, 0

    with open(outpath, "w") as f:
        for afd in tqdm(afds, desc="Fetching"):
            utc = afd["utc_valid"]
            date_str = utc[:10]
            try:
                if date_str not in cache:
                    end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
                    cache[date_str] = fetch_api(date_str, end)
                    time.sleep(0.5)

                data = cache[date_str]
                hourly = data["hourly"]
                idx = closest_idx(hourly["time"], utc)
                daily = data.get("daily", {})

                rec = {
                    "utc_valid": utc,
                    "matched_hour": hourly["time"][idx],
                    "weather_text": serialize(hourly, daily, idx, utc),
                }
                f.write(json.dumps(rec) + "\n")
                ok += 1
            except Exception as e:
                print(f"  {utc}: {e}")
                fail += 1

    print(f"\n{ok} ok, {fail} failed → {outpath}")
    print(f"API calls: {len(cache)} (cached by date)")


if __name__ == "__main__":
    main()
