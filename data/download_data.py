#!/usr/bin/env python3
"""Download the Boeing (BA) daily OHLCV series used by the 2026 re-run.

Why a download script instead of the CSV: the price data belongs to the data
provider, so it is not redistributed in this repository (see data/README.md).

Source: stockanalysis.com history API. Yahoo/yfinance is IP-blocked (HTTP 429)
from many hosts, which is why the original notebook's data loader is not usable
here either.

The output format matches the file the re-run expects:

    Date,Open,High,Low,Close,AdjClose,Volume

Usage
-----
    python3 data/download_data.py                # -> data/BA_5Y_daily.csv
    python3 data/download_data.py --force        # re-download even if present

The script is idempotent: a second run with a valid file present reports
"already present" and exits 0 without touching the file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request

API = "https://stockanalysis.com/api/symbol/s/{sym}/history?range={rng}&period=Day"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
HEADER = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]

MIN_ROWS = 1000          # 5 years of trading days
FIRST_DATE_MAX = "2021-10-31"   # the series must start no later than this


def fetch(symbol: str = "BA", rng: str = "5Y") -> list[dict]:
    url = API.format(sym=symbol, rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") or []
    if not rows:
        raise RuntimeError(f"no rows returned by {url}")
    return rows


def write_csv(rows: list[dict], path: str) -> int:
    # API returns newest-first; the project expects oldest-first.
    rows = sorted(rows, key=lambda r: r["t"])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r["t"], r["o"], r["h"], r["l"], r["c"], r["a"], r["v"]])
    os.replace(tmp, path)
    return len(rows)


def validate(path: str) -> tuple[bool, str]:
    """Check the file's *shape*, not just its existence."""
    if not os.path.exists(path):
        return False, "missing"
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return False, "empty"
    if rows[0] != HEADER:
        return False, f"unexpected header: {rows[0]}"
    body = [r for r in rows[1:] if r]
    if len(body) < MIN_ROWS:
        return False, f"only {len(body)} rows (expected >= {MIN_ROWS})"
    if body[0][0] > FIRST_DATE_MAX:
        return False, f"series starts at {body[0][0]} (expected <= {FIRST_DATE_MAX})"
    if any(len(r) != len(HEADER) for r in body):
        return False, "ragged rows"
    return True, (f"{len(body)} rows x {len(HEADER)} cols, "
                  f"{body[0][0]} .. {body[-1][0]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "BA_5Y_daily.csv"))
    ap.add_argument("--symbol", default="BA")
    ap.add_argument("--force", action="store_true", help="re-download even if valid")
    args = ap.parse_args(argv)

    ok, msg = validate(args.out)
    if ok and not args.force:
        print(f"[download_data] already present: {args.out} ({msg})")
        return 0

    print(f"[download_data] fetching {args.symbol} 5Y daily from stockanalysis.com ...")
    n = write_csv(fetch(args.symbol), args.out)
    ok, msg = validate(args.out)
    if not ok:
        print(f"[download_data] VALIDATION FAILED: {msg}", file=sys.stderr)
        return 1
    print(f"[download_data] wrote {n} rows -> {args.out}")
    print(f"[download_data] validated: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
