"""Update the descriptive metrics of the AIF project with current data (rate-limit safe).

Replicates the exact methodology of the notebook:
  - P/E ratio          -> Ticker.info['trailingPE']
  - P/S ratio (12m)    -> Ticker.info['priceToSalesTrailing12Months']
  - log-returns        -> sum of daily log returns of Adj Close over 12 months
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [("Boeing", "BA"), ("Airbus", "AIR.PA"), ("Toyota", "TM")]

OLD = {
    "Boeing": {"pe": None, "ps": 1.5021484, "logret": -0.3024603990632331},
    "Airbus": {"pe": 19.887064, "ps": 1.45767, "logret": -0.14509776191597012},
    "Toyota": {"pe": 10.374558, "ps": 0.006291955, "logret": -0.14860826741387712},
}


def retry(fn, tries=6, base=10, label=""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            wait = base * (i + 1)
            print(f"[retry {i+1}/{tries}] {label}: {type(e).__name__}: {e} -> sleep {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise last


def download(tick, start, end):
    def _go():
        df = yf.download(tick, start=start, end=end, auto_adjust=False,
                         progress=False, threads=False)
        if df is None or len(df) == 0:
            raise RuntimeError("empty dataframe")
        return df
    df = retry(_go, label=f"download {tick}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_info(tick):
    def _go():
        info = yf.Ticker(tick).info
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            raise RuntimeError("info empty/incomplete")
        return info
    return retry(_go, label=f"info {tick}")


out = {}
end_ts = pd.Timestamp.now("UTC").normalize()
end = end_ts.strftime("%Y-%m-%d")
start_12m = (end_ts - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

for name, tick in TICKERS:
    print(f"=== {name} ({tick}) ===", file=sys.stderr, flush=True)
    info = get_info(tick)
    time.sleep(6)

    hist = download(tick, start_12m, end)
    time.sleep(6)
    adj = hist["Adj Close"].dropna() if "Adj Close" in hist.columns else hist["Close"].dropna()
    logret = float(np.log(adj).diff().sum())
    p0, p1 = float(adj.iloc[0]), float(adj.iloc[-1])
    pct = (p1 / p0 - 1.0) * 100.0

    hist_s = download(tick, "2022-08-31", end)
    time.sleep(6)
    adj_s = hist_s["Adj Close"].dropna() if "Adj Close" in hist_s.columns else hist_s["Close"].dropna()

    out[name] = {
        "ticker": tick,
        "shortName": info.get("shortName") or info.get("longName"),
        "currency": info.get("currency"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe": info.get("trailingPE"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "logret_12m": logret,
        "pct_12m": pct,
        "p0_12m": p0, "p1_12m": p1,
        "period_12m": f"{start_12m} .. {end}",
        "since_2022_pct": (float(adj_s.iloc[-1]) / float(adj_s.iloc[0]) - 1.0) * 100.0,
        "since_2022_first": float(adj_s.iloc[0]),
        "since_2022_last": float(adj_s.iloc[-1]),
        "since_2022_high": float(adj_s.max()),
        "since_2022_low": float(adj_s.min()),
        "since_2022_period": f"2022-08-31 .. {end}",
        "n_obs_12m": int(len(adj)),
        "old": OLD[name],
    }
    print(json.dumps(out[name], indent=2, default=str), flush=True)
    with open("aif-latex/updated_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

print("DONE", file=sys.stderr, flush=True)
