"""Update the AIF descriptive metrics using stockanalysis.com (Yahoo is IP-blocked).

Methodology mirrors the notebook:
  - P/E  -> trailing PE ratio
  - P/S  -> trailing PS ratio
  - log-returns -> sum of daily log returns of adjusted close over the last 12 months
"""
import json
import urllib.request

import numpy as np
import pandas as pd

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

TICKERS = [("Boeing", "BA"), ("Airbus", "AIR"), ("Toyota", "TM")]

OLD = {
    "Boeing": {"pe": None, "ps": 1.5021484, "logret": -0.3024603990632331},
    "Airbus": {"pe": 19.887064, "ps": 1.45767, "logret": -0.14509776191597012},
    "Toyota": {"pe": 10.374558, "ps": 0.006291955, "logret": -0.14860826741387712},
}

NEW_FUND = {  # from stockanalysis statistics pages (trailing)
    "Boeing": {"pe": 70.79, "ps": 1.60},
    "Airbus": {"pe": 25.04, "ps": 1.93},
    "Toyota": {"pe": 7.91, "ps": 0.68},
}


def fetch(sym, rng, period="Day"):
    url = (f"https://stockanalysis.com/api/symbol/s/{sym}/history"
           f"?range={rng}&period={period}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.loads(r.read())
    df = pd.DataFrame(d["data"])
    df["t"] = pd.to_datetime(df["t"])
    return df.sort_values("t").set_index("t")


out = {}
for name, sym in TICKERS:
    d1 = fetch(sym, "1Y", "Day")
    adj = d1["a"].astype(float)
    logret = float(np.log(adj).diff().sum())
    pct = (float(adj.iloc[-1]) / float(adj.iloc[0]) - 1.0) * 100.0

    d5 = fetch(sym, "5Y", "Weekly")
    adj5 = d5["a"].astype(float)
    since = adj5[adj5.index >= "2022-08-31"]
    since_pct = (float(since.iloc[-1]) / float(since.iloc[0]) - 1.0) * 100.0

    out[name] = {
        "symbol": sym,
        "price": float(adj.iloc[-1]),
        "date_last": str(adj.index[-1].date()),
        "pe": NEW_FUND[name]["pe"],
        "ps": NEW_FUND[name]["ps"],
        "logret_12m": logret,
        "pct_12m": pct,
        "p0_12m": float(adj.iloc[0]),
        "p1_12m": float(adj.iloc[-1]),
        "period_12m": f"{adj.index[0].date()} .. {adj.index[-1].date()}",
        "since_2022_pct": since_pct,
        "since_2022_first": float(since.iloc[0]),
        "since_2022_last": float(since.iloc[-1]),
        "since_2022_high": float(since.max()),
        "since_2022_low": float(since.min()),
        "since_2022_period": f"{since.index[0].date()} .. {since.index[-1].date()}",
        "old": OLD[name],
    }
    print(f"--- {name} ({sym}) ---")
    print(json.dumps(out[name], indent=2, default=str))

with open("aif-latex/updated_metrics.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("WROTE aif-latex/updated_metrics.json")
