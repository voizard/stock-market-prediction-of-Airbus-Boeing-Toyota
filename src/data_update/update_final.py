"""Final: Airbus 2022 reference + comparison chart + summary table."""
import json
import re
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


# ---------- ariva: Airbus daily rows for a given month/year ----------
def ariva_rows(url):
    h = get(url)
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").replace("&euro;", "")
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        cells = [c.strip() for c in cells]
        if cells and re.match(r"\d{2}\.\d{2}\.\d{2}", cells[0]):
            rows.append(cells)
    return rows


air_2022 = []
for my in ["2022-08", "2022-09"]:
    m, y = my.split("-")
    try:
        r = ariva_rows(f"https://www.ariva.de/aktien/airbus-aktie/historische_kurse?month={m}&year={y}")
        print(f"ariva {my}: {len(r)} rows", r[0][:5] if r else "")
        air_2022 += r
    except Exception as e:  # noqa: BLE001
        print(f"ariva {my} FAIL {e}")

# pick the row closest to 31.08.2022
def close_of(rows, target="31.08.22"):
    for r in rows:
        if r[0] == target:
            return float(r[4].replace(".", "").replace(",", "."))
    return None


air_2022_close = close_of(air_2022)
print("Airbus close 31.08.2022:", air_2022_close)

# ---------- 1y daily series for the chart ----------
def sa_api(sym, rng="1Y", period="Day"):
    d = json.loads(get(f"https://stockanalysis.com/api/symbol/s/{sym}/history?range={rng}&period={period}"))
    df = pd.DataFrame(d["data"])
    df["t"] = pd.to_datetime(df["t"])
    return df.sort_values("t").set_index("t")


series = {"Boeing (BA)": sa_api("BA")["a"].astype(float),
          "Toyota (TM)": sa_api("TM")["a"].astype(float)}

air = json.load(open("/tmp/air_extract.json"))
air_pts = sorted(air["points"])
air_s = pd.Series({pd.to_datetime(t, unit="s").normalize(): c for t, c in air_pts})
series["Airbus (EPA:AIR)"] = air_s

plt.figure(figsize=(9, 5))
for name, s in series.items():
    norm = s / s.iloc[0] * 100.0
    plt.plot(norm.index, norm.values, label=name, linewidth=1.6)
plt.axhline(100, color="grey", linewidth=0.8, linestyle="--")
plt.title("Price development, last 12 months (normalised, 100 = 1 year ago)")
plt.ylabel("Index")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aif-latex/figs/update_2026_1y.png", dpi=130)
print("chart -> aif-latex/figs/update_2026_1y.png")

# ---------- summary ----------
m = json.load(open("aif-latex/updated_metrics.json"))
print("\n=== 12m ===")
for k in ["Boeing", "Airbus", "Toyota"]:
    if k == "Airbus":
        lr = float(np.log(series["Airbus (EPA:AIR)"]).diff().sum())
        print(f"Airbus  12m logret = {lr:.4f}  pct={ (series['Airbus (EPA:AIR)'].iloc[-1]/series['Airbus (EPA:AIR)'].iloc[0]-1)*100:.2f}%  price={series['Airbus (EPA:AIR)'].iloc[-1]:.2f}")
    else:
        print(f"{k:8s} 12m logret = {m[k]['logret_12m']:.4f}  pct={m[k]['pct_12m']:.2f}%  price={m[k]['price']}")

json.dump({"airbus_2022_08_31_close": air_2022_close,
           "airbus_ariva_rows": len(air_2022)},
          open("aif-latex/airbus_2022.json", "w"), indent=2)
