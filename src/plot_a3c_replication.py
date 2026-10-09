"""Abbildung: A3C-Replikation ueber die drei Aktien (Boeing, Airbus, Toyota).

Liest results/results_{ba,air,tm}_a3c{,_nstep10,_nstep10boot}.json (20 Seeds, 100-119)
und erzeugt figures/update_2026_a3c_replication.png.

Gezeigt wird die OUT-OF-SAMPLE-Marge (Testfenster) als gepaarte Differenz gegen die
Baseline je Aktie, plus gepoolt AIR+TM (n=40). Fehlerbalken = 95-%-KI der mittleren
gepaarten Differenz; * markiert p < 0,05.
"""
import json
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

STOCKS = [("ba", "Boeing"), ("air", "Airbus"), ("tm", "Toyota")]


def margins(tk, tag, window="test"):
    suf = f"_{tag}" if tag else ""
    with open(os.path.join(RESULTS, f"results_{tk}_a3c{suf}.json")) as fh:
        d = json.load(fh)
    ag = sorted(d["agents"], key=lambda a: a["seed"])
    return np.array([a[window]["agent_avg"] - a[window]["market_avg"] for a in ag])


def diff_ci(d):
    n = d.size
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    t, p2 = stats.ttest_1samp(d, 0.0)
    return d.mean(), half, p2


labels, d10, d10b, ci10, ci10b, sig10, sig10b = [], [], [], [], [], [], []
for tk, name in STOCKS:
    base = margins(tk, "")
    a10 = margins(tk, "nstep10") - base
    a10b = margins(tk, "nstep10boot") - base
    m10, c10, p10 = diff_ci(a10)
    m10b, c10b, p10b = diff_ci(a10b)
    labels.append(name)
    d10.append(m10), ci10.append(c10), sig10.append(p10)
    d10b.append(m10b), ci10b.append(c10b), sig10b.append(p10b)
    print(f"{name:7s} nstep10-baseline {m10:+.4f} (p={p10:.3f}) | "
          f"nstep10boot-baseline {m10b:+.4f} (p={p10b:.3f})")

# gepoolt AIR+TM
pool = np.concatenate([margins(tk, "nstep10boot") - margins(tk, "") for tk in ("air", "tm")])
pm, pc, pp = diff_ci(pool)
labels.append("AIR+TM\n(gepoolt)")
d10.append(np.nan), ci10.append(np.nan), sig10.append(np.nan)
d10b.append(pm), ci10b.append(pc), sig10b.append(pp)
print(f"gepoolt nstep10boot-baseline {pm:+.4f} (p={pp:.3f}, n={pool.size})")

x = np.arange(len(labels))
w = 0.36
fig, ax = plt.subplots(figsize=(9.2, 4.6))

b1 = ax.bar(x - w / 2, d10, w, yerr=ci10, capsize=4, color="#2E7D32",
            label="nstep 10 $-$ Baseline")
b2 = ax.bar(x + w / 2, d10b, w, yerr=ci10b, capsize=4, color="#1F4E79",
            label="nstep 10 + Bootstrap $-$ Baseline")
ax.axhline(0, color="#333333", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Marge out-of-sample (Agent $-\\,$ Buy \\& Hold, NAV-Punkte)")
ax.set_title("A3C-Replikation: Out-of-Sample-Vorteil gegenueber der Baseline\n"
             "(20 Seeds je Aktie; Fehlerbalken: 95-%-KI)", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

for bars, means, cis, ps in ((b1, d10, ci10, sig10), (b2, d10b, ci10b, sig10b)):
    for rect, m, c, p in zip(bars, means, cis, ps):
        if np.isnan(m):
            continue
        star = "*" if p < 0.05 else "n.s."
        y = m + c if m >= 0 else m - c
        ax.annotate(f"{m:+.3f}\n{star}", (rect.get_x() + rect.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 6 if m >= 0 else -22),
                    ha="center", fontsize=8)

fig.tight_layout()
out = os.path.join(FIGS, "update_2026_a3c_replication.png")
fig.savefig(out, dpi=190)
print(f"Abbildung -> {out}")
