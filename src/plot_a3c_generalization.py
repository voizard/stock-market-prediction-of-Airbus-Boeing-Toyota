"""Abbildung: A3C-Generalisierung ueber neun Aktien.

Liest results/results_{ba,air,tm,lmt,cat,jpm,ko,nvda,xom}_a3c{,_nstep10,_nstep10boot}.json
(20 Seeds, 100-119) und erzeugt figures/update_2026_a3c_generalization.png.

Gezeigt wird die OUT-OF-SAMPLE-Marge (Testfenster) der gepaarten Differenz
`nstep10boot - baseline` je Aktie sowie gepoolt (6 neue Titel, n=120; alle 9, n=180).
Fehlerbalken = 95-%-KI der mittleren gepaarten Differenz; * markiert p < 0,05.

Die drei Originaltitel (Boeing, Airbus, Toyota) stammen aus dem Replikationslauf,
die sechs neuen (LMT, CAT, JPM, KO, NVDA, XOM) aus dem Generalisierungslauf.
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

ORIGINAL = [("ba", "Boeing"), ("air", "Airbus"), ("tm", "Toyota")]
NEW = [("lmt", "LMT"), ("cat", "CAT"), ("jpm", "JPM"),
       ("ko", "KO"), ("nvda", "NVDA"), ("xom", "XOM")]


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


labels, means, cis, ps, colors = [], [], [], [], []

for tk, name in ORIGINAL + NEW:
    d = margins(tk, "nstep10boot") - margins(tk, "")
    m, c, p = diff_ci(d)
    labels.append(name)
    means.append(m), cis.append(c), ps.append(p)
    colors.append("#8C8C8C" if (tk, name) in ORIGINAL else "#1F4E79")
    print(f"{name:7s} nstep10boot-baseline {m:+.4f} (p={p:.4f}, n={d.size})")

# gepoolt: 6 neue Titel, dann alle 9
for group, name, col in ((NEW, "6 neu\n(gepoolt)", "#C55A11"),
                         (ORIGINAL + NEW, "alle 9\n(gepoolt)", "#375623")):
    pool = np.concatenate([margins(tk, "nstep10boot") - margins(tk, "") for tk, _ in group])
    m, c, p = diff_ci(pool)
    labels.append(name)
    means.append(m), cis.append(c), ps.append(p)
    colors.append(col)
    print(f"gepoolt {name!r} {m:+.4f} (p={p:.4f}, n={pool.size})")

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10.2, 4.8))
bars = ax.bar(x, means, 0.62, yerr=cis, capsize=4, color=colors)

ax.axhline(0, color="#333333", lw=1.0)
ax.axvline(2.5, color="#999999", lw=0.8, ls="--")
ax.axvline(8.5, color="#999999", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Marge out-of-sample (Agent $-\\,$ Buy \\& Hold, NAV-Punkte)")
ax.set_title("A3C-Generalisierung: Out-of-Sample-Vorteil der Bootstrap-Variante\n"
             "gegenueber der Baseline, je Aktie und gepoolt (20 Seeds je Aktie;\n"
             "Fehlerbalken: 95-%-KI)", fontsize=11)
ax.text(1.0, ax.get_ylim()[1] * 0.92, "Original (3)", ha="center", fontsize=9,
        color="#555555")
ax.text(5.5, ax.get_ylim()[1] * 0.92, "neu (6)", ha="center", fontsize=9, color="#555555")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

for rect, m, c, p in zip(bars, means, cis, ps):
    star = "*" if p < 0.05 else "n.s."
    y = m + c if m >= 0 else m - c
    ax.annotate(f"{m:+.3f}\n{star}", (rect.get_x() + rect.get_width() / 2, y),
                textcoords="offset points", xytext=(0, 6 if m >= 0 else -24),
                ha="center", fontsize=8)

fig.tight_layout()
out = os.path.join(FIGS, "update_2026_a3c_generalization.png")
fig.savefig(out, dpi=190)
print(f"Abbildung -> {out}")
