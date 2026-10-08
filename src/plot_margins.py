"""Abbildung: Trainings- vs. Testmarge je Aktie (Agent minus Buy & Hold).

Liest results/results_{ba,air,tm}_newdata.json (20 Seeds je Aktie, Seeds 100-119)
und erzeugt figures/update_2026_margins.png.

Linkes Panel: Marge (Agent - Buy & Hold) im Trainings- und im Testfenster,
Fehlerbalken = 95%-Konfidenzintervall der mittleren gepaarten Differenz.
Rechtes Panel: Anteil der In-Sample-Marge, der out-of-sample uebrig bleibt.
"""
import json
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

TICKERS = [("ba", "Boeing"), ("air", "Airbus"), ("tm", "Toyota")]
WINDOWS = [("train", "Trainingsfenster (in-sample)"), ("test", "Testfenster (out-of-sample)")]

# zweiseitiger t-Wert fuer 95 % (df = 19 Seeds)
T_CRIT_19 = 2.093


def load(ticker):
    with open(os.path.join(RESULTS, f"results_{ticker}_newdata.json")) as fh:
        return json.load(fh)


def margins(data, window):
    """Gepaarte Differenzen Agent - Buy & Hold je Seed."""
    d = np.array([a[window]["agent_avg"] - a[window]["market_avg"]
                  for a in data["agents"] if window in a])
    m = d.mean()
    ci = T_CRIT_19 * d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else float("nan")
    return m, ci, d.size


def stars(mean, ci):
    if mean - ci <= 0.0 <= mean + ci:
        return "n.s."
    return "***" if abs(mean) > 0 else "*"


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={"width_ratios": [1.45, 1]})

labels = [name for _, name in TICKERS]
x = np.arange(len(TICKERS))
width = 0.36

train_m, train_ci, test_m, test_ci = [], [], [], []
for ticker, _ in TICKERS:
    data = load(ticker)
    a, b, n = margins(data, "train")
    c, d, _ = margins(data, "test")
    train_m.append(a), train_ci.append(b)
    test_m.append(c), test_ci.append(d)
    print(f"{ticker:4s} n={n:2d}  Marge Train {a:+.3f} +/- {b:.3f}   "
          f"Marge Test {c:+.3f} +/- {d:.3f}   {stars(c, d)}")

train_m, test_m = np.array(train_m), np.array(test_m)
train_ci, test_ci = np.array(train_ci), np.array(test_ci)

b1 = ax1.bar(x - width / 2, train_m, width, yerr=train_ci, capsize=4,
             color="tab:blue", label="In-Sample-Marge (Training)")
b2 = ax1.bar(x + width / 2, test_m, width, yerr=test_ci, capsize=4,
             color="tab:orange", label="Out-of-Sample-Marge (Test)")
ax1.axhline(0, color="black", lw=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Marge Agent $-\\,$ Buy & Hold  (NAV-Punkte, Start $=1{,}0$)")
ax1.set_title("Marge im Trainings- und Testfenster\n(20 Seeds je Aktie, "
              "Fehlerbalken: 95-%-KI)")
ax1.legend(loc="upper right")
ax1.grid(axis="y", alpha=0.25)

for bars, vals, cis in ((b1, train_m, train_ci), (b2, test_m, test_ci)):
    for rect, v, ci in zip(bars, vals, cis):
        txt = f"{v:+.2f}"
        if bars is b2:
            txt += f"\n{stars(v, ci)}"
            # ueber das Konfidenzintervall setzen, damit nichts ueberlappt
            y = v + ci if v >= 0 else v - ci
        else:
            y = v + ci
        ax1.annotate(txt, (rect.get_x() + rect.get_width() / 2, y),
                     textcoords="offset points",
                     xytext=(0, 7 if v >= 0 else -20), ha="center", fontsize=9)

share = np.array([t / i * 100 if i != 0 else float("nan")
                  for t, i in zip(test_m, train_m)])
colors = ["tab:green" if s >= 50 else "tab:orange" if s > 0 else "tab:red" for s in share]
ax2.bar(x, share, 0.55, color=colors)
ax2.axhline(0, color="black", lw=0.9)
ax2.axhline(100, color="grey", ls="--", lw=1, alpha=0.7)
ax2.text(0.02, 0.93, "gestrichelt: 100 % der In-Sample-Marge erhalten",
         transform=ax2.transAxes, ha="left", va="top", fontsize=8, color="grey")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Anteil der In-Sample-Marge, der im Test bleibt  [%]")
ax2.set_title("Wie viel vom Trainingsvorteil überlebt?\n(Test-Marge / Trainings-Marge)")
ax2.grid(axis="y", alpha=0.25)
for rect, s in zip(ax2.patches, share):
    ax2.annotate(f"{s:+.1f} %", (rect.get_x() + rect.get_width() / 2, s),
                 textcoords="offset points", xytext=(0, 5 if s >= 0 else -16),
                 ha="center", fontsize=9)

fig.suptitle("A2C auf Transportaktien (2026): In-Sample- vs. Out-of-Sample-Vorteil "
             "gegenüber Buy & Hold", fontsize=13)
fig.tight_layout()
out = os.path.join(FIGS, "update_2026_margins.png")
fig.savefig(out, dpi=140)
print(f"\nAbbildung -> {out}")
