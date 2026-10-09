"""Abbildung: A3C Ein-Faktor-Zerlegung und Block-Bootstrap (Boeing, 20 Seeds).

Liest results/results_ba_a3c*.json (Seeds 100-119) und erzeugt
figures/update_2026_a3c.png.

Zwei Panels, beide in NAV-Punkten (Start = 1,0):
  links  : Marge out-of-sample (Agent - Buy & Hold) mit 95-%-KI ueber die 20 Seeds
  rechts : Marge in-sample auf dem echten Trainingsfenster

Der Punkt der Abbildung ist der Kontrast: Die Arme, die auf dem echten Fenster
trainieren, tragen eine riesige In-Sample-Marge, die out-of-sample fast
verschwindet; die beiden Bootstrap-Arme (die den echten Pfad nie sehen) haben
in-sample ~0 und trotzdem die beste Testmarge.
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

# (Label, Datei) -- die Arme in der Reihenfolge der Berichtstabelle
ARMS = [
    ("Baseline", "results_ba_a3c.json"),
    ("ent_coef 0.02", "results_ba_a3c_ent02.json"),
    ("obs_noise 0.01", "results_ba_a3c_noise01.json"),
    ("nstep 10", "results_ba_a3c_nstep10.json"),
    ("Buendel (3 Faktoren)", "results_ba_a3c_tuned.json"),
    ("+ Bootstrap", "results_ba_a3c_boot5t.json"),
    ("nstep 10 + Bootstrap", "results_ba_a3c_nstep10boot.json"),
]

# zweiseitiger t-Wert fuer 95 % (df = 19 Seeds)
T_CRIT_19 = 2.093


def margins(path, window):
    with open(os.path.join(RESULTS, path)) as fh:
        d = json.load(fh)
    ag = sorted(d["agents"], key=lambda e: e["seed"])
    return np.array([a[window]["agent_avg"] - a[window]["market_avg"] for a in ag])


def ci95(x):
    n = x.size
    return T_CRIT_19 * x.std(ddof=1) / math.sqrt(n)


labels = [a[0] for a in ARMS]
test = [margins(a[1], "test") for a in ARMS]
train = [margins(a[1], "train") for a in ARMS]
t_mean = np.array([x.mean() for x in test])
t_err = np.array([ci95(x) for x in test])
r_mean = np.array([x.mean() for x in train])
r_err = np.array([ci95(x) for x in train])

y = np.arange(len(ARMS))[::-1]
colors = ["#c9ced6", "#c9ced6", "#c9ced6", "#2E7D32", "#2E7D32", "#1F4E79", "#1F4E79"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax1.barh(y, t_mean, xerr=t_err, color=colors, height=0.62,
         error_kw=dict(ecolor="#333333", lw=1.1, capsize=3))
ax1.axvline(0.0, color="#333333", lw=1.0, ls="--")
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel("Marge out-of-sample (NAV-Punkte)")
ax1.set_title("Testfenster: Agent $-$ Buy \\& Hold", fontsize=10, loc="left")
for yy, m, e in zip(y, t_mean, t_err):
    ax1.text(max(m, 0) + e + 0.012, yy, f"{m:+.3f}", va="center", fontsize=8)
ax1.set_xlim(-0.20, 0.46)

ax2.barh(y, r_mean, xerr=r_err, color=colors, height=0.62,
         error_kw=dict(ecolor="#333333", lw=1.1, capsize=3))
ax2.axvline(0.0, color="#333333", lw=1.0, ls="--")
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Marge in-sample (NAV-Punkte)")
ax2.set_title("Trainingsfenster: Memorieren", fontsize=10, loc="left")
for yy, m, e in zip(y, r_mean, r_err):
    ax2.text(m + e + 0.15, yy, f"{m:+.2f}", va="center", fontsize=8)
ax2.set_xlim(-0.6, 12.2)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.7)
    ax.set_axisbelow(True)

fig.suptitle("A3C auf Boeing: Ein-Faktor-Zerlegung und Block-Bootstrap "
             "(20 Seeds, Seeds 100--119)", fontsize=10.5, x=0.012, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))

os.makedirs(FIGS, exist_ok=True)
out = os.path.join(FIGS, "update_2026_a3c.png")
fig.savefig(out, dpi=190)
print(f"Abbildung -> {out}")
