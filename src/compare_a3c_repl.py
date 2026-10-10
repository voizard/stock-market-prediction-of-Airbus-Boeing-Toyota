"""Replication analysis for the A3C result on AIR and TM (baseline / nstep10 / nstep10boot).

Mirrors the Boeing analysis: per arm the test and train margin, 95% CI, hit rate,
one-sided p(margin>0), worst seed and return/risk; then the paired tests over the
20 seed differences and a pooled AIR+TM test (n=40).

Usage:  python3 compare_a3c_repl.py [AIR TM | BA]
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

L = "/data/.openclaw/workspace/aif-latex/aif_run"

ARMS = [
    ("baseline",    "results_{tk}_a3c.json"),
    ("nstep10",     "results_{tk}_a3c_nstep10.json"),
    ("nstep10boot", "results_{tk}_a3c_nstep10boot.json"),
]


def load(tk, fname):
    with open(os.path.join(L, fname.format(tk=tk.lower()))) as fh:
        d = json.load(fh)
    return d, sorted(d["agents"], key=lambda a: a["seed"])


def margins(ag, window):
    return np.array([a[window]["agent_avg"] - a[window]["market_avg"] for a in ag])


def hitrate(ag, window):
    better = tot = 0
    for a in ag:
        for x, y in zip(a[window]["final_agent_navs"], a[window]["final_market_navs"]):
            tot += 1
            better += x > y
    return better / tot, tot


def ci95(x):
    n = x.size
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)


def one_sided_p(x):
    t, p2 = stats.ttest_1samp(x, 0.0)
    return (p2 / 2 if t > 0 else 1 - p2 / 2), t


def paired(a, b):
    diff = a - b
    t, p2 = stats.ttest_rel(a, b)
    return diff.mean(), t, p2, int((diff > 0).sum()), diff.size


def arm_block(tk):
    print(f"\n########## {tk.upper()} ##########")
    store = {}
    for name, fname in ARMS:
        _, ag = load(tk, fname)
        store[name] = ag
        mt = margins(ag, "test")
        mr = margins(ag, "train")
        p1, _ = one_sided_p(mt)
        hr, n = hitrate(ag, "test")
        rr_a = np.mean([a["test"]["agent_retrisk"] for a in ag])
        rr_m = np.mean([a["test"]["market_retrisk"] for a in ag])
        print(f"{name:12s} n={len(ag):2d} | test {mt.mean():+.4f} +/- {ci95(mt):.4f} "
              f"| train {mr.mean():+.4f} | hit {hr*100:5.1f}% | p>0 {p1:.4f} "
              f"| worst {mt.min():+.3f} | R/R {rr_a:+.3f} vs mkt {rr_m:+.3f}")
    print("--- paired tests (df=n-1) ---")
    for x, y in [("nstep10", "baseline"), ("nstep10boot", "baseline"),
                 ("nstep10boot", "nstep10")]:
        d, t, p, pos, n = paired(margins(store[x], "test"), margins(store[y], "test"))
        print(f"  {x:12s} - {y:12s}: {d:+.4f}  t={t:+.2f}  p={p:.4f}  ({pos}/{n} seeds +)")
    return store


def pooled(stores):
    print(f"\n########## POOLED ({' + '.join(stores)}) ##########")
    for x, y in [("nstep10", "baseline"), ("nstep10boot", "baseline"),
                 ("nstep10boot", "nstep10")]:
        diffs = []
        for tk, store in stores.items():
            diffs.append(margins(store[x], "test") - margins(store[y], "test"))
        d = np.concatenate(diffs)
        t, p2 = stats.ttest_1samp(d, 0.0)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        print(f"  {x:12s} - {y:12s}: {d.mean():+.4f}  t={t:+.2f}  p(1s)={p1:.4f}  "
              f"({int((d>0).sum())}/{d.size} seeds +)")


def main():
    tks = sys.argv[1:] or ["AIR", "TM"]
    stores = {tk: arm_block(tk) for tk in tks}
    if len(stores) > 1:
        pooled(stores)


if __name__ == "__main__":
    main()
