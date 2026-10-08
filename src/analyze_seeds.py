"""Auswertung der Seed-Laeufe: Mittel, Streuung, Konfidenzintervall und
gepaarter t-Test Agent vs. Buy & Hold.

Aufruf:
    python3 analyze_seeds.py TM [BA AIR ...]

Liest results_<ticker>_newdata.json und druckt je Fenster (train/test):
  - Agent-Mittel ueber die Seeds, Streuung, 95%-KI
  - Markt-Mittel
  - Differenz Agent - Markt je Seed, gepaarter t-Test (H0: Differenz = 0)
  - Episoden-Ebene: 20 Seeds x 20 Episoden, Trefferquote, gepaarter t-Test
"""
import json
import math
import os
import sys


def t_crit(df, conf=0.95):
    """Zweiseitiger t-Wert (naeherungsweise, ohne scipy)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
             14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
             20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
             30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984}
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    ks = sorted(table)
    for k in ks:
        if df <= k:
            return table[k]
    return 1.96


def mean(x):
    return sum(x) / len(x)


def stdev(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def paired_t(a, b):
    """Gepaarter t-Test fuer gleich lange Listen a, b."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    md, sd = mean(d), stdev(d)
    if sd == 0:
        return md, float("nan"), 0.0
    t = md / (sd / math.sqrt(n))
    # zweiseitiger p-Wert ueber Normalapproximation (df gross genug ab n=20)
    p = math.erfc(abs(t) / math.sqrt(2))
    return md, t, p


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, float("nan"), float("nan")
    t = (ma - mb) / se
    return ma - mb, t, math.erfc(abs(t) / math.sqrt(2))


def report(ticker):
    path = f"/data/.openclaw/workspace/aif-latex/aif_run/results_{ticker.lower()}_newdata.json"
    if not os.path.exists(path):
        print(f"[{ticker}] keine Ergebnisdatei ({path})")
        return
    d = json.load(open(path))
    agents = d["agents"]
    print(f"\n{'='*74}\n{ticker}   ({len(agents)} Seeds, obs={d.get('test_obs_dim')}, "
          f"Seeds {d.get('seed_base')}..{d.get('seed_base',0)+len(agents)-1})\n{'='*74}")
    print(f"Train {d['train_window'][0]} .. {d['train_window'][1]}   |   "
          f"Test {d['test_window'][0]} .. {d['test_window'][1]}")

    for win in ("train", "test"):
        have = [a for a in agents if win in a and "final_agent_navs" in a[win]]
        if not have:
            print(f"\n-- {win}: noch keine Daten --")
            continue
        ag = [a[win]["agent_avg"] for a in have]
        mk = [a[win]["market_avg"] for a in have]
        ar = [a[win]["agent_retrisk"] for a in have]
        mr = [a[win]["market_retrisk"] for a in have]
        n = len(ag)

        print(f"\n-- {win.upper()} ({n} Seeds) --")
        print(f"  Agent  je Seed : {[round(v,3) for v in ag]}")
        print(f"  Markt  je Seed : {[round(v,3) for v in mk]}")
        m, s = mean(ag), stdev(ag)
        ci = t_crit(n - 1) * s / math.sqrt(n) if n > 1 else float("nan")
        print(f"  Agent : Mittel {m:.3f}  Streuung {s:.3f}  95%-KI [{m-ci:.3f}, {m+ci:.3f}]")
        print(f"  Markt : Mittel {mean(mk):.3f}  Streuung {stdev(mk):.3f}")

        md, t, p = paired_t(ag, mk)
        verdict = ("Agent signifikant besser" if md > 0 and p < 0.05 else
                   "Agent signifikant schlechter" if md < 0 and p < 0.05 else
                   "kein signifikanter Unterschied")
        print(f"  Differenz Agent-Markt (gepaart, Seed-Ebene): {md:+.3f}  t={t:.2f}  p={p:.4f}  -> {verdict}")

        # Episoden-Ebene: alle Seeds x Episoden
        fa = [v for a in have for v in a[win]["final_agent_navs"]]
        fm = [v for a in have for v in a[win]["final_market_navs"]]
        wins = sum(1 for x, y in zip(fa, fm) if x > y)
        md2, t2, p2 = paired_t(fa, fm)
        print(f"  Episoden (n={len(fa)}): Agent {mean(fa):.3f} vs Markt {mean(fm):.3f}  "
              f"Trefferquote {wins/len(fa)*100:.1f}%  t={t2:.2f}  p={p2:.4f}")

        md3, t3, p3 = welch_t(ar, mr)
        print(f"  Rendite/Risiko: Agent {mean(ar):.3f} vs Markt {mean(mr):.3f}  "
              f"Diff {md3:+.3f}  t={t3:.2f}  p={p3:.4f}")


if __name__ == "__main__":
    for tk in (sys.argv[1:] or ["TM", "BA", "AIR"]):
        report(tk)
