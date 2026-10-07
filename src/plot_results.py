"""Step 4: turn results_ba_newdata.json into a summary table + figure."""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

RUN = "/data/.openclaw/workspace/aif-latex/aif_run"
FIGS = "/data/.openclaw/workspace/aif-latex/figs"
res = json.load(open(f"{RUN}/results_ba_newdata.json"))
agents = res["agents"]

print(f"Train window: {res['train_window']}  |  Test window: {res['test_window']}")
print(f"Agents: {len(agents)}  |  obs dim: {agents[0]['train']['final_agent_navs'] and ''}"
      f"{res.get('test_obs_dim', '?')}")
print()
hdr = f"{'#':>2} | {'train agent':>11} {'train B&H':>9} {'better':>6} | " \
      f"{'test agent':>10} {'test B&H':>8} {'better':>6}"
print(hdr)
print("-" * len(hdr))
for e in agents:
    t, s = e["train"], e.get("test", {})
    print(f"{e['agent']:>2} | {t['agent_avg']:>11.4f} {t['market_avg']:>9.4f} "
          f"{t['agent_better']:>6.2f} | "
          f"{s.get('agent_avg', float('nan')):>10.4f} "
          f"{s.get('market_avg', float('nan')):>8.4f} "
          f"{s.get('agent_better', float('nan')):>6.2f}")

for tag in ("train", "test"):
    if all(tag in e for e in agents):
        a = np.array([e[tag]["agent_avg"] for e in agents])
        m = np.array([e[tag]["market_avg"] for e in agents])
        b = np.array([e[tag]["agent_better"] for e in agents])
        print(f"\n[{tag}] mean agent NAV {a.mean():.4f} | mean buy&hold NAV {m.mean():.4f} "
              f"| agent better in {b.mean()*100:.0f}% of episodes")

# --- figure: final NAV per episode (agent vs. buy & hold) --------------------
# NOTE: final_*_navs are the 20 *final* NAVs (one scalar per episode), not a
# NAV time series -- so we plot them per episode, not against "trading day".
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, tag, title in ((axs[0], "train", "Trainingsfenster (in-sample)"),
                       (axs[1], "test", "Testfenster (out-of-sample)")):
    a = np.array([e[tag]["final_agent_navs"] for e in agents])   # (agents, episodes)
    m = np.array([e[tag]["final_market_navs"] for e in agents])
    a_mean, m_mean = a.mean(axis=0), m.mean(axis=0)
    x = np.arange(1, a_mean.size + 1)
    ax.scatter(x - 0.12, a_mean, s=26, color="tab:blue", label="A2C-Agent")
    ax.scatter(x + 0.12, m_mean, s=26, color="tab:orange", label="Buy & Hold")
    ax.axhline(a_mean.mean(), color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.axhline(m_mean.mean(), color="tab:orange", ls="--", lw=1, alpha=0.7)
    ax.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
    ax.set_title(f"{title}\nAgent {a_mean.mean():.3f} vs. B&H {m_mean.mean():.3f} "
                 f"(je 20 Episoden, Mittel der 5 Agenten)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("End-NAV (Start = 1,0)")
    ax.legend()
fig.suptitle("A2C auf Boeing (neue Daten) — Nachbau mit pandas_ta_classic", fontsize=13)
fig.tight_layout()
out = f"{FIGS}/update_2026_a2c.png"
fig.savefig(out, dpi=130)
print(f"\nfigure -> {out}")
