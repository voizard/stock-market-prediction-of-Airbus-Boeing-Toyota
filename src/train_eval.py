"""Step 3: train 5 A2C agents on BA (new data) with Chris's protocol and
evaluate them with his ai_trade_performance on train + test window.

Protocol copied from the notebook:
    A2C('MlpPolicy', env, verbose=0); learn(total_timesteps=75000)
    ai_trade_performance(env, model, num_plays=20)
    train/test split: test env re-uses use_variables + scaler of the train env
"""
import json
import gc
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import resource

import numpy as np
import gymnasium as gym

sys.path.insert(0, "/data/.openclaw/workspace/aif-latex/aif_run")
import env_runner as R  # noqa: E402  (patches pandas_ta/yfinance, defines make_env)

sys.path.insert(0, "/data/.openclaw/workspace/aif-latex/course")
from aif_analysis import ai_trade_performance, analyze_actions_taken  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from stable_baselines3 import A2C  # noqa: E402

OUT = "/data/.openclaw/workspace/aif-latex/aif_run"
TRAIN = ("2021-10-06", "2024-05-31")
TEST = ("2024-06-01", "2026-10-06")
N_AGENTS = 5
TIMESTEPS = 75000
PLAYS = 20
EPISODE_SEEDS = list(range(20))  # identical episodes for agent and buy & hold


class Adapter(gym.Env):
    """Thin gymnasium wrapper so SB3 2.x can train on Chris's old-API env."""

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Discrete(int(env.action_space.n))
        lo = np.asarray(env.observation_space.low, dtype=np.float32)
        hi = np.asarray(env.observation_space.high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.asarray(self.env.reset(seed=seed), dtype=np.float32), {}

    def step(self, action):
        o, r, d, info = self.env.step(int(action))
        return np.asarray(o, dtype=np.float32), float(r), bool(d), False, info


def log(msg):
    with open("/proc/self/status") as f:
        vm = [l for l in f if l.startswith("VmRSS:")][0].split()[1]
    print(f"{msg} [RSS {int(vm)/1024:.0f} MB]", flush=True)


def dump(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    log(f"[step3] train window {TRAIN} | test window {TEST}")
    t0 = time.time()
    env_train = R.make_env(start=TRAIN[0], end=TRAIN[1])
    log(f"[step3] train env rows={len(env_train.data_source.df)} "
        f"obs={env_train.observation_space.shape} ({time.time()-t0:.1f}s)")

    results = {"train_window": TRAIN, "test_window": TEST, "agents": [],
               "note": "train env window: " + str(TRAIN)}
    out_json = f"{OUT}/results_ba_newdata.json"

    for i in range(N_AGENTS):
        ts = time.time()
        adapter = Adapter(env_train)
        model = A2C("MlpPolicy", adapter, verbose=0, seed=100 + i)
        model.learn(total_timesteps=TIMESTEPS)
        log(f"[step3] agent {i+1}/{N_AGENTS} trained in {time.time()-ts:.1f}s")
        model.save(f"{OUT}/agent_ba_{i+1}.zip")

        entry = {"agent": i + 1, "seed": 100 + i}
        res, f_navs, m_navs = ai_trade_performance(
            env_train, model, seeds=EPISODE_SEEDS, num_plays=PLAYS, plot_results=False)
        entry["train"] = {
            "agent_avg": float(res.loc["return", "agent_avg"]),
            "market_avg": float(res.loc["return", "market_avg"]),
            "agent_better": float(res.loc["return", "agent_better"]),
            "agent_retrisk": float(res.loc["return_risk", "agent_avg"]),
            "market_retrisk": float(res.loc["return_risk", "market_avg"]),
            "agent_retrisk_better": float(res.loc["return_risk", "agent_better"]),
            "final_agent_navs": [float(x) for x in f_navs],
            "final_market_navs": [float(x) for x in m_navs],
        }
        log(f"[step3]   train: agent {entry['train']['agent_avg']:.4f} vs "
            f"buy&hold {entry['train']['market_avg']:.4f} | agent_better "
            f"{entry['train']['agent_better']:.2f}")
        entry["actions_train"] = {k: int(v) for k, v in
                                  analyze_actions_taken(env_train, model, plot_actions=False).items()}

        results["agents"].append(entry)
        dump(results, out_json)  # incremental: survive a crash
        del model, adapter
        gc.collect()

    # --- test window only after training (keeps peak memory lower) -----------
    log("[step3] building test env")
    env_test = R.make_env(
        start=TEST[0], end=TEST[1],
        use_variables=env_train.data_source.df.columns[1:].to_list(),
        scaler=env_train.data_source.scaler,
    )
    log(f"[step3] test env rows={len(env_test.data_source.df)} "
        f"obs={env_test.observation_space.shape}")
    results["test_obs_dim"] = int(env_test.observation_space.shape[0])

    from stable_baselines3 import A2C as _A2C
    for entry in results["agents"]:
        model = _A2C.load(f"{OUT}/agent_ba_{entry['agent']}.zip")
        res, f_navs, m_navs = ai_trade_performance(
            env_test, model, seeds=EPISODE_SEEDS, num_plays=PLAYS, plot_results=False)
        entry["test"] = {
            "agent_avg": float(res.loc["return", "agent_avg"]),
            "market_avg": float(res.loc["return", "market_avg"]),
            "agent_better": float(res.loc["return", "agent_better"]),
            "agent_retrisk": float(res.loc["return_risk", "agent_avg"]),
            "market_retrisk": float(res.loc["return_risk", "market_avg"]),
            "agent_retrisk_better": float(res.loc["return_risk", "agent_better"]),
            "final_agent_navs": [float(x) for x in f_navs],
            "final_market_navs": [float(x) for x in m_navs],
        }
        entry["actions_test"] = {k: int(v) for k, v in
                                 analyze_actions_taken(env_test, model, plot_actions=False).items()}
        log(f"[step3]   test : agent {entry['test']['agent_avg']:.4f} vs "
            f"buy&hold {entry['test']['market_avg']:.4f} | agent_better "
            f"{entry['test']['agent_better']:.2f}")
        dump(results, out_json)
        del model
        gc.collect()

    log(f"[step3] DONE in {time.time()-t0:.1f}s -> {out_json}")


if __name__ == "__main__":
    main()
