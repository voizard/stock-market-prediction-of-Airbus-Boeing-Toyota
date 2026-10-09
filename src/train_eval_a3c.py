"""A3C (Asynchronous Advantage Actor-Critic) for Chris's trading environment.

Drop-in counterpart to `train_eval_multi.py` (which trains SB3's *synchronous*
A2C). Same protocol, so A2C vs. A3C differ only in the learning algorithm:

    * same environment (env_runner -> Chris's aif_environment.TradingEnvironment)
    * same train/test windows, same episode seeds, same evaluation functions
      (ai_trade_performance / analyze_actions_taken from aif_analysis.py)
    * same N_AGENTS / SEED_BASE / TIMESTEPS semantics

What is different (the A3C part):
    * N_WORKERS threads, each with its own copy of the environment, run rollouts
      in parallel and push gradients into one *shared global* network
      (Hogwild-style asynchronous updates; only the optimizer step is locked).
    * Local copies of the network are synced from the global net every rollout.
    * n-step returns, entropy bonus, value loss, gradient clipping.
    * RMSprop with the same defaults as SB3's A2C (lr=7e-4, alpha=0.99, eps=1e-5)
      so the comparison isolates "synchronous vs. asynchronous".

The trained net is wrapped in `A3CAgent`, which mimics the small part of the SB3
`model.predict(obs, deterministic=True) -> (action, state)` interface that the
evaluation helpers use, plus `save`/`load`.

Usage
-----
    python3 train_eval_a3c.py BA                       # full run, 5 agents
    N_AGENTS=1 TIMESTEPS=2000 N_WORKERS=2 python3 train_eval_a3c.py BA   # smoke test

Env vars: N_AGENTS SEED_BASE TIMESTEPS N_WORKERS
          A3C_GAMMA A3C_NSTEP A3C_LR A3C_VF_COEF A3C_ENT_COEF A3C_MAX_GRAD_NORM
          A3C_OBS_NOISE (std of Gaussian noise added to the observation while
          training, i.e. a cheap regulariser; 0 = off)
          A3C_TAG (suffix for the output files, so an experiment never
          overwrites the baseline results_<tk>_a3c.json)

Batch 2 ("Kuo lever" = a more diverse training environment):
          A3C_BOOT (N block-bootstrap replicates of the training window; agent i
          trains on replicate i mod N. Blocks of A3C_BOOT_BLOCK consecutive rows are
          drawn with replacement, so the intra-block structure survives while the
          long-range path is reshuffled -> many plausible market paths instead of one)
          A3C_BOOT_BLOCK A3C_BOOT_SEED
          A3C_ROLLING (K overlapping rolling sub-windows inside the training window)
          A3C_ROLL_OVERLAP (fraction shared by neighbouring windows, default 0.5)
          A3C_ROLL_MODE = cycle (agent i trains on window i mod K -> regime ensemble)
                        | seq   (every agent walks through all K windows in order)
          Both levers leave the test window, the episode seeds and the scaler
          untouched, so the out-of-sample comparison stays honest.
"""
import copy
import gc
import json
import os
import sys
import threading
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

sys.path.insert(0, "/data/.openclaw/workspace/aif-latex/aif_run")
import env_runner as R  # noqa: E402  (patches pandas_ta/yfinance, defines make_env)

sys.path.insert(0, "/data/.openclaw/workspace/aif-latex/course")
from aif_analysis import ai_trade_performance, analyze_actions_taken  # noqa: E402

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
TICKER = (sys.argv[1] if len(sys.argv) > 1 else "BA").upper()
TK = TICKER.lower()

OUT = "/data/.openclaw/workspace/aif-latex/aif_run"
TRAIN = ("2021-10-06", "2024-05-31")
TEST = ("2024-06-01", "2026-10-06")
N_AGENTS = int(os.environ.get("N_AGENTS", "5"))
SEED_BASE = int(os.environ.get("SEED_BASE", "100"))
TIMESTEPS = int(os.environ.get("TIMESTEPS", "75000"))
PLAYS = 20
EPISODE_SEEDS = list(range(20))  # identical episodes for agent and buy & hold

N_WORKERS = int(os.environ.get("N_WORKERS", "4"))
HIDDEN = int(os.environ.get("A3C_HIDDEN", "64"))
GAMMA = float(os.environ.get("A3C_GAMMA", "0.99"))
NSTEP = int(os.environ.get("A3C_NSTEP", "5"))
LR = float(os.environ.get("A3C_LR", "7e-4"))
VF_COEF = float(os.environ.get("A3C_VF_COEF", "0.5"))
ENT_COEF = float(os.environ.get("A3C_ENT_COEF", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("A3C_MAX_GRAD_NORM", "0.5"))
OBS_NOISE = float(os.environ.get("A3C_OBS_NOISE", "0.0"))
TAG = os.environ.get("A3C_TAG", "").strip()
SUFFIX = f"_{TAG}" if TAG else ""

# --- Batch 2: environment-diversity levers (Kuo) ----------------------------
BOOT = int(os.environ.get("A3C_BOOT", "0"))            # block-bootstrap replicates
BOOT_BLOCK = int(os.environ.get("A3C_BOOT_BLOCK", "20"))  # rows per resampled block
BOOT_SEED = int(os.environ.get("A3C_BOOT_SEED", "7"))
BOOT_MIX = int(os.environ.get("A3C_BOOT_MIX", "0"))    # also train on the real window first
ROLLING = int(os.environ.get("A3C_ROLLING", "0"))       # rolling sub-windows
ROLL_OVERLAP = float(os.environ.get("A3C_ROLL_OVERLAP", "0.5"))
ROLL_MODE = os.environ.get("A3C_ROLL_MODE", "cycle").strip().lower()


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    """Shared trunk + policy head (3 logits) + value head (scalar)."""

    def __init__(self, obs_dim, n_actions=3, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h)


class A3CAgent:
    """SB3-compatible wrapper around a trained ActorCritic net."""

    def __init__(self, model, obs_dim, hidden=64, n_actions=3):
        self.model = model
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.n_actions = int(n_actions)
        self.model.eval()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        x = torch.as_tensor(obs.reshape(1, -1) if single else obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(x)
            actions = logits.argmax(dim=1) if deterministic else Categorical(logits=logits).sample()
        a = actions.cpu().numpy().astype(np.int64)
        if single:
            return a[0], None  # 0-d array, exactly like SB3
        return a, None

    def save(self, path):
        torch.save(
            {"state_dict": self.model.state_dict(), "obs_dim": self.obs_dim,
             "hidden": self.hidden, "n_actions": self.n_actions},
            path,
        )

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu")
        model = ActorCritic(blob["obs_dim"], blob["n_actions"], blob["hidden"])
        model.load_state_dict(blob["state_dict"])
        return cls(model, blob["obs_dim"], blob["hidden"], blob["n_actions"])


# --------------------------------------------------------------------------- #
# A3C training
# --------------------------------------------------------------------------- #
def _worker(worker_id, global_model, optimizer, base_env, cfg, opt_lock, reset_lock):
    seed = cfg["seed"] + worker_id
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = copy.deepcopy(base_env)
    local = ActorCritic(cfg["obs_dim"], cfg["n_actions"], cfg["hidden"])
    local.train()

    budget = cfg["steps_per_worker"]
    done_steps = 0
    s = env.reset(seed=int(rng.integers(1, 10 ** 6)))

    while done_steps < budget:
        with opt_lock:  # consistent local copy for the rollout
            local.load_state_dict(global_model.state_dict())

        log_probs, values, rewards, entropies = [], [], [], []
        last_done = False
        for _ in range(cfg["nstep"]):
            obs_np = np.asarray(s, dtype=np.float32)
            if cfg["obs_noise"] > 0.0:  # regularisation: noisy observations while training
                obs_np = obs_np + rng.normal(
                    0.0, cfg["obs_noise"], size=obs_np.shape).astype(np.float32)
            x = torch.as_tensor(obs_np).unsqueeze(0)
            logits, value = local(x)
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a).reshape(-1))
            entropies.append(dist.entropy().reshape(-1))
            values.append(value.reshape(-1))

            s, r, done, _ = env.step(int(a.item()))
            rewards.append(float(r))
            done_steps += 1
            last_done = bool(done)
            if last_done:
                with reset_lock:
                    s = env.reset(seed=int(rng.integers(1, 10 ** 6)))
                break
            if done_steps >= budget:
                break

        if not rewards:
            break

        # n-step bootstrapped return
        if last_done:
            R = 0.0
        else:
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0)
                _, v = local(x)
                R = float(v.item())
        returns = []
        for r in reversed(rewards):
            R = r + cfg["gamma"] * R
            returns.append(R)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32)

        values_t = torch.cat(values)
        log_probs_t = torch.cat(log_probs)
        entropies_t = torch.cat(entropies)

        advantage = returns_t - values_t.detach()
        actor_loss = -(log_probs_t * advantage).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        loss = actor_loss + cfg["vf_coef"] * critic_loss - cfg["ent_coef"] * entropies_t.mean()

        local.zero_grad(set_to_none=True)
        loss.backward()

        with opt_lock:  # async push of this worker's gradients into the global net
            # NOTE: no optimizer.zero_grad() here - every global grad is overwritten
            # below, and calling zero_grad outside the lock would race with another
            # worker's optimizer.step() (grad becomes None mid-step).
            for gp, lp in zip(global_model.parameters(), local.parameters()):
                gp.grad = lp.grad.clone()
            nn.utils.clip_grad_norm_(global_model.parameters(), cfg["max_grad_norm"])
            optimizer.step()


def train_a3c(base_env, seed, obs_dim, n_actions=3, log=None, model=None, steps=None, label=""):
    steps = int(TIMESTEPS if steps is None else steps)
    cfg = {
        "seed": seed, "obs_dim": obs_dim, "n_actions": n_actions, "hidden": HIDDEN,
        "gamma": GAMMA, "nstep": NSTEP, "vf_coef": VF_COEF, "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM, "obs_noise": OBS_NOISE,
        "steps_per_worker": max(NSTEP, steps // N_WORKERS),
    }
    global_model = model if model is not None else ActorCritic(obs_dim, n_actions, HIDDEN)
    global_model.train()
    optimizer = torch.optim.RMSprop(global_model.parameters(), lr=LR, alpha=0.99, eps=1e-5)

    opt_lock = threading.Lock()
    reset_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(i, global_model, optimizer, base_env, cfg, opt_lock, reset_lock),
            daemon=True,
        )
        for i in range(N_WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = cfg["steps_per_worker"] * N_WORKERS
    if log:
        log(f"[{TK}] A3C trained{label}: {N_WORKERS} workers x {cfg['steps_per_worker']} steps "
            f"(~{total} env steps, target {steps})")
    global_model.eval()
    return global_model


# --------------------------------------------------------------------------- #
# Batch 2 helpers: environment diversity (Kuo lever)
# --------------------------------------------------------------------------- #
def _bootstrap_csvs(ticker, start, end, n_reps, block, seed, outdir="/tmp/a3c_boot"):
    """Block-bootstrap the raw OHLCV series -> n_reps synthetic CSVs.

    Blocks of `block` consecutive rows are drawn with replacement and concatenated,
    so within-block structure (autocorrelation, OHLC relations) survives while the
    long-range path is reshuffled. Written as normal OHLCV CSVs and registered in
    env_runner.CSV under synthetic tickers, so the untouched environment reads them.
    """
    src = pd.read_csv(R.CSV[ticker], index_col=0, parse_dates=True)[R.COLS]
    src = src.loc[(src.index >= pd.Timestamp(start)) & (src.index <= pd.Timestamp(end))]
    m = len(src)
    if m < 2 * block:
        raise SystemExit(f"bootstrap window too short: {m} rows for block={block}")
    vals = src.to_numpy(dtype=float)
    n_blocks = int(np.ceil(m / block))
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    idx = pd.bdate_range(start=start, periods=m)
    paths = []
    for j in range(n_reps):
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        syn = np.concatenate([vals[s:s + block] for s in starts], axis=0)[:m]
        p = f"{outdir}/{ticker}_boot{j}.csv"
        pd.DataFrame(syn, index=idx, columns=R.COLS).to_csv(p)
        R.CSV[f"{ticker}_BOOT{j}"] = p
        paths.append(p)
    return paths


def _bootstrap_csvs(ticker, start, end, n_reps, block, seed, outdir="/tmp/a3c_boot"):
    """Stationary block bootstrap of the training window -> n_reps synthetic CSVs.

    Resampling raw *levels* would glue price gaps onto the block seams, so the
    bootstrap works on the daily *shape* instead: log(Close/prev Close),
    log(Open/prev Close), log(High/Close), log(Low/Close) and log(Volume). Blocks of
    `block` consecutive shape rows are drawn with replacement and the path is rebuilt
    from the first real close, so the synthetic series is continuous by construction
    while its long-range order is reshuffled (within-block autocorrelation survives).
    """
    src = pd.read_csv(R.CSV[ticker], index_col=0, parse_dates=True)[R.COLS]
    src = src.loc[(src.index >= pd.Timestamp(start)) & (src.index <= pd.Timestamp(end))]
    m = len(src)
    if m < 2 * block:
        raise SystemExit(f"bootstrap window too short: {m} rows for block={block}")
    prev_close = src["Close"].shift(1).to_numpy()
    shape = np.column_stack([
        np.log(src["Close"].to_numpy() / prev_close),
        np.log(src["Open"].to_numpy() / prev_close),
        np.log(src["High"].to_numpy() / src["Close"].to_numpy()),
        np.log(src["Low"].to_numpy() / src["Close"].to_numpy()),
        np.log(np.maximum(src["Volume"].to_numpy(dtype=float), 1.0)),
    ])[1:]  # drop the first row: no previous close
    c0 = float(src["Close"].iloc[0])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil((m - 1) / block))
    idx = pd.bdate_range(start=start, periods=m)
    head = pd.DataFrame([src.iloc[0][R.COLS].to_numpy()], index=idx[:1], columns=R.COLS)
    paths = []
    for j in range(n_reps):
        starts = rng.integers(0, len(shape) - block + 1, size=n_blocks)
        rows = np.concatenate([shape[s:s + block] for s in starts], axis=0)[:m - 1]
        close = c0 * np.exp(np.cumsum(rows[:, 0]))
        pc = np.concatenate([[c0], close[:-1]])
        body = pd.DataFrame({
            "Open": pc * np.exp(rows[:, 1]),
            "High": close * np.exp(rows[:, 2]),
            "Low": close * np.exp(rows[:, 3]),
            "Close": close,
            "Volume": np.exp(rows[:, 4]),
        }, index=idx[1:])[R.COLS]
        p = f"{outdir}/{ticker}_boot{j}.csv"
        os.makedirs(outdir, exist_ok=True)
        pd.concat([head, body]).to_csv(p)
        R.CSV[f"{ticker}_BOOT{j}"] = p
        paths.append(p)
    return paths


def _rolling_windows(start, end, k, overlap):
    """K equal-length sub-windows of [start, end] sharing `overlap` of their span.

    NOTE: unusable for the 5-year CSVs in this project. `DataLoader` asserts
    len(df) > 2*trading_days (= 504 rows) per environment, and the BA/TM training
    window holds only ~634 usable rows, so any genuine sub-window is rejected. The
    guard below turns that into a clear error instead of an assertion deep in the env.
    """
    idx = pd.bdate_range(start=start, end=end)
    m = len(idx)
    span = int(round(m / (1 + (k - 1) * (1 - overlap))))
    step = max(1, int(round(span * (1 - overlap))))
    wins = []
    for i in range(k):
        a = idx[min(i * step, m - 1)]
        b = idx[min(i * step + span - 1, m - 1)]
        wins.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    shortest = min(len(pd.bdate_range(a, b)) for a, b in wins)
    if shortest <= 2 * 252:
        raise SystemExit(
            f"rolling windows unusable: shortest window has {shortest} rows, the env "
            f"requires >504 (2 x trading_days). Use A3C_BOOT instead.")
    return wins


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg):
    with open("/proc/self/status") as f:
        vm = [l for l in f if l.startswith("VmRSS:")][0].split()[1]
    print(f"{msg} [RSS {int(vm)/1024:.0f} MB]", flush=True)


def dump(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def evaluate(env, agent):
    res, f_navs, m_navs = ai_trade_performance(
        env, agent, seeds=EPISODE_SEEDS, num_plays=PLAYS, plot_results=False)
    out = {
        "agent_avg": float(res.loc["return", "agent_avg"]),
        "market_avg": float(res.loc["return", "market_avg"]),
        "agent_better": float(res.loc["return", "agent_better"]),
        "agent_retrisk": float(res.loc["return_risk", "agent_avg"]),
        "market_retrisk": float(res.loc["return_risk", "market_avg"]),
        "agent_retrisk_better": float(res.loc["return_risk", "agent_better"]),
        "final_agent_navs": [float(x) for x in f_navs],
        "final_market_navs": [float(x) for x in m_navs],
    }
    actions = {k: int(v) for k, v in analyze_actions_taken(env, agent, plot_actions=False).items()}
    return out, actions


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    log(f"[{TK}] A3C | ticker={TICKER} | train {TRAIN} | test {TEST} | "
        f"{N_AGENTS} agents x {TIMESTEPS} steps | {N_WORKERS} workers | "
        f"nstep={NSTEP} ent={ENT_COEF} obs_noise={OBS_NOISE} "
        f"boot={BOOT} rolling={ROLLING}/{ROLL_MODE} tag={TAG or '-'}")
    t0 = time.time()
    env_train = R.make_env(ticker=TICKER, start=TRAIN[0], end=TRAIN[1])
    obs_dim = int(env_train.observation_space.shape[0])
    log(f"[{TK}] train env rows={len(env_train.data_source.df)} obs_dim={obs_dim} "
        f"({time.time()-t0:.1f}s)")

    # --- per-agent training environments (Batch 2 levers) --------------------
    train_kw = {"use_variables": env_train.data_source.df.columns[1:].to_list(),
                "scaler": env_train.data_source.scaler}
    full = [(TICKER, TRAIN[0], TRAIN[1])]
    plans, plan_note = [], "full train window"
    if BOOT > 0:
        _bootstrap_csvs(TICKER, TRAIN[0], TRAIN[1], BOOT, BOOT_BLOCK, BOOT_SEED)
        if BOOT_MIX:
            plans = [[full[0], (f"{TICKER}_BOOT{i % BOOT}", TRAIN[0], TRAIN[1])]
                     for i in range(N_AGENTS)]
        else:
            plans = [[(f"{TICKER}_BOOT{i % BOOT}", TRAIN[0], TRAIN[1])] for i in range(N_AGENTS)]
        plan_note = f"block bootstrap, {BOOT} replicates, block={BOOT_BLOCK} rows"
        if BOOT_MIX:
            plan_note += " + real window first"
        log(f"[{TK}] {plan_note}")
    elif ROLLING > 1:
        wins = _rolling_windows(TRAIN[0], TRAIN[1], ROLLING, ROLL_OVERLAP)
        if ROLL_MODE == "seq":
            plans = [[(TICKER, a, b) for a, b in wins] for _ in range(N_AGENTS)]
        else:
            plans = [[(TICKER, *wins[i % ROLLING])] for i in range(N_AGENTS)]
        plan_note = f"rolling windows ({ROLL_MODE}, overlap {ROLL_OVERLAP}): {wins}"
        log(f"[{TK}] {plan_note}")
    else:
        plans = [list(full) for _ in range(N_AGENTS)]

    results = {"ticker": TICKER, "algo": "A3C", "train_window": TRAIN, "test_window": TEST,
               "agents": [], "note": "train env: " + plan_note,
               "hparams": {"n_workers": N_WORKERS, "gamma": GAMMA, "nstep": NSTEP, "lr": LR,
                           "vf_coef": VF_COEF, "ent_coef": ENT_COEF, "hidden": HIDDEN,
                           "obs_noise": OBS_NOISE, "tag": TAG, "boot": BOOT,
                           "boot_block": BOOT_BLOCK, "boot_mix": BOOT_MIX, "rolling": ROLLING,
                           "roll_mode": ROLL_MODE, "roll_overlap": ROLL_OVERLAP}}
    out_json = f"{OUT}/results_{TK}_a3c{SUFFIX}.json"
    if os.path.exists(out_json):  # resume
        with open(out_json) as f:
            results = json.load(f)
        results.update({"ticker": TICKER, "algo": "A3C", "train_window": TRAIN, "test_window": TEST})
        log(f"[{TK}] resuming {out_json}: {len(results.get('agents', []))} agents present")
    results["seed_base"] = SEED_BASE
    results["n_agents"] = N_AGENTS
    have = {a["agent"]: a for a in results.get("agents", [])}

    for i in range(N_AGENTS):
        n = i + 1
        if n in have and "train" in have[n] and os.path.exists(f"{OUT}/agent_a3c_{TK}_{n}{SUFFIX}.pt"):
            log(f"[{TK}] agent {n}/{N_AGENTS} already done - skip")
            continue
        ts = time.time()
        plan = plans[i]
        per = max(NSTEP, TIMESTEPS // len(plan))
        model = None
        for w, (tk, a, b) in enumerate(plan):
            env_w = env_train if (tk, a, b) == full[0] else R.make_env(
                ticker=tk, start=a, end=b, **train_kw)
            if env_w is not env_train:
                d_w = int(env_w.observation_space.shape[0])
                if d_w != obs_dim:
                    raise SystemExit(f"obs_dim mismatch on {tk} {a}..{b}: {d_w} != {obs_dim}")
            steps_w = TIMESTEPS - per * w if w == len(plan) - 1 else per
            model = train_a3c(env_w, SEED_BASE + i + w * 7919, obs_dim, log=log, model=model,
                              steps=steps_w, label=f" [{w + 1}/{len(plan)} {a}..{b}]")
            if env_w is not env_train:
                del env_w
                gc.collect()
        agent = A3CAgent(model, obs_dim, HIDDEN)
        log(f"[{TK}] agent {n}/{N_AGENTS} trained in {time.time()-ts:.1f}s")
        agent.save(f"{OUT}/agent_a3c_{TK}_{n}{SUFFIX}.pt")

        entry = {"agent": n, "seed": SEED_BASE + i}
        entry["train"], entry["actions_train"] = evaluate(env_train, agent)
        log(f"[{TK}]   train: agent {entry['train']['agent_avg']:.4f} vs "
            f"buy&hold {entry['train']['market_avg']:.4f} | agent_better "
            f"{entry['train']['agent_better']:.2f}")

        results["agents"].append(entry)
        have[n] = entry
        results["agents"] = [have[k] for k in sorted(have)]
        dump(results, out_json)
        del model, agent
        gc.collect()

    # --- test window ---------------------------------------------------------
    log(f"[{TK}] building test env")
    env_test = R.make_env(
        ticker=TICKER, start=TEST[0], end=TEST[1],
        use_variables=env_train.data_source.df.columns[1:].to_list(),
        scaler=env_train.data_source.scaler,
    )
    log(f"[{TK}] test env rows={len(env_test.data_source.df)}")
    results["test_obs_dim"] = int(env_test.observation_space.shape[0])

    for entry in sorted(results["agents"], key=lambda e: e["agent"]):
        if "test" in entry:
            log(f"[{TK}]   test : agent {entry['agent']} already evaluated - skip")
            continue
        agent = A3CAgent.load(f"{OUT}/agent_a3c_{TK}_{entry['agent']}{SUFFIX}.pt")
        entry["test"], entry["actions_test"] = evaluate(env_test, agent)
        log(f"[{TK}]   test : agent {entry['test']['agent_avg']:.4f} vs "
            f"buy&hold {entry['test']['market_avg']:.4f} | agent_better "
            f"{entry['test']['agent_better']:.2f}")
        dump(results, out_json)
        del agent
        gc.collect()

    log(f"[{TK}] DONE in {time.time()-t0:.1f}s -> {out_json}")


if __name__ == "__main__":
    main()
