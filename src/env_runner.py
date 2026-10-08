"""Runs Chris's aif_environment.py / aif_analysis.py on locally cached OHLCV data.

Deviations from the original notebook (documented, unavoidable in this container):
  * yfinance is IP-blocked  -> DataLoader.load_data() is patched to read local CSVs
  * pandas_ta 0.3.14b is unavailable -> pandas_ta_classic is aliased as `pandas_ta`
    (more indicators: 348 vs. the original 204)
  * multiprocessing Pool in df.ta.strategy() fails here (container process limits)
    -> cores forced to 0 (serial execution, identical results)
"""
import sys
import types
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pandas_ta_classic

sys.modules["pandas_ta"] = pandas_ta_classic
import pandas_ta_classic.core as tacore

# The `.ta` accessor is `AnalysisIndicators`; it spawns a Pool(cpu_count()) in
# strategy(). The container forbids that many processes -> force serial mode.
for _name in ("AnalysisIndicators", "Strategy"):
    _cls = getattr(tacore, _name, None)
    if _cls is not None and hasattr(_cls, "_cores"):
        _cls._cores = 0
        print(f"[runner] patched {_name}._cores = 0 (serial)")

# --- stub yfinance (offline) so that `import aif_environment` succeeds --------
yf = types.ModuleType("yfinance")


class _OfflineTicker:
    def __init__(self, *a, **k):
        pass

    def history(self, *a, **k):
        raise RuntimeError("yfinance is IP-blocked; data comes from local CSVs")


yf.Ticker = _OfflineTicker
sys.modules.setdefault("yfinance", yf)

# --- stub tensorflow (only needed for feat_importance_a2c) -------------------
# Needs a real ModuleSpec: torch._dynamo iterates sys.modules and calls
# importlib.util.find_spec() on every entry, which raises for a spec-less stub.
if "tensorflow" not in sys.modules:
    try:
        import tensorflow  # noqa: F401
    except Exception:
        import importlib.machinery

        tf = types.ModuleType("tensorflow")
        tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
        tf.__loader__ = None
        tf.__path__ = []
        tf.__version__ = "0.0.0-stub"
        tf.keras = None
        sys.modules["tensorflow"] = tf

COURSE = "/data/.openclaw/workspace/aif-latex/course"
DATA = "/data/.openclaw/workspace/aif-latex/data"
if COURSE not in sys.path:
    sys.path.insert(0, COURSE)

import aif_environment as AE  # noqa: E402

CSV = {
    "BA": f"{DATA}/BA_5Y_daily.csv",
    "TM": f"{DATA}/TM_5Y_daily.csv",
    "AIR": f"{DATA}/AIR_5Y_daily.csv",
}

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _load_data_from_csv(self):
    path = CSV[self.ticker]
    d = pd.read_csv(path, index_col=0, parse_dates=True)[COLS]
    d = d.loc[(d.index >= pd.Timestamp(self.start_date)) & (d.index <= pd.Timestamp(self.end_date))]
    self.df = d
    self.raw_data = d.copy()


AE.DataLoader.load_data = _load_data_from_csv


def make_env(ticker="BA", start="2021-10-06", end="2026-10-06", **kw):
    return AE.TradingEnvironment(ticker=ticker, start_date=start, end_date=end, **kw)


if __name__ == "__main__":
    env = make_env()
    print("\n--- environment smoke test ---")
    print("observation_space:", env.observation_space.shape, env.observation_space.dtype)
    print("action_space     :", env.action_space)
    print("df (scaled) cols :", env.data_source.df.shape)
    obs = env.reset(seed=42)
    print("reset() -> obs   :", np.asarray(obs).shape)
    done = False
    n = 0
    r_sum = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        r_sum += r
        n += 1
    print(f"episode finished after {n} steps, sum reward {r_sum:.4f}")
    print("final NAV:", env.simulator.navs[-1], "| buy&hold NAV:", env.simulator.market_navs[-1])
