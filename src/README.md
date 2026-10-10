# src/

| File | What it is |
|---|---|
| `notebook_code.py` | every code cell of `notebooks/PROJECT_ASSIGNMENT_VOIZARD_clean.ipynb`, extracted without markdown and without outputs |
| `env_runner.py` | builds the trading environment on **local CSV data**; patches `pandas_ta` → `pandas_ta_classic`, forces serial indicator computation, stubs `yfinance`/`tensorflow` |
| `train_eval.py` | trains 5 A2C agents (75,000 steps, seeds 100–104) and evaluates them with the helper package's `ai_trade_performance` on train **and** test window |
| `plot_results.py` | prints the results table and writes `figures/update_2026_a2c.png` |
| `train_eval_a3c.py` | A3C counterpart to `train_eval_multi.py`: asynchronous workers on the shared network, n-step returns, entropy bonus, observation noise and the block bootstrap; writes `results/results_<tk>_a3c[_tag].json` |
| `plot_a3c_ablation.py` | one-factor decomposition and block bootstrap figure, reads `results/results_ba_a3c*.json`, writes `figures/update_2026_a3c.png` |
| `compare_a3c_repl.py` | replication/generalization analysis across stocks: per-arm test/train margins, hit rate, p, and the paired tests per stock plus the pooled test over any ticker list passed on the command line |
| `plot_a3c_replication.py` | replication figure: out-of-sample margin of the two levers against the baseline, per stock + pooled, reads `results/results_{ba,air,tm}_a3c*.json` |
| `plot_a3c_generalization.py` | generalization figure across nine stocks: paired `nstep10boot − baseline` per stock plus pooled (six new, all nine), reads `results/results_{ba,air,tm,lmt,cat,jpm,ko,nvda,xom}_a3c*.json` |
| `run_a3c_generalization.sh` | the six-stock generalization run (LMT, CAT, JPM, KO, NVDA, XOM × baseline/nstep10/nstep10boot, 20 seeds, three waves, resume-safe) |
| `make_pdf.py` | renders a Markdown memo to PDF (WeasyPrint) |
| `data_update/` | the throwaway scripts that fetched the 2026 market data (Boeing/Toyota history, Airbus from ariva.de, P/E + P/S) |

## The two helper modules are not part of this repository

`train_eval.py` imports two third-party modules:

```python
sys.path.insert(0, ".../src/course")
from aif_analysis import ai_trade_performance, analyze_actions_taken
```

`aif_environment.py` and `aif_analysis.py` were available locally when the re-run was
produced, so the results in `results/results_ba_newdata.json` are genuine. They are **not
redistributed here**, and the upstream repository that hosted them returns **404**. To re-run
the RL part, place those two files in `src/course/` and adjust the paths at the top of
`train_eval.py` / `env_runner.py` (they point at the machine that produced the run).

Consequence: a fresh clone of this repository **cannot reproduce the RL part without those
two files**. The deviations of the re-run are documented in the report.

## Run order

```bash
python3 data/download_data.py
python3 src/train_eval.py        # ~35 min, writes results/results_ba_newdata.json
python3 src/plot_results.py      # table + figures/update_2026_a2c.png
```
