# src/

| File | What it is |
|---|---|
| `notebook_code.py` | every code cell of `notebooks/PROJECT_ASSIGNMENT_VOIZARD_clean.ipynb`, extracted without markdown and without outputs |
| `env_runner.py` | builds the trading environment on **local CSV data**; patches `pandas_ta` → `pandas_ta_classic`, forces serial indicator computation, stubs `yfinance`/`tensorflow` |
| `train_eval.py` | trains 5 A2C agents (75,000 steps, seeds 100–104) and evaluates them with the helper package's `ai_trade_performance` on train **and** test window |
| `plot_results.py` | prints the results table and writes `figures/update_2026_a2c.png` |
| `make_pdf.py` | renders a Markdown memo to PDF (WeasyPrint) |
| `data_update/` | the throwaway scripts that fetched the 2026 market data (Boeing/Toyota history, Airbus from ariva.de, P/E + P/S) |

## Missing dependency: the helper package

`train_eval.py` imports the helper modules:

```python
sys.path.insert(0, ".../src/course")
from aif_analysis import ai_trade_performance, analyze_actions_taken
```

`aif_environment.py` and `aif_analysis.py` are third-party code and are
**not redistributed** in this repository — the upstream repository returns **404**. To re-run the RL part, place those two files in
`src/course/` (and adjust the paths at the top of `train_eval.py` / `env_runner.py`).

Consequence: the RL part of this repository is **not reproducible from scratch**. The
results in `results/results_ba_newdata.json` and `paper/UPDATE_2026_REPORT.md` were produced
with those modules present locally, and the deviations are documented in the report.

## Run order

```bash
python3 data/download_data.py
python3 src/train_eval.py        # ~35 min, writes results/results_ba_newdata.json
python3 src/plot_results.py      # table + figures/update_2026_a2c.png
```
