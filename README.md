# Applied Reinforcement Learning (A2C) on Transportation Stocks — with a 2026 data update

The project trains **A2C** agents (Stable-Baselines3) on daily price series of **Boeing (BA)**,
**Airbus (AIR.PA)** and **Toyota (TM)**, compares them against a **buy-and-hold** baseline, and
tests whether the agent's actions carry any predictive value.

---

## Key findings

- **The 2022 notebook result is reproduced and holds up: A2C does not beat buy-and-hold
  reliably.** In the original run the agent reached a return of 1.167 vs. the market's 1.264
  on Boeing.
- **The 2026 re-run on fresh data (Boeing, 2024-06 → 2026-10, out-of-sample)** gives
  agent **0.986** vs. buy-and-hold **0.946** on average NAV, with the agent ahead in **65 %**
  of the 20 evaluation episodes — but with a spread from **0.56** (agent 1) to **1.24**
  (agent 2). The advantage is **within seed variance and not significant**.
- **Buy-and-hold loses money in both windows** (NAV 0.974 in-sample, 0.946 out-of-sample):
  Boeing fell over both periods, so the real question is "lose less", not "win".
- **The data update uncovered a factual error in the original notebook:** the reported Toyota
  P/S of `0.0063` ("close to zero") was a yfinance outlier. The real value is **0.68**.
- **Log returns stay negative for all three stocks** in the last 12 months
  (BA −15.6 %, AIR −9.5 %, TM −7.4 %), but **positive since August 2022**
  (BA +20 %, TM +40 %, AIR +50–65 %) — the loss picture was window-specific.

## Results — 2026 re-run (Boeing, 5 agents, 20 episodes each)

| # | Seed | Train agent | Train B&H | Agent better | Test agent | Test B&H | Agent better |
|---|------|------------:|----------:|-------------:|-----------:|---------:|-------------:|
| 1 | 100 | 1.6705 | 0.9647 | 100 % | **0.5616** | 0.9475 | 0 % |
| 2 | 101 | 2.4073 | 0.9671 | 100 % | 1.2403 | 0.9408 | 100 % |
| 3 | 102 | 4.0534 | 0.9956 | 100 % | 0.9828 | 0.9412 | 75 % |
| 4 | 103 | 7.0580 | 0.9711 | 100 % | 1.0106 | 0.9493 | 70 % |
| 5 | 104 | 5.1547 | 0.9727 | 100 % | 1.1326 | 0.9485 | 80 % |
| **Ø** | | **4.0688** | **0.9742** | **100 %** | **0.9856** | **0.9455** | **65 %** |

Training window 2021-10-06 → 2024-05-31 (in-sample), test window 2024-06-01 → 2026-10-06
(out-of-sample). The in-sample column is **not** evidence of skill — the agents were trained
on exactly that window. Full numbers: `results/results_ba_newdata.json`.

![A2C on Boeing, new data](figures/update_2026_a2c.png)

## Data sources

| Data | Source | Note |
|---|---|---|
| Boeing / Toyota prices | stockanalysis.com history API | `data/download_data.py` |
| Airbus prices | stockanalysis.com (EPA:AIR) + ariva.de | not fully scriptable; see `src/data_update/` |
| P/E, P/S | stockanalysis.com | trailing values |
| Course package (`aif_course`) | `github.com/RalfKellner/aif_course` | **404 — no longer available** |

The price CSV is **not** redistributed here — it belongs to the data provider. Fetch it with
`data/download_data.py`; see `data/README.md`.

## Methods

- **Fundamentals**: P/E and P/S as trailing values; "12-month return" as the sum of daily
  log returns of the adjusted close.
- **RL**: `A2C("MlpPolicy")`, 75,000 timesteps per agent, five seeds (100–104); evaluation with
  the course's `ai_trade_performance(num_plays=20)`. The test environment re-uses the training
  environment's feature list and scaler, as in the notebook.
- **Fair comparison**: agent and buy-and-hold run on the *same* 20 episode seeds.

## Limitations (read before trusting any number)

1. **Different environment, different feature count.** The re-run uses `pandas_ta_classic`,
   which provides **348 indicators instead of 204** — the observation vector grows from
   **205 to 334**. Part of any performance difference can come from that, not from the data update.
2. **The course package is gone.** `aif_environment.py` / `aif_analysis.py` (instructor's code,
   not redistributed here) are required to run the RL part; their repository returns 404.
   The re-run therefore uses a locally reconstructed environment runner (`src/env_runner.py`)
   with documented substitutions (local CSVs instead of yfinance, serial instead of
   multiprocessing).
3. **Library drift.** Original: SB3 1.6.0 + gym 0.21 + torch 1.12.1 + Python 3.8.
   Re-run: SB3 2.9 + gymnasium 1.4 + torch 2.x + Python 3.13.
4. **Only one pre-trained agent was available** (`A2C_showcase_agent.zip`), while the notebook
   uses five — so "original agent vs. new agent" could not be compared.
5. **In-sample results are not evidence.** The train column only shows that the agents learned
   the training window.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/                  README + download script (no data committed)
├── notebooks/             the cleaned assignment notebook
├── src/                   code without markdown/outputs + the 2026 re-run
│   └── data_update/       scripts that fetched the 2026 market data
├── paper/                 LaTeX report, PDF and the 2026 update report
│   └── latex/             main.tex, parts/, figures, code listing
├── figures/               result figures
└── results/               raw JSON of the 2026 re-run
```

## Reproduction

```bash
pip install -r requirements.txt

python3 data/download_data.py          # fetches data/BA_5Y_daily.csv (~1255 rows)
python3 src/train_eval.py              # trains 5 A2C agents, evaluates train + test (~35 min)
python3 src/plot_results.py            # prints the table above and writes the figure

# reports
python3 src/make_pdf.py paper/UPDATE_2026_REPORT.md paper/AIF_Update_2026.pdf
cd paper/latex && mkdir -p out && tectonic main.tex --outdir out
```

`src/train_eval.py` needs the course modules `aif_environment.py` and `aif_analysis.py` in
`src/course/` (see limitation 2 — they are course material and not part of this repository).

## Provenance

The notebook in `notebooks/` is the author's own work. It was cleaned up (spelling,
execution counters, relative paths) without changing results. The 2026 update
(`paper/UPDATE_2026_REPORT.md`, `src/`) is independent work by the same author. The template
and helper package the notebook builds on belong to a third party and are **not**
redistributed here.

## License

MIT — see [LICENSE](LICENSE). The price data is subject to the data provider's terms and is
not covered by this license.
