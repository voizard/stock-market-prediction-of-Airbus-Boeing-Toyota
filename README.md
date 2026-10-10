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
  agent **1.094** vs. buy-and-hold **0.943** on average NAV over **20 seeds**, with the agent
  ahead in **75 %** of the 400 evaluation episodes. The margin (**+0.151**, paired *t*-test
  *p* = 0.0037) **is significant** — with only 5 seeds it had been +0.040 and inside the seed
  spread, i.e. indistinguishable from noise.
- **With 20 seeds on all three stocks, the in-sample margin does not predict the out-of-sample
  outcome at all.** In-sample the agent beats buy-and-hold everywhere (+0.53 to +2.70,
  *p* < 0.0001); out-of-sample only Boeing keeps an advantage, **Airbus lands exactly on
  buy-and-hold** (+0.005, *p* = 0.89, 47.8 % hit rate) and **Toyota turns negative**
  (−0.130, *p* = 0.0001, 13.8 % hit rate). Toyota was the *second best* in-sample and the
  *worst* out-of-sample; Airbus the *worst* in-sample and the *second best* out-of-sample.
- **Buy-and-hold loses money in both windows** (NAV 0.974 in-sample, 0.946 out-of-sample):
  Boeing fell over both periods, so the real question is "lose less", not "win".
- **The data update uncovered a factual error in the original notebook:** the reported Toyota
  P/S of `0.0063` ("close to zero") was a yfinance outlier. The real value is **0.68**.
- **Log returns stay negative for all three stocks** in the last 12 months
  (BA −15.6 %, AIR −9.5 %, TM −7.4 %), but **positive since August 2022**
  (BA +20 %, TM +40 %, AIR +50–65 %) — the loss picture was window-specific.
- **A second algorithm (A3C) confirms the result, and a decomposition shows where the
  effect really comes from.** Asynchronous A3C on Boeing, 20 seeds, has **no edge** over
  buy-and-hold either (margin +0.019, hit rate 51 %, *p* = 0.32). Splitting the usual
  three-factor “tuning” bundle shows the effect comes from the **credit assignment**
  (`nstep` 5→10, 85 % of the bundle), not from regularisation — the entropy bonus and
  observation noise are inert. A **block bootstrap** of the *daily returns* (many
  synthetic paths instead of the one real path) adds ≈ +0.09 and, combined with `nstep` 10,
  gives the **first arm that beats the baseline significantly** (+0.200, *p* = 0.030).
- **A replication on the other two stocks tempers that: the mechanism is reproducible, the
  payoff is not.** Repeating the whole protocol on **Airbus** and **Toyota** (20 seeds each)
  gives `nstep 10 + bootstrap − baseline` = **+0.175 on Toyota** (*p* = 0.008) but
  **−0.145 on Airbus** (*p* = 0.012), and **+0.015 pooled** (*p* = 0.75, *n* = 40). The
  bootstrap removes memorisation on every stock (in-sample margin ≈ 0 vs. +2…+5), but its
  out-of-sample benefit is stock-specific. The “first significant edge” is therefore a
  single-stock result, not a general property of the method.
- **A generalization check on six further, sector-diverse stocks does not confirm the
  bootstrap benefit.** Repeating the whole protocol on **LMT, CAT, JPM, KO, NVDA and XOM**
  (20 seeds each) reproduces the *mechanism* — the bootstrap arm has an in-sample margin of
  ≈ 0 on all six — but not the *payoff*: pooled over the six the paired
  `nstep 10 + bootstrap − baseline` is **−0.036** (*p* = 0.36, *n* = 120), and pooled over
  all nine price series **+0.002** (*p* = 0.96, *n* = 180). Two stocks improve significantly
  (XOM +0.157, *p* = 0.004) and two deteriorate significantly (NVDA −0.248, *p* = 0.012);
  pooled they cancel out. With nine price series the honest conclusion is: the mechanism is
  robust, the payoff is stock-specific.

## Results — 2026 re-run, 20 seeds per stock

Mean end-NAV over 20 seeds (100–119), start = 1.0. Margin = agent − buy-and-hold; *p* from a
paired *t*-test over the seed differences. Hit rate over all 400 episodes per stock and window.

| Stock | Window | Agent | Spread | 95 % CI | Buy & Hold | Margin | *p* | Hit rate |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Boeing | in-sample | 3.669 | 1.400 | [3.01; 4.33] | 0.973 | **+2.697** | < 0.0001 | 100.0 % |
| Boeing | out-of-sample | 1.094 | 0.231 | [0.99; 1.20] | 0.943 | **+0.151** | 0.0037 | 75.0 % |
| Airbus | in-sample | 1.818 | 0.423 | [1.62; 2.02] | 1.292 | **+0.526** | < 0.0001 | 79.8 % |
| Airbus | out-of-sample | 1.096 | 0.156 | [1.02; 1.17] | 1.091 | **+0.005** | 0.887 | 47.8 % |
| Toyota | in-sample | 2.455 | 0.724 | [2.12; 2.79] | 1.594 | **+0.861** | < 0.0001 | 87.0 % |
| Toyota | out-of-sample | 0.852 | 0.144 | [0.79; 0.92] | 0.983 | **−0.130** | 0.0001 | 13.8 % |

![In-sample vs. out-of-sample margin per stock](figures/update_2026_margins.png)

Training window 2021-10-06 → 2024-05-31 (in-sample), test window 2024-06-01 → 2026-10-06
(out-of-sample). The in-sample column is **not** evidence of skill — the agents were trained
on exactly that window. Raw numbers: `results/results_{ba,air,tm}_newdata.json`.

## Results — Boeing, 5 agents, 20 episodes each

The earlier, smaller run (seeds 100–104, a subset of the 20 above). Its test advantage of
+0.04 NAV points was **not** significant — the reason the run was extended to 20 seeds.

| # | Seed | Train agent | Train B&H | Agent better | Test agent | Test B&H | Agent better |
|---|------|------------:|----------:|-------------:|-----------:|---------:|-------------:|
| 1 | 100 | 1.6705 | 0.9647 | 100 % | **0.5616** | 0.9475 | 0 % |
| 2 | 101 | 2.4073 | 0.9671 | 100 % | 1.2403 | 0.9408 | 100 % |
| 3 | 102 | 4.0534 | 0.9956 | 100 % | 0.9828 | 0.9412 | 75 % |
| 4 | 103 | 7.0580 | 0.9711 | 100 % | 1.0106 | 0.9493 | 70 % |
| 5 | 104 | 5.1547 | 0.9727 | 100 % | 1.1326 | 0.9485 | 80 % |
| **Ø** | | **4.0688** | **0.9742** | **100 %** | **0.9856** | **0.9455** | **65 %** |

![A2C on Boeing, new data](figures/update_2026_a2c.png)

## Data sources

| Data | Source | Note |
|---|---|---|
| Boeing / Toyota prices | stockanalysis.com history API | `data/download_data.py` |
| Airbus prices | Yahoo Finance, `AIR.PA` (Euronext Paris) | 1,307 trading days to 2026-10-06 |
| P/E, P/S | stockanalysis.com | trailing values |
| RL helper modules (`aif_course`) | third-party helper, available locally, **not** part of this repository | upstream repository: **404** |

The price CSV is **not** redistributed here — it belongs to the data provider. Fetch it with
`data/download_data.py`; see `data/README.md`.

## Methods

- **Fundamentals**: P/E and P/S as trailing values; "12-month return" as the sum of daily
  log returns of the adjusted close.
- **RL**: `A2C("MlpPolicy")`, 75,000 timesteps per agent; **20 seeds (100–119) per stock** in
  the extended run, five seeds (100–104) in the first Boeing run; evaluation with the helper
  package's `ai_trade_performance(num_plays=20)`. The test environment re-uses the training
  environment's feature list and scaler, as in the notebook.
- **Fair comparison**: agent and buy-and-hold run on the *same* 20 episode seeds.
- **Statistics**: margin = mean paired difference agent − buy-and-hold over the seeds;
  two-sided 95 % CI and paired *t*-test over the 20 seed differences (`src/analyze_seeds.py`).

## Limitations (read before trusting any number)

1. **Different environment, different feature count.** The re-run uses `pandas_ta_classic`,
   which provides **348 indicators instead of 204** — the observation vector grows from
   **205 to 334**. Part of any performance difference can come from that, not from the data update.
2. **The RL part is not self-contained.** `aif_environment.py` / `aif_analysis.py` are
   third-party helper modules. They were available locally when the re-run was produced, so
   the numbers are genuine — but they are **not included in this repository** (the upstream
   repository that hosted them returns 404), so a fresh clone has to supply them under
   `src/course/`. The re-run uses a locally reconstructed environment runner
   (`src/env_runner.py`) with documented substitutions (local CSVs instead of yfinance,
   serial instead of multiprocessing).
3. **Library drift.** Original: SB3 1.6.0 + gym 0.21 + torch 1.12.1 + Python 3.8.
   Re-run: SB3 2.9 + gymnasium 1.4 + torch 2.x + Python 3.13.
4. **Only one pre-trained agent was available** (`A2C_showcase_agent.zip`), while the notebook
   uses five — so "original agent vs. new agent" could not be compared.
5. **In-sample results are not evidence.** The train column only shows that the agents learned
   the training window — which the 20-seed run makes explicit: the in-sample margin is large
   everywhere and predicts nothing about the test window.
6. **20 seeds are better than 5, but still a small sample.** The confidence intervals in the
   table above are correspondingly wide.
7. **Airbus prices come from a different source** than Boeing and Toyota (Yahoo, `AIR.PA`,
   in EUR, 1,307 trading days to 2026-10-06).

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/                  README + download script (no data committed)
├── notebooks/             the cleaned assignment notebook
├── src/                   code without markdown/outputs + the 2026 re-run
│   ├── train_eval.py      first re-run (Boeing, 5 agents, notebook protocol)
│   ├── train_eval_multi.py  extended run, any ticker, resumable
│   ├── analyze_seeds.py   mean, spread, 95 % CI, paired t-test
│   ├── plot_margins.py    in-sample vs. out-of-sample margin figure
│   ├── train_eval_a3c.py  A3C counterpart: async workers, n-step returns, block bootstrap
│   ├── plot_a3c_ablation.py  one-factor decomposition + bootstrap figure
│   ├── compare_a3c_repl.py   replication/generalization: paired tests per stock + pooled (any ticker list)
│   ├── plot_a3c_replication.py  replication figure across the three stocks
│   ├── plot_a3c_generalization.py  generalization figure across nine stocks
│   ├── run_a3c_generalization.sh   the six-stock generalization run (6 stocks x 3 arms)
│   └── data_update/       scripts that fetched the 2026 market data
├── paper/                 LaTeX sources and PDFs of both reports
│   ├── latex/             main report: main.tex, parts/, figures, code listing
│   ├── latex-update2026/  2026 update report: main.tex, parts/, figures
│   └── UPDATE_2026_REPORT.md  Markdown version of the update report
├── figures/               result figures
└── results/               raw JSON of the 2026 re-run
```

## Reproduction

```bash
pip install -r requirements.txt

python3 data/download_data.py          # fetches data/BA_5Y_daily.csv (~1255 rows)
python3 src/train_eval.py              # trains 5 A2C agents, evaluates train + test (~35 min)
python3 src/plot_results.py            # prints the 5-agent table and writes the figure

# extended run: 20 seeds per stock (~5.5 min per agent, ~2 h per stock)
python3 src/train_eval_multi.py BA
python3 src/analyze_seeds.py TM BA AIR # tables + significance tests
python3 src/plot_margins.py            # margin figure

# A3C ablation on Boeing (20 seeds, ~2 h per arm)
python3 src/train_eval_a3c.py BA                    # baseline
A3C_NSTEP=10 python3 src/train_eval_a3c.py BA       # + credit assignment
A3C_NSTEP=10 A3C_BOOT=5 A3C_BOOT_BLOCK=20 python3 src/train_eval_a3c.py BA   # + bootstrap
python3 src/plot_a3c_ablation.py                    # decomposition figure
python3 src/compare_a3c_repl.py AIR TM              # replication: paired tests per stock + pooled
python3 src/plot_a3c_replication.py                 # replication figure (3 stocks)
bash src/run_a3c_generalization.sh                  # generalization: 6 stocks x 3 arms, 20 seeds (~10 h)
python3 src/compare_a3c_repl.py LMT CAT JPM KO NVDA XOM   # generalization: pooled over the six
python3 src/compare_a3c_repl.py BA AIR TM LMT CAT JPM KO NVDA XOM  # pooled over all nine
python3 src/plot_a3c_generalization.py              # generalization figure (9 stocks)

# reports (LaTeX via Tectonic)
cd paper/latex && tectonic -X compile main.tex --outdir out              # main report
cd paper/latex-update2026 && tectonic -X compile main.tex --outdir out   # 2026 update
```

`src/train_eval.py` needs the helper modules `aif_environment.py` and `aif_analysis.py` in
`src/course/` (see limitation 2 — they are third-party code and not part of this repository).

## Provenance

The notebook in `notebooks/` is the author's own work. It was cleaned up (spelling,
execution counters, relative paths) without changing results. The 2026 update
(`paper/latex-update2026/`, `src/`) is independent work by the same author. The template
and helper package the notebook builds on belong to a third party and are **not**
redistributed here.

## License

MIT — see [LICENSE](LICENSE). The price data is subject to the data provider's terms and is
not covered by this license.
