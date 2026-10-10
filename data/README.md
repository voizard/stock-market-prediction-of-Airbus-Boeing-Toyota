# data/

**No market data is committed to this repository.** The price series belongs to the data
provider, so only the fetch script lives here.

## `BA_5Y_daily.csv` (Boeing, 5 years, daily)

| | |
|---|---|
| Source | stockanalysis.com history API (`/api/symbol/s/BA/history?range=5Y&period=Day`) |
| Canonical link | <https://stockanalysis.com/stocks/ba/history/> |
| Expected form | 7 columns `Date,Open,High,Low,Close,AdjClose,Volume`, oldest first, ~1255 rows |
| Coverage | 2021-10 → today (must start no later than 2021-10-31) |
| License | provider's terms; not redistributed here |

Fetch it with:

```bash
python3 data/download_data.py          # idempotent; validates shape, not just existence
```

The script checks the **form** of the file (header, row count, first date, row width), not
only that a file exists — a truncated download is rejected.

## The other eight series

The same script fetches any symbol (`--symbol`, `--out`). The 2026 re-run and the A3C
generalization test use nine series in total:

| Ticker | Series | Used in |
|---|---|---|
| BA | Boeing | re-run + A3C |
| TM | Toyota | re-run + A3C |
| AIR | Airbus (EPA:AIR, from the Yahoo export — see below) | re-run + A3C |
| LMT | Lockheed Martin | A3C generalization |
| CAT | Caterpillar | A3C generalization |
| JPM | JPMorgan Chase | A3C generalization |
| KO | Coca-Cola | A3C generalization |
| NVDA | NVIDIA | A3C generalization |
| XOM | Exxon Mobil | A3C generalization |

```bash
for S in LMT CAT JPM KO NVDA XOM; do
  python3 data/download_data.py --symbol "$S" --out "data/${S}_5Y_daily.csv"
done
```

These six CSVs are registered in `src/env_runner.py` under `CSV`, which is what makes the
generalization run (`src/run_a3c_generalization.sh`) able to address them by ticker.

## Why not yfinance?

The original notebook loads prices through `yfinance`. From many hosts (including the one this
re-run was executed on) Yahoo answers with **HTTP 429 on every endpoint**, so the data loader
is replaced by the script above. The Airbus series is not fully scriptable through the same
API; the scripts in `src/data_update/` document how those values were obtained
(stockanalysis.com EPA:AIR page + ariva.de for historical prices).
