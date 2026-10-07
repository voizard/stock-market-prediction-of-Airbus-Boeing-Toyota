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

## Why not yfinance?

The original notebook loads prices through `yfinance`. From many hosts (including the one this
re-run was executed on) Yahoo answers with **HTTP 429 on every endpoint**, so the data loader
is replaced by the script above. The Airbus series is not fully scriptable through the same
API; the scripts in `src/data_update/` document how those values were obtained
(stockanalysis.com EPA:AIR page + ariva.de for historical prices).
