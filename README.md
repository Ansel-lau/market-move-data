# market-move-data

A free data robot for the **market-move-explainer** Claude skill.

Every weekday at 6:17 am Singapore time, GitHub Actions runs `daily_data.py`.
It downloads a year of prices for every asset in `watchlist.csv`, computes
each day's move, how unusual it was (sigma) and the market / sector /
specific split, then saves the results in `data/`:

| File | What it holds |
| --- | --- |
| `data/latest.json` | The most recent US session |
| `data/YYYY-MM-DD.json` | One file per session (the archive) |
| `data/index.json` | The list of sessions available |

Claude reads these files from the raw URL, e.g.
`https://raw.githubusercontent.com/<your-username>/market-move-data/main/data/latest.json`.

## Sources (all free)
- Prices: Yahoo Finance via `yfinance` (unofficial; update the library if it breaks)
- Rates: US Treasury daily yield curves and the New York Fed (FRED as a fallback).
  The 2-year yield uses the Treasury curve (`fred:DGS2`), not Yahoo's `2YY=F`,
  which is rarely updated.
- Headlines and earnings dates: Finnhub (free key, stored as the
  `FINNHUB_API_KEY` repository secret, never in a file)

## Common jobs
- **Add an asset:** add a row to `watchlist.csv` (Yahoo symbol in `symbol`),
  and a row in the Notion Watchlist.
- **Rebuild a past day:** Actions > Daily market data > Run workflow, type the date.
- **Fill history:** Run workflow with `backfill` = 20.
- **Test on your Mac:** `pip install -r requirements.txt` then `python daily_data.py`.

The method is explained in the skill's `reference/attribution-method.md`.
