#!/usr/bin/env python3
"""
Daily data feed for the market-move-explainer skill.

Run after the US close (GitHub Actions does this every weekday). It writes
  data/latest.json          the most recent US session
  data/<YYYY-MM-DD>.json    an archive copy for that session
  data/index.json           the list of sessions available

For every asset in watchlist.csv it records the close, the move, how unusual
the move was (sigma), the market/sector/specific split (or NA with the
reason), peer moves and, for stocks, recent headlines and earnings dates.

Free sources only
  prices     Yahoo Finance via yfinance (no key)
  rates      FRED CSV downloads (no key)
  headlines  Finnhub (free key in the FINNHUB_API_KEY environment variable;
             skipped if it is missing)

Usage
  python daily_data.py                       # latest US session
  python daily_data.py --date 2026-10-30     # one past session
  python daily_data.py --backfill 20         # the last 20 sessions
  python daily_data.py --prices-csv test.csv --no-network   # offline test
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from attribution import SIGMA_WINDOW, attribute, _naive

HERE = Path(__file__).resolve().parent
MARKET_CALENDAR = "SPY"          # a US session = a day SPY traded
RATE_SYMBOLS = {"^TNX", "^FVX", "^TYX", "^IRX", "2YY=F"}   # quoted as a yield in %

BACKDROP = [("S&P 500 (SPY)", "SPY"), ("Nasdaq 100 (QQQ)", "QQQ"), ("VIX", "^VIX"),
            ("US 10Y yield", "^TNX"), ("US Dollar Index", "DX-Y.NYB"),
            ("Brent", "BZ=F"), ("WTI", "CL=F"), ("Gold", "GC=F"), ("Bitcoin", "BTC-USD")]
SECTOR_ETFS = {"XLK": "Technology", "XLC": "Communication services",
               "XLY": "Consumer discretionary", "XLP": "Consumer staples",
               "XLF": "Financials", "XLV": "Health care", "XLI": "Industrials",
               "XLE": "Energy", "XLB": "Materials", "XLRE": "Real estate",
               "XLU": "Utilities"}
FRED_SERIES = {"DGS2": "US 2Y Treasury yield, %", "DGS10": "US 10Y Treasury yield, %",
               "DFII10": "US 10Y real (TIPS) yield, %",
               "DFEDTARU": "Fed funds target, upper bound, %",
               "DFEDTARL": "Fed funds target, lower bound, %"}
FINNHUB = "https://finnhub.io/api/v1"


# ------------------------------------------------------------- helpers ---
def clean(x):
    """Make a value JSON-safe: NaN/inf -> None, numpy -> python, round floats."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_watchlist(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("ticker", "").strip() and not r["ticker"].startswith("#")]
    for r in rows:
        r["peers"] = (r.get("peers") or "").split()
        for k in ("symbol", "market", "sector", "class", "news"):
            r[k] = (r.get(k) or "").strip()
    return rows


def all_symbols(watch: list[dict]) -> list[str]:
    syms = [MARKET_CALENDAR] + [s for _, s in BACKDROP] + list(SECTOR_ETFS)
    for r in watch:
        syms += [r["symbol"], r["market"], r["sector"], *r["peers"]]
    return list(dict.fromkeys(s for s in syms if s and not s.startswith("fred:")))


def download(symbols: list[str], end: pd.Timestamp | None, attempts: int = 3) -> pd.DataFrame:
    import yfinance as yf
    start = (end or pd.Timestamp.today()) - pd.Timedelta(days=640)
    stop = end + pd.Timedelta(days=1) if end is not None else None
    last_err = None
    for i in range(attempts):
        try:
            raw = yf.download(symbols, start=start, end=stop, auto_adjust=True,
                              progress=False, threads=True)
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            closes.index = _naive(closes.index)
            if closes.dropna(how="all").empty:
                raise RuntimeError("Yahoo returned no data")
            return closes.sort_index()
        except Exception as e:  # network hiccups, rate limits
            last_err = e
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"price download failed after {attempts} tries: {last_err}")


def fix_rate_scale(prices: pd.DataFrame) -> None:
    """Some feeds quote ^TNX etc. as yield x 10; convert to plain % if so."""
    for s in RATE_SYMBOLS & set(prices.columns):
        col = prices[s].dropna()
        if len(col) and col.median() > 20:
            prices[s] = prices[s] / 10


def change(series: pd.Series, t: pd.Timestamp, kind: str):
    """Move on t vs the previous price in the asset's own series.
    kind 'pct' -> % change; 'bp' -> basis-point change of a yield."""
    s = series.dropna().loc[:t]
    if len(s) < 2 or s.index[-1] != t:
        return None
    if kind == "bp":
        diffs = s.diff() * 100
    else:
        diffs = s.pct_change() * 100
    move = float(diffs.iloc[-1])
    hist = diffs.iloc[1:-1].tail(SIGMA_WINDOW)
    sd = hist.std(ddof=1)
    sigma = move / sd if len(hist) >= 20 and sd and not np.isnan(sd) else None
    return {"close": float(s.iloc[-1]), "prev_date": str(s.index[-2].date()),
            "move": move, "sigma": sigma}


def unit_for(row_class: str, symbol: str) -> str:
    return "bp" if row_class == "rate" or symbol in RATE_SYMBOLS else "pct"


# ------------------------------------------------------- free web data ---
def fetch_fred(series_id: str, since: pd.Timestamp) -> pd.Series:
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": series_id, "cosd": since.strftime("%Y-%m-%d")},
                     timeout=30, headers={"User-Agent": "market-move-data"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]))


def fred_block(t: pd.Timestamp, errors: list) -> dict:
    out = {}
    for sid, label in FRED_SERIES.items():
        try:
            s = fetch_fred(sid, t - pd.Timedelta(days=40)).loc[:t]
            if len(s) < 2:
                raise RuntimeError("not enough observations")
            out[sid] = {"label": label, "date": str(s.index[-1].date()),
                        "value": float(s.iloc[-1]),
                        "change_bp": float((s.iloc[-1] - s.iloc[-2]) * 100),
                        "note": "FRED lags about one business day"}
        except Exception as e:
            errors.append(f"FRED {sid}: {e}")
    return out


def finnhub(path: str, params: dict, key: str):
    import requests
    for i in range(3):
        r = requests.get(f"{FINNHUB}/{path}", params={**params, "token": key}, timeout=20)
        if r.status_code == 429:          # free tier: 60 calls/minute
            time.sleep(20 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Finnhub rate limit")


def headlines(symbol: str, t: pd.Timestamp, key: str) -> list[dict]:
    items = finnhub("company-news", {"symbol": symbol,
                                     "from": (t - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                                     "to": (t + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}, key)
    items = sorted(items or [], key=lambda x: x.get("datetime", 0), reverse=True)[:10]
    return [{"time_utc": datetime.fromtimestamp(i["datetime"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "source": i.get("source"), "headline": i.get("headline"), "url": i.get("url")}
            for i in items if i.get("headline")]


def earnings(symbol: str, t: pd.Timestamp, key: str) -> list[dict]:
    data = finnhub("calendar/earnings", {"symbol": symbol,
                                         "from": (t - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                                         "to": (t + pd.Timedelta(days=30)).strftime("%Y-%m-%d")}, key)
    hour = {"bmo": "before market open", "amc": "after market close", "dmh": "during market hours"}
    return [{"date": e.get("date"), "time": hour.get(e.get("hour"), e.get("hour") or "unknown"),
             "eps_estimate": e.get("epsEstimate"), "eps_actual": e.get("epsActual")}
            for e in (data or {}).get("earningsCalendar", [])]


def market_news(t: pd.Timestamp, key: str) -> list[dict]:
    items = finnhub("news", {"category": "general"}, key) or []
    cutoff = (t - pd.Timedelta(days=1)).timestamp()
    items = [i for i in items if i.get("datetime", 0) >= cutoff][:15]
    return [{"time_utc": datetime.fromtimestamp(i["datetime"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
             "source": i.get("source"), "headline": i.get("headline"), "url": i.get("url")}
            for i in items]


# --------------------------------------------------------- one session ---
def build_session(prices: pd.DataFrame, watch: list[dict], t: pd.Timestamp,
                  network: bool, finnhub_key: str | None, is_latest: bool) -> dict:
    errors: list[str] = []
    out = {
        "schema": 1,
        "session_date": str(t.date()),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "sources": {"prices": "Yahoo Finance via yfinance (adjusted closes)",
                    "rates": "FRED, Federal Reserve Bank of St. Louis",
                    "headlines": "Finnhub" if finnhub_key and network else "not fetched"},
        "method": {"move": "close on session_date vs the asset's previous close",
                   "sigma_window_days": SIGMA_WINDOW,
                   "attribution": "two-factor OLS on the 252 common trading days before the "
                                  "session; NA if < 120 days or R^2 < 0.10",
                   "units": "move_pct in %, move_bp in basis points, components in "
                            "percentage points (pp)"},
        "backdrop": [], "sectors": [], "assets": [], "fred": {}, "market_news": [],
        "errors": errors,
    }

    for label, sym in BACKDROP:
        if sym not in prices:
            continue
        c = change(prices[sym], t, unit_for("", sym))
        if c:
            key = "move_bp" if unit_for("", sym) == "bp" else "move_pct"
            out["backdrop"].append({"name": label, "symbol": sym, "close": c["close"],
                                    key: c["move"]})

    for etf, name in SECTOR_ETFS.items():
        if etf in prices:
            c = change(prices[etf], t, "pct")
            if c:
                out["sectors"].append({"etf": etf, "sector": name, "move_pct": c["move"]})
    out["sectors"].sort(key=lambda r: r["move_pct"], reverse=True)

    if network:
        out["fred"] = fred_block(t, errors)

    for r in watch:
        a = {"ticker": r["ticker"], "name": r["name"], "class": r["class"],
             "symbol": r["symbol"], "notes": []}
        sym, cls = r["symbol"], r["class"]
        if sym.startswith("fred:"):
            sid = sym.split(":", 1)[1]
            f = out["fred"].get(sid)
            if f:
                a.update({"close": f["value"], "as_of": f["date"], "move_bp": f["change_bp"],
                          "sigma": None})
                a["notes"].append("Policy rate from FRED; changes only at FOMC decisions.")
            else:
                a["notes"].append("FRED data unavailable in this run.")
            out["assets"].append(a)
            continue
        if sym not in prices or prices[sym].dropna().empty:
            a["notes"].append("No price data from Yahoo for this symbol.")
            errors.append(f"{r['ticker']}: no data for {sym}")
            out["assets"].append(a)
            continue

        unit = unit_for(cls, sym)
        c = change(prices[sym], t, unit)
        if c is None:
            a["notes"].append(f"No price on {t.date()} (holiday or not yet updated).")
            out["assets"].append(a)
            continue
        a.update({"close": c["close"], "prev_date": c["prev_date"],
                  ("move_bp" if unit == "bp" else "move_pct"): c["move"],
                  "sigma": c["sigma"],
                  "above_2_sigma": (abs(c["sigma"]) >= 2) if c["sigma"] is not None else None})

        # market / sector / specific split
        if cls in ("stock", "etf", "commodity", "crypto") and r["market"]:
            if any(x not in prices for x in (r["market"], r["sector"]) if x):
                a["attribution"] = {"na_reason": "benchmark data missing"}
            else:
                at = attribute(prices, sym, r["market"], r["sector"] or None, t)
                a["attribution"] = {k: getattr(at, k) for k in (
                    "market", "sector", "move_pct", "interval_start", "market_pp",
                    "sector_pp", "specific_pp", "beta_market", "beta_sector", "r2",
                    "observations", "na_reason", "notes")}
                if at.move_pct is not None and abs(at.move_pct - c["move"]) > 0.05:
                    a["notes"].append(
                        f"Split measured from {at.interval_start} (last day all benchmarks "
                        f"traded), so its move ({at.move_pct:+.2f}%) differs from the headline move.")
        else:
            a["attribution"] = {"na_reason": "no split for rates, FX, policy or the market itself"}

        # peers
        peers = {}
        for p in r["peers"]:
            if p in prices:
                pc = change(prices[p], t, unit_for("", p))
                if pc:
                    peers[p] = {("move_bp" if unit_for("", p) == "bp" else "move_pct"): pc["move"]}
        a["peers"] = peers

        if cls == "commodity":
            a["notes"].append("Continuous front-month futures: check for a contract roll if "
                              "the move looks out of line with news.")
        if cls == "crypto":
            a["notes"].append("Crypto trades 24/7; the move is close-to-close on Yahoo's UTC "
                              "daily bars and the latest bar may be partial.")

        if r["news"] == "yes" and finnhub_key and network:
            try:
                a["headlines"] = headlines(sym, t, finnhub_key)
                a["earnings"] = earnings(sym, t, finnhub_key)
            except Exception as e:
                errors.append(f"Finnhub {sym}: {e}")
        out["assets"].append(a)

    if finnhub_key and network and is_latest:
        try:
            out["market_news"] = market_news(t, finnhub_key)
        except Exception as e:
            errors.append(f"Finnhub market news: {e}")
    return clean(out)


# ---------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="session date YYYY-MM-DD (default: latest)")
    ap.add_argument("--backfill", type=int, default=0, help="also build the N sessions before")
    ap.add_argument("--watchlist", default=str(HERE / "watchlist.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--prices-csv", help="offline test: read prices from a CSV instead of Yahoo")
    ap.add_argument("--no-network", action="store_true", help="skip FRED and Finnhub")
    args = ap.parse_args()

    watch = load_watchlist(Path(args.watchlist))
    syms = all_symbols(watch)
    end = pd.Timestamp(args.date) if args.date else None
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv, parse_dates=["date"]).set_index("date").sort_index()
    else:
        prices = download(syms, end)
    fix_rate_scale(prices)

    cal = prices[MARKET_CALENDAR].dropna()
    if end is not None:
        if end not in cal.index:
            sys.exit(f"{end.date()} was not a US trading session (no {MARKET_CALENDAR} price).")
        sessions = cal.loc[:end].index
    else:
        sessions = cal.index
    targets = list(sessions[-(args.backfill + 1):])
    latest = cal.index[-1]

    key = os.environ.get("FINNHUB_API_KEY") or None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        is_latest = (t == latest) and not args.date
        data = build_session(prices, watch, t, not args.no_network, key, is_latest)
        text = json.dumps(data, indent=1, ensure_ascii=False)
        (out_dir / f"{t.date()}.json").write_text(text)
        if t == latest and not args.date:      # a past --date never overwrites latest
            (out_dir / "latest.json").write_text(text)
        n_err = len(data["errors"])
        print(f"{t.date()}: {len(data['assets'])} assets written, {n_err} warnings")
        for e in data["errors"]:
            print(f"  warning: {e}")

    dates = sorted(p.stem for p in out_dir.glob("20??-??-??.json"))
    (out_dir / "index.json").write_text(json.dumps(
        {"latest": dates[-1] if dates else None, "sessions": dates}, indent=1))


if __name__ == "__main__":
    main()
