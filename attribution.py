#!/usr/bin/env python3
"""
Market-move attribution: how unusual was a move, and how much of it was
market, sector, or asset-specific?

Two commands:

  screen     Which assets moved more than their sigma trigger on a date?
  attribute  Split one asset's move into market + sector + specific.

Price data comes from either
  --source yfinance       (needs `pip install yfinance pandas numpy`; works on
                           your own computer and on GitHub Actions)
  --source prices.csv     (a CSV with a `date` column and one column of closing
                           prices per symbol)

Examples
  python attribution.py screen --symbols GOOG,META,MSFT,NBIS --date 2026-10-30
  python attribution.py attribute --symbol NBIS --market SPY --sector SMH
  python attribution.py attribute --symbol BTC-USD --market QQQ --json

Rules (from reference/attribution-method.md):
  - returns are measured over the SAME interval for the asset and its
    benchmarks: only dates on which all of them have a price are used
  - estimation window = up to 252 such days BEFORE the event day
  - sigma uses the 60 days before the event day
  - components are NA when < 120 observations or R^2 < 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

EST_WINDOW = 252
SIGMA_WINDOW = 60
MIN_OBS = 120
MIN_R2 = 0.10


# ---------------------------------------------------------------- data ----
def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    return idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx


def load_prices(symbols: list[str], source: str, end: str | None) -> pd.DataFrame:
    """Closing prices indexed by date, one column per symbol."""
    symbols = list(dict.fromkeys(s for s in symbols if s))
    if source == "yfinance":
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance is not installed. Run: pip install yfinance")
        stop = pd.Timestamp(end) + pd.Timedelta(days=1) if end else None
        start = (pd.Timestamp(end) if end else pd.Timestamp.today()) - pd.Timedelta(days=600)
        raw = yf.download(symbols, start=start, end=stop, auto_adjust=True,
                          progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": symbols[0]})
        closes.index = _naive(closes.index)
        empty = [s for s in symbols if s not in closes or closes[s].dropna().empty]
        if empty:
            sys.exit(f"No price data returned for: {', '.join(empty)}. Check the "
                     "ticker spelling and that this computer can reach Yahoo Finance.")
        return closes.sort_index()
    df = pd.read_csv(source)
    date_col = next(c for c in df.columns if c.lower() == "date")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    missing = [s for s in symbols if s not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns for: {', '.join(missing)}")
    return df[symbols]


# -------------------------------------------------------------- results ---
@dataclass
class Screen:
    symbol: str
    date: str
    move_pct: float | None
    sigma: float | None
    trigger: float
    mover: bool | None
    note: str = ""


@dataclass
class Attribution:
    symbol: str
    date: str
    market: str
    sector: str | None
    move_pct: float | None = None       # over the aligned interval (see notes)
    sigma: float | None = None
    market_pp: float | None = None
    sector_pp: float | None = None
    specific_pp: float | None = None
    beta_market: float | None = None
    beta_sector: float | None = None
    r2: float | None = None
    observations: int = 0
    interval_start: str | None = None   # previous date all series traded
    na_reason: str | None = None
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------- maths -----
def sigma_of(r: pd.Series, t: pd.Timestamp) -> float | None:
    """Return on t divided by the stdev of the SIGMA_WINDOW returns before t."""
    r = r.dropna()
    if t not in r.index:
        return None
    hist = r.loc[:t].iloc[:-1].tail(SIGMA_WINDOW)
    sd = hist.std(ddof=1)
    if len(hist) < 20 or not sd or np.isnan(sd):
        return None
    return float(r.loc[t] / sd)


def ols(y: np.ndarray, X: np.ndarray):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return coef, r2


def attribute(prices: pd.DataFrame, symbol: str, market: str,
              sector: str | None, date: str | pd.Timestamp | None) -> Attribution:
    """Two-factor split of `symbol`'s return on `date` (latest common date if None)."""
    cols = [symbol, market] + ([sector] if sector else [])
    px = prices[cols].dropna()                    # dates where ALL series traded
    t = pd.Timestamp(date) if date is not None else (px.index[-1] if len(px) else None)
    res = Attribution(symbol=symbol, date=str(t.date()) if t is not None else "",
                      market=market, sector=sector)
    if t is None or t not in px.index:
        res.na_reason = "the asset and its benchmarks did not all trade on this date"
        return res
    rets = px.pct_change().iloc[1:]
    pos = px.index.get_loc(t)
    res.interval_start = str(px.index[pos - 1].date()) if pos > 0 else None
    res.move_pct = round(float(rets.loc[t, symbol]) * 100, 2)
    sg = sigma_of(rets[symbol], t)
    res.sigma = round(sg, 2) if sg is not None else None

    window = rets.loc[:t].iloc[:-1].tail(EST_WINDOW)
    res.observations = len(window)
    if len(window) < MIN_OBS:
        res.na_reason = f"only {len(window)} days of overlapping history (< {MIN_OBS})"
        return res

    y = window[symbol].to_numpy()
    rm = window[market].to_numpy()
    if sector:
        rs = window[sector].to_numpy()
        (b0, b_sm), _ = ols(rs, rm.reshape(-1, 1))
        sector_excess = rs - (b0 + b_sm * rm)
        (_, bm, bs), r2 = ols(y, np.column_stack([rm, sector_excess]))
        sx_t = float(rets.loc[t, sector] - (b0 + b_sm * rets.loc[t, market]))
    else:
        (_, bm), r2 = ols(y, rm.reshape(-1, 1))
        bs, sx_t = 0.0, 0.0

    res.r2 = round(r2, 3)
    res.beta_market = round(float(bm), 2)
    res.beta_sector = round(float(bs), 2) if sector else None
    if r2 < MIN_R2:
        res.na_reason = f"R^2 {r2:.3f} is below {MIN_R2}: the benchmarks explain too little to split the move"
        return res

    move = float(rets.loc[t, symbol]) * 100
    mkt_pp = float(bm * rets.loc[t, market]) * 100
    sec_pp = float(bs * sx_t) * 100 if sector else None
    res.market_pp = round(mkt_pp, 1)
    res.sector_pp = round(sec_pp, 1) if sec_pp is not None else None
    res.specific_pp = round(move - mkt_pp - (sec_pp or 0.0), 1)
    if r2 < 0.25:
        res.notes.append("R^2 below 0.25: treat the split as rough")
    return res


def screen(prices: pd.DataFrame, symbols: list[str], date: str | None,
           trigger: float) -> list[Screen]:
    out = []
    for s in symbols:
        r = prices[s].dropna().pct_change().iloc[1:]
        t = pd.Timestamp(date) if date else (r.index[-1] if len(r) else None)
        if t is None or t not in r.index:
            out.append(Screen(s, str(date or ""), None, None, trigger, None,
                              "no price on this date"))
            continue
        sg = sigma_of(r, t)
        out.append(Screen(s, str(t.date()), round(float(r.loc[t]) * 100, 2),
                          round(sg, 2) if sg is not None else None, trigger,
                          (abs(sg) >= trigger) if sg is not None else None,
                          "" if sg is not None else "not enough history for sigma"))
    return out


# --------------------------------------------------------------- CLI -----
def fmt_pp(x):
    return "NA" if x is None else f"{x:+.1f} pp"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("screen", help="flag assets whose move exceeds the sigma trigger")
    sc.add_argument("--symbols", required=True, help="comma-separated, e.g. GOOG,META")
    sc.add_argument("--trigger", type=float, default=2.0)
    at = sub.add_parser("attribute", help="split one move into market/sector/specific")
    at.add_argument("--symbol", required=True)
    at.add_argument("--market", default="SPY")
    at.add_argument("--sector", default=None)
    for sp in (sc, at):
        sp.add_argument("--date", default=None, help="event date YYYY-MM-DD (default: latest)")
        sp.add_argument("--source", default="yfinance", help="'yfinance' or path to a CSV")
        sp.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = p.parse_args()

    if a.cmd == "screen":
        syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
        rows = screen(load_prices(syms, a.source, a.date), syms, a.date, a.trigger)
        if a.json:
            print(json.dumps([asdict(r) for r in rows], indent=2))
            return
        for r in rows:
            flag = "MOVER" if r.mover else ("  -  " if r.mover is False else "  ?  ")
            mv = "n/a" if r.move_pct is None else f"{r.move_pct:+.2f}%"
            sg = "n/a" if r.sigma is None else f"{r.sigma:+.2f}σ"
            print(f"{flag}  {r.symbol:<10} {r.date}  {mv:>8}  {sg:>7}  {r.note}")
        return

    syms = [a.symbol, a.market] + ([a.sector] if a.sector else [])
    r = attribute(load_prices(syms, a.source, a.date), a.symbol, a.market, a.sector, a.date)
    if a.json:
        print(json.dumps(asdict(r), indent=2))
        return
    mv = "n/a" if r.move_pct is None else f"{r.move_pct:+.2f}%"
    print(f"{r.symbol} on {r.date}: {mv}"
          + (f" ({r.sigma:+.2f}σ vs 60-day)" if r.sigma is not None else "")
          + (f", measured from {r.interval_start}" if r.interval_start else ""))
    print(f"  Market ({r.market}):            {fmt_pp(r.market_pp)}")
    if r.sector:
        print(f"  Sector ({r.sector} ex-market): {fmt_pp(r.sector_pp)}")
    print(f"  Specific:                {fmt_pp(r.specific_pp)}")
    if r.na_reason:
        print(f"  NA because: {r.na_reason}")
    if r.r2 is not None:
        print(f"  betas: market {r.beta_market}, sector {r.beta_sector}; "
              f"R^2 {r.r2}; {r.observations} days")
    for n in r.notes:
        print(f"  note: {n}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"Error: {e}")
