"""End-to-end analysis runner.

Loads ingested data, runs lead-lag analysis across all macro events,
generates all figures and tables, and prints a summary.

Usage:
    python -m src.analysis.run_analysis [--events CPI FOMC NFP] [--equity SPY]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

# Default equity tickers to test lead-lag against
DEFAULT_EQUITIES = ["SPY", "QQQ", "TLT", "GLD", "XLF"]

# Default event types to analyse
DEFAULT_EVENTS = ["CPI", "NFP", "FOMC", "GDP", "PCE", "PMI"]


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_macro_events(
    db_path: Path | None = None,
    event_types: list[str] | None = None,
    impact: str | None = None,
) -> pd.DataFrame:
    """Load macro events from the DuckDB calendar."""
    db = db_path or DATA_DIR / "ticks.duckdb"
    con = duckdb.connect(str(db), read_only=True)

    conditions = ["country = 'US'"]
    params: list = []
    if impact:
        conditions.append("impact = ?")
        params.append(impact)
    if event_types:
        placeholders = ", ".join(["?"] * len(event_types))
        conditions.append(f"type IN ({placeholders})")
        params.extend(event_types)

    where = " AND ".join(conditions)
    df = con.execute(f"SELECT * FROM macro_events WHERE {where} ORDER BY date", params).fetchdf()
    con.close()

    df["date"] = pd.to_datetime(df["date"], utc=True)
    logger.info("Loaded %d macro events", len(df))
    return df


def load_equity_bars(ticker: str) -> pd.DataFrame:
    """Load 1-min bars for an equity ticker from Parquet."""
    path = DATA_DIR / "eodhd" / "bars_1m" / f"{ticker}__US.parquet"
    if not path.exists():
        logger.warning("No bars found for %s at %s", ticker, path)
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_pm_bars(slug: str) -> pd.DataFrame:
    """Load 1-min bars for a prediction market slug from Parquet."""
    bars_dir = DATA_DIR / "telonex" / "bars_1m"
    # Try both yes and no outcomes
    for suffix in ["__yes.parquet", "__no.parquet", "__Yes.parquet", "__No.parquet"]:
        path = bars_dir / f"{slug}{suffix}"
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            return df

    logger.warning("No bars found for slug %s", slug)
    return pd.DataFrame()


def find_pm_slugs_for_event_type(event_type: str) -> list[str]:
    """Find prediction market slugs that match a given event type.

    Scans the telonex catalog for markets related to the event type.
    """
    catalog_path = DATA_DIR / "telonex" / "telonex_markets.parquet"
    if not catalog_path.exists():
        return []

    catalog = pd.read_parquet(catalog_path)
    search_text = (
        catalog["event_title"].fillna("").astype(str) + " "
        + catalog["question"].fillna("").astype(str) + " "
        + catalog["slug"].fillna("").astype(str)
    ).str.lower()

    # Map event types to search keywords
    event_keywords = {
        "CPI": ["cpi ", "consumer price index", "inflation us"],
        "NFP": ["nonfarm payroll", "jobs report", "unemployment rate"],
        "FOMC": ["fomc", "fed rate", "fed decision", "fed meeting", "fed cut", "fed hike",
                 "federal reserve", "powell"],
        "GDP": ["gdp ", "gross domestic"],
        "PCE": ["pce ", "core pce"],
        "PMI": ["pmi "],
        "RETAIL_SALES": ["retail sales"],
        "RATE_DECISION": ["interest rate", "rate decision"],
    }

    keywords = event_keywords.get(event_type, [event_type.lower()])
    mask = pd.Series(False, index=catalog.index)
    for kw in keywords:
        mask |= search_text.str.contains(kw, regex=False, na=False)

    slugs = catalog.loc[mask, "slug"].dropna().unique().tolist()
    logger.info("Found %d PM slugs for event type %s", len(slugs), event_type)
    return slugs


# ---------------------------------------------------------------------------
# Event window extraction
# ---------------------------------------------------------------------------

def extract_event_windows(
    events: pd.DataFrame,
    eq_bars: pd.DataFrame,
    pm_bars: pd.DataFrame,
    before: timedelta = timedelta(hours=72),
    after: timedelta = timedelta(hours=72),
    min_obs: int = 30,
) -> list[dict]:
    """Extract aligned (pm, equity) windows around each event.

    Returns list of dicts with keys:
        event_date, event_type, pm_close, eq_close, pm_returns, eq_returns
    """
    windows = []
    for _, event in events.iterrows():
        t = event["date"]
        w_start = t - before
        w_end = t + after

        pm_w = pm_bars.loc[w_start:w_end, "close"] if "close" in pm_bars.columns else pd.Series()
        eq_w = eq_bars.loc[w_start:w_end, "close"] if "close" in eq_bars.columns else pd.Series()

        # Align on shared index
        aligned = pd.DataFrame({"pm": pm_w, "eq": eq_w}).dropna()
        if len(aligned) < min_obs:
            continue

        windows.append({
            "event_date": t,
            "event_type": event.get("type", ""),
            "pm_close": aligned["pm"],
            "eq_close": aligned["eq"],
            "pm_returns": aligned["pm"].diff().dropna(),
            "eq_returns": aligned["eq"].pct_change().dropna(),
        })

    return windows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    event_types: list[str] | None = None,
    equity_tickers: list[str] | None = None,
    before_hours: float = 72.0,
    after_hours: float = 72.0,
) -> pd.DataFrame:
    """Run the full lead-lag analysis pipeline.

    For each (event_type, equity_ticker, pm_slug) combination:
    1. Load and align data
    2. Compute signed area, cross-correlation, Granger causality
    3. Collect results into a DataFrame
    4. Generate all figures and tables

    Returns the results DataFrame.
    """
    from src.analysis.lead_lag import analyse_event_window
    from src.analysis.visualize import (
        plot_signed_area_distribution,
        plot_signed_area_by_event,
        plot_cross_correlation,
        build_lead_lag_table,
        save_lead_lag_table,
    )

    event_types = event_types or DEFAULT_EVENTS
    equity_tickers = equity_tickers or DEFAULT_EQUITIES
    before = timedelta(hours=before_hours)
    after = timedelta(hours=after_hours)

    # Load events
    events = load_macro_events(event_types=event_types)
    if events.empty:
        logger.error("No macro events found. Run the calendar pipeline first.")
        return pd.DataFrame()

    all_results = []

    for event_type in event_types:
        type_events = events[events["type"] == event_type]
        if type_events.empty:
            logger.info("No events for type %s, skipping", event_type)
            continue

        pm_slugs = find_pm_slugs_for_event_type(event_type)
        if not pm_slugs:
            logger.info("No PM slugs for %s, skipping", event_type)
            continue

        # Use the first available PM slug with data
        pm_bars = pd.DataFrame()
        pm_slug_used = ""
        for slug in pm_slugs[:10]:  # try up to 10
            pm_bars = load_pm_bars(slug)
            if not pm_bars.empty:
                pm_slug_used = slug
                break

        if pm_bars.empty:
            logger.info("No PM bar data for %s, skipping", event_type)
            continue

        for eq_ticker in equity_tickers:
            eq_bars = load_equity_bars(eq_ticker)
            if eq_bars.empty:
                continue

            windows = extract_event_windows(
                type_events, eq_bars, pm_bars, before=before, after=after,
            )

            if not windows:
                logger.debug("No valid windows for %s / %s / %s", event_type, eq_ticker, pm_slug_used)
                continue

            logger.info(
                "Analysing %d windows: %s / %s / %s",
                len(windows), event_type, eq_ticker, pm_slug_used,
            )

            for w in tqdm(windows, desc=f"{event_type}/{eq_ticker}", unit="evt", leave=False):
                result = analyse_event_window(w["pm_close"], w["eq_close"])
                result["event_type"] = event_type
                result["event_date"] = w["event_date"]
                result["equity_ticker"] = eq_ticker
                result["pm_slug"] = pm_slug_used
                all_results.append(result)

    if not all_results:
        logger.error("No results produced. Check that data is ingested.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Drop the signature array column for the summary table
    table_cols = [c for c in results_df.columns if c != "signature"]
    results_csv = results_df[table_cols]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    logger.info("Saved raw results: %s", RESULTS_DIR / "all_results.csv")

    # Generate figures
    logger.info("Generating figures ...")

    plot_signed_area_distribution(results_df)
    plot_signed_area_by_event(results_df)

    # Build and save summary table
    summary = build_lead_lag_table(results_df)
    save_lead_lag_table(summary)
    logger.info("\n%s", summary.to_string())

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lead-lag analysis on ingested data",
    )
    parser.add_argument("--events", nargs="+", default=None,
                       help="Event types to analyse (default: CPI NFP FOMC GDP PCE PMI)")
    parser.add_argument("--equity", nargs="+", default=None,
                       help="Equity tickers (default: SPY QQQ TLT GLD XLF)")
    parser.add_argument("--before", type=float, default=72.0,
                       help="Hours before event (default: 72)")
    parser.add_argument("--after", type=float, default=72.0,
                       help="Hours after event (default: 72)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    run_analysis(
        event_types=args.events,
        equity_tickers=args.equity,
        before_hours=args.before,
        after_hours=args.after,
    )


if __name__ == "__main__":
    main()
