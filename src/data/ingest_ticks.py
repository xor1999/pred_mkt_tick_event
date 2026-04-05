"""Tick-by-tick data ingestion pipeline backed by DuckDB.

Primary source:
  EODHD Tick Data API -- US equities (ms timestamps, trade metadata).

Kalshi support is included but disabled by default (US-only API access).
Pass ``--source kalshi`` or ``--source both`` to enable if you have access.

Two modes:
  - **Continuous** (``run_pipeline``): fetch ticks for the full lookback window.
  - **Event-window** (``run_event_window_pipeline``): fetch ticks only
    around macro events (e.g. T-72h to T+72h around CPI).

Raw ticks are stored in a single DuckDB database:
  - ``eodhd_ticks``  (ticker, ts, price, shares, mkt, sub_mkt, sl, seq)
  - ``kalshi_ticks``  (ticker, ts, price, count, trade_id) -- only if Kalshi enabled
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

from src.data.eodhd_client import EODHDClient, BudgetExhausted

logger = logging.getLogger(__name__)

DEFAULT_YEARS = 2

# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------

EODHD_DDL = """\
CREATE TABLE IF NOT EXISTS eodhd_ticks (
    ticker   VARCHAR NOT NULL,
    ts       BIGINT  NOT NULL,   -- ms since epoch
    price    DOUBLE  NOT NULL,
    shares   BIGINT  NOT NULL,
    mkt      VARCHAR,            -- exchange code
    sub_mkt  VARCHAR,
    sl       VARCHAR,            -- sales condition
    seq      BIGINT,
    PRIMARY KEY (ticker, ts, seq)
);
"""

KALSHI_DDL = """\
CREATE TABLE IF NOT EXISTS kalshi_ticks (
    ticker   VARCHAR NOT NULL,
    ts       BIGINT  NOT NULL,   -- ms since epoch
    price    DOUBLE  NOT NULL,   -- probability 0-1
    count    BIGINT  NOT NULL,   -- contracts traded
    trade_id VARCHAR,
    PRIMARY KEY (ticker, ts, trade_id)
);
"""


def _init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open (or create) the tick database and ensure tables exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(EODHD_DDL)
    con.execute(KALSHI_DDL)
    return con


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _max_ts(con: duckdb.DuckDBPyConnection, table: str, ticker: str) -> int | None:
    row = con.execute(
        f"SELECT MAX(ts) FROM {table} WHERE ticker = ?", [ticker],
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    return None


def _monthly_chunks(
    start: datetime, end: datetime, chunk_days: int = 30,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((int(cursor.timestamp()), int(chunk_end.timestamp())))
        cursor = chunk_end
    return chunks


def _quarter_chunks(
    start: datetime, end: datetime,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=90), end)
        chunks.append((int(cursor.timestamp()), int(chunk_end.timestamp())))
        cursor = chunk_end
    return chunks


# ---------------------------------------------------------------------------
# EODHD ticks
# ---------------------------------------------------------------------------

def _normalise_eodhd_ticks(ticks: list[dict], ticker: str) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame(columns=["ticker", "ts", "price", "shares", "mkt", "sub_mkt", "sl", "seq"])
    df = pd.DataFrame(ticks)
    df["ticker"] = ticker
    df["ts"] = df["ts"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["shares"] = df["shares"].astype("int64")
    for col in ("mkt", "sub_mkt", "sl"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""
    if "seq" in df.columns:
        df["seq"] = df["seq"].astype("int64")
    else:
        df["seq"] = 0
    return df[["ticker", "ts", "price", "shares", "mkt", "sub_mkt", "sl", "seq"]]


def ingest_eodhd_ticker(
    client: EODHDClient,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start: datetime,
    end: datetime,
) -> int:
    """Fetch raw ticks for a single US ticker and INSERT into DuckDB.

    Incremental: queries max(ts) in DB, only fetches from there onward.
    """
    sym = ticker.split(".")[0]

    max_ts_ms = _max_ts(con, "eodhd_ticks", sym)
    if max_ts_ms is not None:
        resume_s = (max_ts_ms // 1000) + 1
        start = max(start, datetime.fromtimestamp(resume_s, tz=timezone.utc))

    if start >= end:
        return 0

    chunks = _monthly_chunks(start, end)
    total_inserted = 0

    for from_ts, to_ts in chunks:
        try:
            raw = client.get_ticks(ticker, from_ts, to_ts)
            df = _normalise_eodhd_ticks(raw, sym)
            if df.empty:
                continue
            con.execute(
                "INSERT OR IGNORE INTO eodhd_ticks SELECT * FROM df",
            )
            total_inserted += len(df)
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("  %s chunk %d-%d failed: %s", sym, from_ts, to_ts, exc)

    return total_inserted


# ---------------------------------------------------------------------------
# Kalshi ticks
# ---------------------------------------------------------------------------

def _normalise_kalshi_trades(trades: list[dict], ticker: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["ticker", "ts", "price", "count", "trade_id"])

    df = pd.DataFrame(trades)

    if "created_time" in df.columns:
        df["ts"] = (
            pd.to_datetime(df["created_time"], utc=True)
            .astype("int64") // 10**6
        )
    elif "ts" in df.columns:
        df["ts"] = (df["ts"].astype("int64") * 1000)
    else:
        raise KeyError(f"No recognised timestamp column. Columns: {list(df.columns)}")

    price_col = "yes_price" if "yes_price" in df.columns else "price"
    df["price"] = df[price_col].astype(float) / 100.0

    count_col = "count" if "count" in df.columns else "volume"
    df["count"] = df[count_col].astype("int64") if count_col in df.columns else 1

    if "trade_id" not in df.columns:
        id_col = next((c for c in ("id", "trade_id") if c in df.columns), None)
        df["trade_id"] = df[id_col].astype(str) if id_col else ""

    df["ticker"] = ticker
    return df[["ticker", "ts", "price", "count", "trade_id"]]


def ingest_kalshi_contract(
    client: KalshiClient,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start: datetime,
    end: datetime,
) -> int:
    """Fetch raw trades for a Kalshi contract and INSERT into DuckDB."""
    max_ts_ms = _max_ts(con, "kalshi_ticks", ticker)
    if max_ts_ms is not None:
        resume_s = (max_ts_ms // 1000) + 1
        start = max(start, datetime.fromtimestamp(resume_s, tz=timezone.utc))

    if start >= end:
        return 0

    chunks = _quarter_chunks(start, end)
    total_inserted = 0

    for from_ts, to_ts in chunks:
        try:
            trades = client.get_all_trades(ticker, min_ts=from_ts, max_ts=to_ts)
            df = _normalise_kalshi_trades(trades, ticker)
            if df.empty:
                continue
            con.execute(
                "INSERT OR IGNORE INTO kalshi_ticks SELECT * FROM df",
            )
            total_inserted += len(df)
        except Exception as exc:
            logger.warning("  %s chunk %d-%d failed: %s", ticker, from_ts, to_ts, exc)

    return total_inserted


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

def _resolve_eodhd_tickers(
    mode: str,
    data_dir: Path,
    custom_tickers: list[str] | None,
    client: EODHDClient,
    skip_universe: bool,
) -> list[str] | None:
    from src.data.watchlists import get_watchlist, estimate_cost

    if mode == "custom":
        if not custom_tickers:
            logger.error("mode='custom' requires --ticker flags.")
            return None
        tickers = get_watchlist("custom", custom_tickers)
    elif mode == "macro":
        tickers = get_watchlist("macro")
    elif mode == "full":
        universe_path = data_dir / "eodhd" / "universe.parquet"
        if not universe_path.exists():
            if skip_universe:
                logger.error("--skip-universe set but %s not found.", universe_path)
                return None
            logger.info("No existing universe -- building one ...")
            from src.data.ingest_eodhd import build_universe
            eodhd_dir = data_dir / "eodhd"
            eodhd_dir.mkdir(parents=True, exist_ok=True)
            universe = build_universe(client, eodhd_dir, include_etfs=True)
            if universe.empty:
                return None
        else:
            universe = pd.read_parquet(universe_path)
            logger.info("Reusing existing universe (%d tickers)", len(universe))
        code_col = "Code" if "Code" in universe.columns else universe.columns[0]
        tickers = universe[code_col].tolist()
    else:
        logger.error("Unknown mode: %s", mode)
        return None

    est = estimate_cost(len(tickers))
    logger.info(
        "Mode '%s': %d tickers | ~%d API calls | ~%d credits | ~%.0f min | ~%.1f GB disk",
        mode, est["tickers"], est["api_calls"], est["credits"],
        est["est_runtime_min"], est["est_disk_gb"],
    )
    return tickers


# ---------------------------------------------------------------------------
# Full pipeline (continuous)
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: Path,
    years: int = DEFAULT_YEARS,
    source: str = "eodhd",
    mode: str = "macro",
    eodhd_tickers: list[str] | None = None,
    kalshi_tickers: list[str] | None = None,
    budget: int | None = None,
    skip_universe: bool = False,
    skip_catalog: bool = False,
) -> None:
    """Run the tick-by-tick ingestion pipeline.

    Default source is "eodhd" only.  Pass source="kalshi" or "both" to
    enable Kalshi (requires US-based API access).
    """
    data_dir = Path(data_dir)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)
    logger.info(
        "Tick ingestion window: %s -- %s (%d years)",
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), years,
    )

    db_path = data_dir / "ticks.duckdb"
    con = _init_db(db_path)
    logger.info("DuckDB: %s", db_path)

    try:
        if source in ("eodhd", "both"):
            _run_eodhd_ticks(con, data_dir, start, end, mode, eodhd_tickers, budget, skip_universe)

        if source in ("kalshi", "both"):
            _run_kalshi_ticks(con, data_dir, start, end, kalshi_tickers, skip_catalog)
    finally:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM eodhd_ticks").fetchone()
        logger.info("  eodhd_ticks: %s ticks across %s tickers", f"{row[0]:,}", row[1])
        if source in ("kalshi", "both"):
            row = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM kalshi_ticks").fetchone()
            logger.info("  kalshi_ticks: %s ticks across %s tickers", f"{row[0]:,}", row[1])
        con.close()


def _run_eodhd_ticks(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    start: datetime,
    end: datetime,
    mode: str,
    tickers: list[str] | None,
    budget: int | None,
    skip_universe: bool,
) -> None:
    client = EODHDClient(budget=budget)

    effective_mode = "custom" if tickers else mode
    ticker_list = _resolve_eodhd_tickers(effective_mode, data_dir, tickers, client, skip_universe)
    if not ticker_list:
        return

    n_tickers = len(ticker_list)
    n_chunks = len(_monthly_chunks(start, end))
    logger.info(
        "EODHD ticks: %d tickers x %d chunks (~30d each) = %d API calls (10 credits each)",
        n_tickers, n_chunks, n_tickers * n_chunks,
    )

    stats = {"ok": 0, "empty": 0, "error": 0, "total_ticks": 0}
    try:
        with tqdm(total=n_tickers, desc="EODHD ticks", unit="tkr") as pbar:
            for ticker in ticker_list:
                pbar.set_postfix_str(ticker)
                try:
                    n = ingest_eodhd_ticker(client, con, ticker, start, end)
                    if n > 0:
                        stats["ok"] += 1
                        stats["total_ticks"] += n
                    else:
                        stats["empty"] += 1
                except BudgetExhausted:
                    logger.warning("Budget exhausted -- stopping. Re-run to continue.")
                    break
                except Exception as exc:
                    stats["error"] += 1
                    logger.error("  %s failed: %s", ticker, exc)
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted -- progress saved. Re-run to continue.")

    logger.info("=" * 60)
    logger.info("EODHD TICK INGESTION COMPLETE")
    logger.info("  Tickers OK:       %6d", stats["ok"])
    logger.info("  Tickers empty:    %6d", stats["empty"])
    logger.info("  Tickers error:    %6d", stats["error"])
    logger.info("  New ticks:        %6d", stats["total_ticks"])
    logger.info("  API calls made:   %6d", client.calls_made)
    logger.info("=" * 60)


def _run_kalshi_ticks(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    start: datetime,
    end: datetime,
    tickers: list[str] | None,
    skip_catalog: bool,
) -> None:
    from src.data.kalshi_client import KalshiClient
    from src.data.ingest_kalshi import fetch_market_catalog

    logger.warning("Kalshi API is restricted to US-based users.")
    client = KalshiClient()

    if tickers:
        contract_tickers = tickers
    else:
        markets_path = data_dir / "kalshi" / "kalshi_markets.parquet"
        if not markets_path.exists():
            if skip_catalog:
                logger.error("--skip-catalog set but %s not found.", markets_path)
                return
            logger.info("No existing Kalshi catalog -- fetching ...")
            kalshi_dir = data_dir / "kalshi"
            kalshi_dir.mkdir(parents=True, exist_ok=True)
            markets = fetch_market_catalog(client, kalshi_dir, macro_only=True)
            if markets.empty:
                return
        else:
            markets = pd.read_parquet(markets_path)
            logger.info("Reusing existing Kalshi catalog (%d contracts)", len(markets))

        ticker_col = "ticker" if "ticker" in markets.columns else markets.columns[0]
        contract_tickers = markets[ticker_col].tolist()

    logger.info("Kalshi ticks: %d contracts", len(contract_tickers))

    stats = {"ok": 0, "empty": 0, "error": 0, "total_ticks": 0}
    with tqdm(total=len(contract_tickers), desc="Kalshi ticks", unit="ctr") as pbar:
        for ticker in contract_tickers:
            pbar.set_postfix_str(ticker)
            try:
                n = ingest_kalshi_contract(client, con, ticker, start, end)
                if n > 0:
                    stats["ok"] += 1
                    stats["total_ticks"] += n
                else:
                    stats["empty"] += 1
            except Exception as exc:
                stats["error"] += 1
                logger.error("  %s failed: %s", ticker, exc)
            pbar.update(1)

    logger.info("=" * 60)
    logger.info("KALSHI TICK INGESTION COMPLETE")
    logger.info("  Contracts OK:     %6d", stats["ok"])
    logger.info("  Contracts empty:  %6d", stats["empty"])
    logger.info("  Contracts error:  %6d", stats["error"])
    logger.info("  New ticks:        %6d", stats["total_ticks"])
    logger.info("  API calls made:   %6d", client.calls_made)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Event-window pipeline
# ---------------------------------------------------------------------------

def _ingest_eodhd_windows(
    client: EODHDClient,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    windows: list[tuple[int, int]],
) -> int:
    sym = ticker.split(".")[0]
    total = 0
    for from_ts, to_ts in windows:
        try:
            raw = client.get_ticks(ticker, from_ts, to_ts)
            df = _normalise_eodhd_ticks(raw, sym)
            if df.empty:
                continue
            con.execute("INSERT OR IGNORE INTO eodhd_ticks SELECT * FROM df")
            total += len(df)
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("  %s window %d-%d failed: %s", sym, from_ts, to_ts, exc)
    return total


def _ingest_kalshi_windows(
    client: KalshiClient,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    windows: list[tuple[int, int]],
) -> int:
    total = 0
    for from_ts, to_ts in windows:
        try:
            trades = client.get_all_trades(ticker, min_ts=from_ts, max_ts=to_ts)
            df = _normalise_kalshi_trades(trades, ticker)
            if df.empty:
                continue
            con.execute("INSERT OR IGNORE INTO kalshi_ticks SELECT * FROM df")
            total += len(df)
        except Exception as exc:
            logger.warning("  %s window %d-%d failed: %s", ticker, from_ts, to_ts, exc)
    return total


def run_event_window_pipeline(
    data_dir: Path,
    years: int = DEFAULT_YEARS,
    source: str = "eodhd",
    mode: str = "macro",
    eodhd_tickers: list[str] | None = None,
    kalshi_tickers: list[str] | None = None,
    budget: int | None = None,
    skip_universe: bool = False,
    skip_catalog: bool = False,
    categories: list[str] | None = None,
    before_hours: float = 72.0,
    after_hours: float = 72.0,
) -> None:
    """Fetch ticks ONLY around macro events from the calendar.

    Dramatically cheaper than continuous ingestion.
    """
    from src.data.macro_calendar import (
        init_calendar_table,
        fetch_and_store_events,
        get_merged_windows,
        calendar_summary,
    )

    data_dir = Path(data_dir)
    db_path = data_dir / "ticks.duckdb"
    con = _init_db(db_path)
    init_calendar_table(con)
    logger.info("DuckDB: %s", db_path)

    # Step 1 -- ensure macro calendar is populated
    row = con.execute("SELECT COUNT(*) FROM macro_events WHERE country = 'US'").fetchone()
    if row[0] == 0:
        logger.info("Macro calendar empty -- fetching events from EODHD ...")
        cal_client = EODHDClient(budget=budget)
        fetch_and_store_events(cal_client, con, country="US", years=years)
        summary = calendar_summary(con)
        if not summary.empty:
            logger.info("Calendar summary:")
            for _, r in summary.head(10).iterrows():
                logger.info("  %-25s  %s  %4d events", r["type"], r["impact"], r["count"])

    # Step 2 -- compute merged event windows
    before = timedelta(hours=before_hours)
    after = timedelta(hours=after_hours)
    windows = get_merged_windows(
        con, country="US", impact="High",
        categories=categories, before=before, after=after,
    )

    if not windows:
        logger.error("No event windows found -- check calendar. Aborting.")
        con.close()
        return

    total_hours = sum((e - s) for s, e in windows) / 3600
    logger.info(
        "Event windows: %d windows covering %.0f hours (%.1f days)",
        len(windows), total_hours, total_hours / 24,
    )

    # Step 3 -- fetch ticks within windows
    try:
        if source in ("eodhd", "both"):
            _run_eodhd_event_ticks(con, data_dir, windows, mode, eodhd_tickers, budget, skip_universe)

        if source in ("kalshi", "both"):
            _run_kalshi_event_ticks(con, data_dir, windows, kalshi_tickers, skip_catalog)
    finally:
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM eodhd_ticks").fetchone()
        logger.info("  eodhd_ticks: %s ticks across %s tickers", f"{row[0]:,}", row[1])
        if source in ("kalshi", "both"):
            row = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM kalshi_ticks").fetchone()
            logger.info("  kalshi_ticks: %s ticks across %s tickers", f"{row[0]:,}", row[1])
        con.close()


def _run_eodhd_event_ticks(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    windows: list[tuple[int, int]],
    mode: str,
    tickers: list[str] | None,
    budget: int | None,
    skip_universe: bool,
) -> None:
    client = EODHDClient(budget=budget)

    effective_mode = "custom" if tickers else mode
    ticker_list = _resolve_eodhd_tickers(effective_mode, data_dir, tickers, client, skip_universe)
    if not ticker_list:
        return

    n_tickers = len(ticker_list)
    logger.info(
        "EODHD event ticks: %d tickers x %d windows -> %d API calls (10 credits each)",
        n_tickers, len(windows), n_tickers * len(windows),
    )

    stats = {"ok": 0, "empty": 0, "error": 0, "total_ticks": 0}
    try:
        with tqdm(total=n_tickers, desc="EODHD event ticks", unit="tkr") as pbar:
            for ticker in ticker_list:
                pbar.set_postfix_str(ticker)
                try:
                    n = _ingest_eodhd_windows(client, con, ticker, windows)
                    if n > 0:
                        stats["ok"] += 1
                        stats["total_ticks"] += n
                    else:
                        stats["empty"] += 1
                except BudgetExhausted:
                    logger.warning("Budget exhausted -- stopping. Re-run to continue.")
                    break
                except Exception as exc:
                    stats["error"] += 1
                    logger.error("  %s failed: %s", ticker, exc)
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted -- progress saved. Re-run to continue.")

    logger.info("=" * 60)
    logger.info("EODHD EVENT-WINDOW TICKS COMPLETE")
    logger.info("  Tickers OK:       %6d", stats["ok"])
    logger.info("  Tickers empty:    %6d", stats["empty"])
    logger.info("  Tickers error:    %6d", stats["error"])
    logger.info("  New ticks:        %6d", stats["total_ticks"])
    logger.info("  API calls made:   %6d", client.calls_made)
    logger.info("=" * 60)


def _run_kalshi_event_ticks(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    windows: list[tuple[int, int]],
    tickers: list[str] | None,
    skip_catalog: bool,
) -> None:
    from src.data.kalshi_client import KalshiClient
    from src.data.ingest_kalshi import fetch_market_catalog

    logger.warning("Kalshi API is restricted to US-based users.")
    client = KalshiClient()

    if tickers:
        contract_tickers = tickers
    else:
        markets_path = data_dir / "kalshi" / "kalshi_markets.parquet"
        if not markets_path.exists():
            if skip_catalog:
                logger.error("--skip-catalog set but %s not found.", markets_path)
                return
            kalshi_dir = data_dir / "kalshi"
            kalshi_dir.mkdir(parents=True, exist_ok=True)
            markets = fetch_market_catalog(client, kalshi_dir, macro_only=True)
            if markets.empty:
                return
        else:
            markets = pd.read_parquet(markets_path)
            logger.info("Reusing existing Kalshi catalog (%d contracts)", len(markets))
        ticker_col = "ticker" if "ticker" in markets.columns else markets.columns[0]
        contract_tickers = markets[ticker_col].tolist()

    logger.info(
        "Kalshi event ticks: %d contracts x %d windows",
        len(contract_tickers), len(windows),
    )

    stats = {"ok": 0, "empty": 0, "error": 0, "total_ticks": 0}
    with tqdm(total=len(contract_tickers), desc="Kalshi event ticks", unit="ctr") as pbar:
        for ticker in contract_tickers:
            pbar.set_postfix_str(ticker)
            try:
                n = _ingest_kalshi_windows(client, con, ticker, windows)
                if n > 0:
                    stats["ok"] += 1
                    stats["total_ticks"] += n
                else:
                    stats["empty"] += 1
            except Exception as exc:
                stats["error"] += 1
                logger.error("  %s failed: %s", ticker, exc)
            pbar.update(1)

    logger.info("=" * 60)
    logger.info("KALSHI EVENT-WINDOW TICKS COMPLETE")
    logger.info("  Contracts OK:     %6d", stats["ok"])
    logger.info("  Contracts empty:  %6d", stats["empty"])
    logger.info("  Contracts error:  %6d", stats["error"])
    logger.info("  New ticks:        %6d", stats["total_ticks"])
    logger.info("  API calls made:   %6d", client.calls_made)
    logger.info("=" * 60)
