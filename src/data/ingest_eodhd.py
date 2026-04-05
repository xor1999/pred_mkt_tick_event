"""EODHD equities intraday ingestion pipeline.

Fetches 1-minute OHLCV bars for a filtered US stock universe over a
2-year lookback window.  Results are stored as one Parquet file per
ticker in ``{data_dir}/bars_1m/{TICKER}__US.parquet``.

Universe filtering:
  1. Common stocks only (no ETFs, funds, preferreds)
  2. Market cap > $1B (via EODHD screener)
  3. Liquidity: avg daily volume > 500k shares OR avg daily turnover > $10M
  4. Curated liquid ETFs appended regardless of screener

The EODHD ``/intraday`` endpoint returns at most ~120 calendar days of
1-min data per request, so we chunk the window into 100-day segments.
Ingestion is incremental: existing files are extended, not re-downloaded.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.eodhd_client import EODHDClient, BudgetExhausted

logger = logging.getLogger(__name__)

CHUNK_DAYS = 100
DEFAULT_YEARS = 2

MIN_AVG_VOLUME = 500_000
MIN_AVG_TURNOVER_USD = 10e6
MIN_MARKET_CAP = 1_000_000_000


# ---------------------------------------------------------------------------
# 1. Build filtered US universe
# ---------------------------------------------------------------------------

def _liquidity_filter(
    client: EODHDClient,
    symbols: pd.DataFrame,
) -> pd.DataFrame:
    """Filter symbols by liquidity using bulk last-day EOD data."""
    logger.info("  Fetching bulk EOD for liquidity filter ...")
    try:
        bulk = client.get_bulk_eod("US")
    except Exception as exc:
        logger.warning("  Bulk EOD failed (%s) -- skipping liquidity filter", exc)
        return symbols

    if bulk.empty:
        logger.warning("  Bulk EOD empty -- skipping liquidity filter")
        return symbols

    bulk.columns = bulk.columns.str.lower()
    code_col_bulk = "code" if "code" in bulk.columns else bulk.columns[0]

    if "volume" in bulk.columns and "close" in bulk.columns:
        bulk["turnover"] = bulk["close"].astype(float) * bulk["volume"].astype(float)
    else:
        bulk["turnover"] = 0

    liquid_mask = (
        (bulk["volume"].astype(float) >= MIN_AVG_VOLUME)
        | (bulk["turnover"] >= MIN_AVG_TURNOVER_USD)
    )
    liquid_codes = set(bulk.loc[liquid_mask, code_col_bulk].str.upper())

    code_col = "Code" if "Code" in symbols.columns else symbols.columns[0]
    before = len(symbols)
    filtered = pd.DataFrame(symbols[symbols[code_col].str.upper().isin(liquid_codes)])
    logger.info(
        "  Liquidity filter: %d -> %d tickers (vol>%dk or turnover>$%dM)",
        before, len(filtered),
        MIN_AVG_VOLUME // 1000, int(MIN_AVG_TURNOVER_USD / 1e6),
    )
    return filtered


def build_universe(
    client: EODHDClient,
    data_dir: Path,
    min_cap: float = MIN_MARKET_CAP,
    include_etfs: bool = True,
) -> pd.DataFrame:
    """Build a filtered US ticker universe and save as universe.parquet."""
    logger.info("Building US equity universe (include_etfs=%s) ...", include_etfs)

    # Step 1 -- all US symbols
    symbols = client.get_exchange_symbols("US")
    if symbols.empty:
        logger.error("No symbols returned for US exchange -- aborting.")
        return pd.DataFrame()
    logger.info("  Raw US symbols: %d", len(symbols))

    # Step 2 -- common stocks + optionally ETFs
    type_col = "Type" if "Type" in symbols.columns else None
    if type_col:
        allowed = {"common stock"}
        if include_etfs:
            allowed.add("etf")
        symbols = pd.DataFrame(symbols[symbols[type_col].str.lower().isin(allowed)])
    logger.info("  After type filter: %d", len(symbols))

    # Step 3 -- market cap > $1B via screener
    screened_codes: set[str] = set()
    offset = 0
    while offset <= 999:
        filters = [
            ["market_capitalization", ">", float(min_cap)],
            ["exchange", "=", "us"],
        ]
        try:
            batch = client.get_screener(filters, limit=100, offset=offset)
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("  Screener page offset=%d failed: %s", offset, exc)
            break
        if not batch:
            break
        for item in batch:
            code = item.get("code", item.get("Code", ""))
            screened_codes.add(str(code).upper())
        if len(batch) < 100:
            break
        offset += 100

    code_col = "Code" if "Code" in symbols.columns else symbols.columns[0]
    symbols = pd.DataFrame(symbols[symbols[code_col].str.upper().isin(screened_codes)])
    logger.info("  Market cap > $%.0fB: %d tickers", min_cap / 1e9, len(symbols))

    if symbols.empty:
        logger.error("No tickers passed market-cap screen -- aborting.")
        return pd.DataFrame()

    # Step 4 -- liquidity filter
    symbols = _liquidity_filter(client, symbols)

    if symbols.empty:
        logger.error("No tickers passed liquidity filter -- aborting.")
        return pd.DataFrame()

    # Step 5 -- ensure curated liquid ETFs are included
    if include_etfs:
        from src.data.watchlists import LIQUID_ETFS
        code_col = "Code" if "Code" in symbols.columns else symbols.columns[0]
        existing_codes = set(symbols[code_col].str.upper())
        missing_etfs = [e for e in LIQUID_ETFS if e.upper() not in existing_codes]
        if missing_etfs:
            all_symbols = client.get_exchange_symbols("US")
            if not all_symbols.empty:
                all_code_col = "Code" if "Code" in all_symbols.columns else all_symbols.columns[0]
                etf_rows = all_symbols[all_symbols[all_code_col].str.upper().isin(
                    {e.upper() for e in missing_etfs}
                )]
                if not etf_rows.empty:
                    symbols = pd.concat([symbols, etf_rows], ignore_index=True)
                    symbols = symbols.drop_duplicates(subset=[code_col], keep="first")
                    logger.info("  Added %d curated ETFs -> %d total", len(etf_rows), len(symbols))

    # Save
    path = data_dir / "universe.parquet"
    symbols.to_parquet(path, index=False)
    logger.info("Universe saved: %d tickers -> %s", len(symbols), path)
    return symbols


# ---------------------------------------------------------------------------
# 2. Fetch 1-min bars for a single ticker (incremental)
# ---------------------------------------------------------------------------

def _ts_chunks(
    start: datetime, end: datetime, chunk_days: int = CHUNK_DAYS,
) -> list[tuple[int, int]]:
    """Split [start, end) into (from_unix, to_unix) pairs."""
    chunks: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((int(cursor.timestamp()), int(chunk_end.timestamp())))
        cursor = chunk_end
    return chunks


def _existing_range(path: Path) -> tuple[datetime | None, datetime | None]:
    """Read the min/max timestamp already stored in a parquet file."""
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None, None
        idx = pd.to_datetime(df.index)
        return idx.min().to_pydatetime(), idx.max().to_pydatetime()
    except Exception:
        return None, None


def _normalise_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw EODHD intraday response into a clean OHLCV DataFrame."""
    if df.empty:
        return df

    df.columns = df.columns.str.lower()

    ts_col = None
    for candidate in ("datetime", "timestamp", "date"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index("timestamp").sort_index()

    col_map = {}
    for target in ("open", "high", "low", "close", "volume"):
        for col in df.columns:
            if col.lower() == target:
                col_map[col] = target
                break
    df = df.rename(columns=col_map)

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep]


def ingest_ticker(
    client: EODHDClient,
    ticker: str,
    exchange: str,
    start: datetime,
    end: datetime,
    bars_dir: Path,
) -> int:
    """Fetch 1-min bars for a single ticker and save to parquet.

    Incremental: only fetches date ranges not already on disk.
    Returns the total number of bars in the final file.
    """
    parquet_path = bars_dir / f"{ticker}__{exchange}.parquet"

    existing_min, existing_max = _existing_range(parquet_path)
    existing_df: pd.DataFrame | None = None
    if existing_min is not None:
        existing_df = pd.read_parquet(parquet_path)

    chunks: list[tuple[int, int]] = []
    if existing_min is None:
        chunks = _ts_chunks(start, end)
    else:
        if start < existing_min:
            chunks.extend(_ts_chunks(start, existing_min))
        if end > existing_max:
            chunks.extend(_ts_chunks(existing_max, end))

    if not chunks:
        return len(existing_df) if existing_df is not None else 0

    new_frames: list[pd.DataFrame] = []
    for from_ts, to_ts in chunks:
        try:
            raw = client.get_intraday(ticker, exchange, from_ts, to_ts, interval="1m")
            normed = _normalise_intraday(raw)
            if not normed.empty:
                new_frames.append(normed)
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("  %s.%s chunk %d-%d failed: %s", ticker, exchange, from_ts, to_ts, exc)

    parts = [existing_df] if existing_df is not None else []
    parts.extend(new_frames)

    if not parts:
        return 0

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_parquet(parquet_path)
    return len(combined)


# ---------------------------------------------------------------------------
# 3. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: Path,
    years: int = DEFAULT_YEARS,
    tickers: list[str] | None = None,
    budget: int | None = None,
    skip_universe: bool = False,
) -> None:
    """Run the full EODHD US equities ingestion pipeline."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bars_dir = data_dir / "bars_1m"
    bars_dir.mkdir(parents=True, exist_ok=True)

    client = EODHDClient(budget=budget)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)
    logger.info(
        "EODHD ingestion window: %s -- %s (%d years)",
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), years,
    )

    if tickers:
        ticker_list = []
        for t in tickers:
            parts = t.rsplit(".", 1)
            if len(parts) == 2:
                ticker_list.append((parts[0], parts[1]))
            else:
                logger.warning("Skipping malformed ticker '%s' (expected TICKER.EXCHANGE)", t)
        logger.info("Explicit ticker list: %d tickers", len(ticker_list))
    else:
        universe_path = data_dir / "universe.parquet"
        if skip_universe and universe_path.exists():
            universe = pd.read_parquet(universe_path)
            logger.info("Reusing existing universe (%d tickers)", len(universe))
        else:
            universe = build_universe(client, data_dir)

        if universe.empty:
            return

        code_col = "Code" if "Code" in universe.columns else universe.columns[0]
        exch_col = "exchange_code"
        ticker_list = list(zip(universe[code_col], universe[exch_col]))
        logger.info("Universe: %d tickers to ingest", len(ticker_list))

    stats = {"ok": 0, "empty": 0, "error": 0, "total_bars": 0}
    try:
        with tqdm(total=len(ticker_list), desc="EODHD intraday", unit="tkr") as pbar:
            for ticker, exchange in ticker_list:
                pbar.set_postfix_str(f"{ticker}.{exchange}")
                try:
                    n = ingest_ticker(client, ticker, exchange, start, end, bars_dir)
                    if n > 0:
                        stats["ok"] += 1
                        stats["total_bars"] += n
                    else:
                        stats["empty"] += 1
                except BudgetExhausted:
                    logger.warning("Budget exhausted -- stopping. Re-run to continue.")
                    break
                except Exception as exc:
                    stats["error"] += 1
                    logger.error("  %s.%s failed: %s", ticker, exchange, exc)
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted -- progress saved. Re-run to continue.")

    logger.info("=" * 60)
    logger.info("EODHD INGESTION COMPLETE")
    logger.info("  Tickers OK:       %6d", stats["ok"])
    logger.info("  Tickers empty:    %6d", stats["empty"])
    logger.info("  Tickers error:    %6d", stats["error"])
    logger.info("  Total 1-min bars: %6d", stats["total_bars"])
    logger.info("  API calls made:   %6d", client.calls_made)
    logger.info("  Parquet dir:      %s", bars_dir)
    logger.info("=" * 60)
