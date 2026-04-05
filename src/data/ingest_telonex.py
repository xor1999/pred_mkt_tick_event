"""Telonex prediction-market data ingestion pipeline.

Fetches fill-level data via the Telonex SDK, resamples to 1-minute OHLCV
bars, and stores results as Parquet files.  Designed for incremental runs.

By default, only macro-relevant markets are ingested (Fed, CPI, GDP,
tariffs, recession, treasury yields, etc.).  Pass ``macro_only=False``
to ingest everything (warning: 872k+ markets).

The SDK handles chunking, concurrency (5 parallel downloads), and file
caching internally -- no artificial throttling needed.

Storage layout
--------------
    {data_dir}/
        telonex_markets.parquet          # filtered catalog snapshot
        bars_1m/
            {slug}__{outcome}.parquet
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.telonex_client import TelonexClient

logger = logging.getLogger(__name__)

DEFAULT_YEARS = 2


# ---------------------------------------------------------------------------
# Macro keyword filter
# ---------------------------------------------------------------------------

# Keywords that identify macro-relevant prediction markets.
# Searched against: event_title + question + slug (lowercased).
MACRO_KEYWORDS: list[str] = [
    # Central banks / rates
    "federal reserve", "fomc", "fed rate", "fed cut", "fed hike",
    "fed decision", "fed meeting", "fed chair", "powell",
    "fed interest", "fed funds",
    "ecb interest", "ecb rate", "ecb decision",
    "bank of england", "boe rate", "boe decision",
    "bank of japan", "boj rate", "boj decision",
    "rate cut", "rate hike", "interest rate",
    # Inflation (US-focused)
    "cpi ", "consumer price index", "core cpi",
    "inflation us", "inflation rate",
    "pce ", "core pce",
    "ppi ",
    # Growth / employment
    "gdp ", "gross domestic product",
    "unemployment rate", "nonfarm payroll", "jobs report",
    "jobless claims",
    "recession",
    "pmi ",
    "retail sales",
    # Equity indices
    "s&p 500", "s&p500", "spx", "spy ",
    "nasdaq 100", "nasdaq-100", "ndx",
    "dow jones", "djia",
    # Fixed income
    "treasury yield", "10-year treasury", "yield curve",
    "10y yield",
    # Tariffs / trade
    "tariff",
    "trade war", "trade deal", "trade agreement",
    # Fiscal / government
    "debt ceiling", "government shutdown",
    "sanctions",
]


def _filter_macro_markets(catalog: pd.DataFrame) -> pd.DataFrame:
    """Filter the full Polymarket catalog to macro-relevant markets only.

    Uses vectorized string matching instead of row-wise apply to handle
    the 600k+ row catalog without blowing up memory.
    """
    # Must have fill data available
    has_fills = (
        catalog["onchain_fills_from"].notna()
        & (catalog["onchain_fills_from"] != "")
    )
    catalog_with_fills = catalog[has_fills].copy()
    logger.info("  Markets with fill data: %d / %d", len(catalog_with_fills), len(catalog))

    # Build a single search string per row (vectorized)
    search_text = (
        catalog_with_fills["event_title"].fillna("").astype(str) + " "
        + catalog_with_fills["question"].fillna("").astype(str) + " "
        + catalog_with_fills["slug"].fillna("").astype(str)
    ).str.lower()

    # Match any macro keyword
    macro_mask = pd.Series(False, index=catalog_with_fills.index)
    for kw in MACRO_KEYWORDS:
        macro_mask |= search_text.str.contains(kw, regex=False, na=False)

    filtered = catalog_with_fills[macro_mask].copy()
    logger.info("  Macro-relevant markets: %d", len(filtered))

    return filtered


# ---------------------------------------------------------------------------
# 1. Market catalog
# ---------------------------------------------------------------------------

def fetch_market_catalog(
    client: TelonexClient,
    data_dir: Path,
    macro_only: bool = True,
) -> pd.DataFrame:
    """Fetch the market catalog, optionally filter to macro, and save."""
    logger.info("Fetching Polymarket catalog via Telonex ...")
    catalog = client.list_markets(download_dir=str(data_dir / "_tmp"))

    if catalog.empty:
        logger.warning("Market catalog is empty -- nothing to ingest.")
        return pd.DataFrame()

    if macro_only:
        catalog = _filter_macro_markets(catalog)

    path = data_dir / "telonex_markets.parquet"
    catalog.to_parquet(path, index=False)
    logger.info("Saved %d markets -> %s", len(catalog), path)
    return catalog


# ---------------------------------------------------------------------------
# 2. Fetch + resample one market/outcome
# ---------------------------------------------------------------------------

def _existing_range(path: Path) -> tuple[datetime | None, datetime | None]:
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


def ingest_market_outcome(
    client: TelonexClient,
    slug: str,
    outcome: str,
    start: datetime,
    end: datetime,
    bars_dir: Path,
) -> int:
    """Fetch 1-min bars for a single market/outcome and save to parquet.

    Makes a single SDK call for the full date range -- the SDK handles
    chunking and caching internally.  Merges with existing data on disk.

    Returns the total number of bars in the final file.
    """
    safe_outcome = outcome.replace(" ", "_").replace("/", "_").lower()
    parquet_path = bars_dir / f"{slug}__{safe_outcome}.parquet"

    existing_min, existing_max = _existing_range(parquet_path)
    existing_df: pd.DataFrame | None = None
    if existing_min is not None:
        existing_df = pd.read_parquet(parquet_path)

    # Check if we already cover the full range
    if existing_min is not None and existing_min <= start and existing_max >= end:
        return len(existing_df) if existing_df is not None else 0

    # Fetch the full range in one call (SDK caches internally)
    from_date = start.strftime("%Y-%m-%d")
    to_date = end.strftime("%Y-%m-%d")

    try:
        bars = client.get_ohlcv_1m(slug, outcome, from_date, to_date)
    except Exception as exc:
        logger.warning("  %s/%s fetch failed: %s", slug, outcome, exc)
        return len(existing_df) if existing_df is not None else 0

    if bars.empty and existing_df is not None:
        return len(existing_df)

    parts = [existing_df] if existing_df is not None else []
    if not bars.empty:
        parts.append(bars)

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
    exchange: str = "polymarket",
    slugs: list[str] | None = None,
    skip_catalog: bool = False,
    macro_only: bool = True,
) -> None:
    """Run the Telonex/Polymarket ingestion pipeline.

    By default, only macro-relevant markets are ingested (Fed, CPI, GDP,
    tariffs, recession, treasury yields, S&P 500, Nasdaq, etc.).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bars_dir = data_dir / "bars_1m"
    bars_dir.mkdir(parents=True, exist_ok=True)

    client = TelonexClient(exchange=exchange, download_dir=str(data_dir / "_cache"))

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)
    logger.info(
        "Ingestion window: %s -- %s (%d years)",
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), years,
    )

    # Step 1 -- market catalog
    catalog_path = data_dir / "telonex_markets.parquet"
    if skip_catalog and catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        logger.info("Reusing existing catalog (%d markets)", len(catalog))
    else:
        catalog = fetch_market_catalog(client, data_dir, macro_only=macro_only)

    if catalog.empty:
        logger.error("No markets found -- aborting.")
        return

    # Filter to requested slugs if provided
    slug_col = "slug" if "slug" in catalog.columns else catalog.columns[0]
    if slugs:
        catalog = catalog[catalog[slug_col].isin(slugs)]
        logger.info("Filtered to %d requested slugs", len(catalog))

    # Determine outcomes and active date range per market.
    # Use onchain_fills_from/to to avoid requesting days with no data.
    has_o0 = "outcome_0" in catalog.columns
    has_o1 = "outcome_1" in catalog.columns
    has_from = "onchain_fills_from" in catalog.columns
    has_to = "onchain_fills_to" in catalog.columns

    # Each entry: (slug, [outcomes], active_start, active_end)
    market_tasks: list[tuple[str, list[str], datetime, datetime]] = []
    total_days = 0

    for _, row in catalog.iterrows():
        market_slug = row[slug_col]
        outcomes = []
        if has_o0 and pd.notna(row["outcome_0"]):
            outcomes.append(str(row["outcome_0"]))
        if has_o1 and pd.notna(row["outcome_1"]):
            outcomes.append(str(row["outcome_1"]))
        if not outcomes:
            outcomes = ["Yes", "No"]

        # Use the catalog's fill date range (clipped to our window)
        m_start = start
        m_end = end
        if has_from and pd.notna(row["onchain_fills_from"]):
            try:
                cat_from = pd.to_datetime(row["onchain_fills_from"], utc=True).to_pydatetime()
                m_start = max(start, cat_from)
            except Exception:
                pass
        if has_to and pd.notna(row["onchain_fills_to"]):
            try:
                cat_to = pd.to_datetime(row["onchain_fills_to"], utc=True).to_pydatetime()
                m_end = min(end, cat_to + timedelta(days=1))
            except Exception:
                pass

        if m_start < m_end:
            market_tasks.append((market_slug, outcomes, m_start, m_end))
            total_days += (m_end - m_start).days * len(outcomes)

    total_tasks = sum(len(outs) for _, outs, _, _ in market_tasks)
    logger.info(
        "Ingesting %d markets (%d market/outcome pairs), ~%d total days of data",
        len(market_tasks), total_tasks, total_days,
    )

    # Step 2 -- fetch bars for each market/outcome (parallelised)
    # Each market/outcome is independent; run up to `max_workers` in parallel.
    # The SDK also does internal concurrency per market (day-level downloads).
    max_workers = 4
    stats = {"ok": 0, "empty": 0, "error": 0, "total_bars": 0}
    lock = threading.Lock()

    # Flatten to individual (slug, outcome, start, end) tasks
    flat_tasks = [
        (slug, outcome, m_start, m_end)
        for slug, outcomes, m_start, m_end in market_tasks
        for outcome in outcomes
    ]

    def _process_one(task: tuple) -> tuple[str, str, int]:
        slug, outcome, m_start, m_end = task
        # Each thread gets its own client to avoid shared state issues
        worker_client = TelonexClient(
            exchange=exchange,
            download_dir=str(data_dir / "_cache"),
        )
        try:
            n = ingest_market_outcome(worker_client, slug, outcome, m_start, m_end, bars_dir)
            return (slug, outcome, n)
        except Exception as exc:
            logger.error("  %s/%s failed: %s", slug, outcome, exc)
            return (slug, outcome, -1)

    with tqdm(total=total_tasks, desc="Telonex ingestion", unit="mkt") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one, t): t for t in flat_tasks}
            for future in as_completed(futures):
                slug, outcome, n = future.result()
                with lock:
                    if n > 0:
                        stats["ok"] += 1
                        stats["total_bars"] += n
                    elif n == 0:
                        stats["empty"] += 1
                    else:
                        stats["error"] += 1
                pbar.set_postfix_str(f"{slug[:30]}/{outcome}")
                pbar.update(1)

    logger.info("=" * 60)
    logger.info("TELONEX INGESTION COMPLETE")
    logger.info("  Markets/outcomes OK:    %6d", stats["ok"])
    logger.info("  Markets/outcomes empty: %6d", stats["empty"])
    logger.info("  Markets/outcomes error: %6d", stats["error"])
    logger.info("  Total 1-min bars:       %6d", stats["total_bars"])
    logger.info("  API calls made:         %6d", client.calls_made)
    logger.info("  Parquet dir:            %s", bars_dir)
    logger.info("=" * 60)
