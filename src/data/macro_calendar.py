"""Historical macro event calendar backed by DuckDB.

Fetches economic events from EODHD (CPI, NFP, FOMC, GDP, PCE, PMI, ...)
and stores them in a ``macro_events`` table in the ticks database.
Provides helpers to query event windows for tick ingestion.

Table schema
------------
    macro_events (
        event_id    VARCHAR PK,
        type        VARCHAR,          -- canonical category (CPI, NFP, FOMC, ...)
        country     VARCHAR,          -- ISO-3166 alpha-2
        date        TIMESTAMP,        -- release datetime (UTC)
        actual      DOUBLE,
        estimate    DOUBLE,
        previous    DOUBLE,
        change      DOUBLE,           -- actual - estimate (surprise)
        impact      VARCHAR,          -- "High", "Medium", "Low"
        currency    VARCHAR,
        raw_event   VARCHAR,          -- original EODHD event type string
    )
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

from src.data.eodhd_client import EODHDClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MACRO_EVENTS_DDL = """\
CREATE TABLE IF NOT EXISTS macro_events (
    event_id    VARCHAR PRIMARY KEY,
    type        VARCHAR NOT NULL,
    country     VARCHAR NOT NULL,
    date        TIMESTAMP NOT NULL,
    actual      DOUBLE,
    estimate    DOUBLE,
    previous    DOUBLE,
    change      DOUBLE,
    impact      VARCHAR,
    currency    VARCHAR,
    raw_event   VARCHAR
);
"""

MACRO_EVENT_CATEGORIES: dict[str, str] = {
    # Inflation
    "CPI":                                "CPI",
    "Core CPI":                           "CPI",
    "Core Consumer Price Index":          "CPI",
    "Consumer Price Index":               "CPI",
    "PCE Price Index":                    "PCE",
    "Core PCE Price Index":               "PCE",
    "PPI":                                "PPI",
    "Producer Price Index":               "PPI",
    # Employment
    "Non-Farm Payrolls":                  "NFP",
    "Nonfarm Payrolls":                   "NFP",
    "Non Farm Payrolls":                  "NFP",
    "Unemployment Rate":                  "UNEMPLOYMENT",
    "Initial Jobless Claims":             "JOBLESS_CLAIMS",
    "JOLTs Job Openings":                 "JOLTS",
    "ADP Employment Change":              "ADP",
    # Growth
    "GDP":                                "GDP",
    "GDP Growth Rate":                    "GDP",
    "Gross Domestic Product":             "GDP",
    # Central bank
    "Interest Rate Decision":             "RATE_DECISION",
    "Fed Interest Rate Decision":         "FOMC",
    "FOMC":                               "FOMC",
    "Fed Funds Rate":                     "FOMC",
    # Activity
    "Retail Sales":                       "RETAIL_SALES",
    "Industrial Production":              "INDUSTRIAL_PROD",
    "ISM Manufacturing PMI":              "PMI",
    "ISM Non-Manufacturing PMI":          "PMI",
    "ISM Services PMI":                   "PMI",
    "PMI":                                "PMI",
    "S&P Global Manufacturing PMI":       "PMI",
    "S&P Global Services PMI":            "PMI",
    # Housing
    "Building Permits":                   "HOUSING",
    "Housing Starts":                     "HOUSING",
    "Existing Home Sales":                "HOUSING",
    "New Home Sales":                     "HOUSING",
    # Consumer
    "Consumer Confidence":                "CONSUMER_CONFIDENCE",
    "Michigan Consumer Sentiment":        "CONSUMER_SENTIMENT",
    "Personal Income":                    "PERSONAL_INCOME",
    "Personal Spending":                  "PERSONAL_SPENDING",
    # Trade
    "Trade Balance":                      "TRADE_BALANCE",
    # Misc
    "Durable Goods Orders":               "DURABLE_GOODS",
    "Factory Orders":                     "FACTORY_ORDERS",
}


def _event_id(event_type: str, country: str, date_str: str) -> str:
    """Deterministic event ID from type + country + datetime."""
    raw = f"{event_type}|{country}|{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _classify_event(raw_type: str) -> str:
    """Map a raw EODHD event type string to a canonical category."""
    if not raw_type:
        return "OTHER"
    if raw_type in MACRO_EVENT_CATEGORIES:
        return MACRO_EVENT_CATEGORIES[raw_type]
    raw_lower = raw_type.lower()
    for pattern, category in MACRO_EVENT_CATEGORIES.items():
        if pattern.lower() in raw_lower:
            return category
    return "OTHER"


def _to_float(val) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None or val == "" or val == "None":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetch + store
# ---------------------------------------------------------------------------

def init_calendar_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the macro_events table if it doesn't exist."""
    con.execute(MACRO_EVENTS_DDL)


def fetch_and_store_events(
    client: EODHDClient,
    con: duckdb.DuckDBPyConnection,
    country: str = "US",
    years: int = 2,
    high_impact_only: bool = False,
) -> int:
    """Fetch historical macro events from EODHD and INSERT into DuckDB.

    Returns total events inserted.
    """
    init_calendar_table(con)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)

    total_inserted = 0
    cursor = start
    chunks = []
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=31), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end

    logger.info(
        "Fetching %s macro events: %s -- %s (%d monthly chunks, 1 credit each)",
        country, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), len(chunks),
    )

    for from_date, to_date in tqdm(chunks, desc=f"Macro events ({country})", unit="mo"):
        try:
            events = client.get_economic_events(country, from_date, to_date)
            if not events:
                continue

            rows = []
            for ev in events:
                impact = (ev.get("impact") or "").strip()
                if high_impact_only and impact.lower() != "high":
                    continue

                raw_type = ev.get("type") or ev.get("event") or ""
                date_str = ev.get("date") or ""

                rows.append({
                    "event_id": _event_id(raw_type, country, date_str),
                    "type": _classify_event(raw_type),
                    "country": country,
                    "date": date_str,
                    "actual": _to_float(ev.get("actual")),
                    "estimate": _to_float(ev.get("estimate")),
                    "previous": _to_float(ev.get("previous")),
                    "change": _to_float(ev.get("change")),
                    "impact": impact,
                    "currency": ev.get("currency") or "",
                    "raw_event": raw_type,
                })

            if rows:
                df = pd.DataFrame(rows)
                con.execute("INSERT OR IGNORE INTO macro_events SELECT * FROM df")
                total_inserted += len(rows)

        except Exception as exc:
            logger.warning("  %s %s--%s failed: %s", country, from_date, to_date, exc)

    logger.info("  %s: %d events inserted", country, total_inserted)
    return total_inserted


# ---------------------------------------------------------------------------
# Event window queries
# ---------------------------------------------------------------------------

def get_event_windows(
    con: duckdb.DuckDBPyConnection,
    country: str = "US",
    impact: str | None = "High",
    categories: list[str] | None = None,
    before: timedelta = timedelta(hours=72),
    after: timedelta = timedelta(hours=72),
) -> pd.DataFrame:
    """Query macro events and compute tick-ingestion windows.

    Returns DataFrame with window_start and window_end columns added.
    """
    conditions = ["country = ?"]
    params: list = [country]

    if impact:
        conditions.append("impact = ?")
        params.append(impact)

    if categories:
        placeholders = ", ".join(["?"] * len(categories))
        conditions.append(f"type IN ({placeholders})")
        params.extend(categories)

    where = " AND ".join(conditions)
    query = f"""
        SELECT
            event_id, type, country, date, actual, estimate, previous,
            change, impact, raw_event
        FROM macro_events
        WHERE {where}
        ORDER BY date
    """
    df = con.execute(query, params).fetchdf()

    if df.empty:
        df["window_start"] = pd.Series(dtype="datetime64[ns, UTC]")
        df["window_end"] = pd.Series(dtype="datetime64[ns, UTC]")
        return df

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["window_start"] = df["date"] - before
    df["window_end"] = df["date"] + after

    return df


def get_merged_windows(
    con: duckdb.DuckDBPyConnection,
    country: str = "US",
    impact: str | None = "High",
    categories: list[str] | None = None,
    before: timedelta = timedelta(hours=72),
    after: timedelta = timedelta(hours=72),
) -> list[tuple[int, int]]:
    """Get merged (non-overlapping) event windows as Unix timestamp pairs.

    When events are close together, their windows are merged to avoid
    duplicate tick fetches.
    """
    df = get_event_windows(con, country, impact, categories, before, after)
    if df.empty:
        return []

    intervals = sorted(zip(df["window_start"], df["window_end"]))
    merged: list[tuple[datetime, datetime]] = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return [(int(s.timestamp()), int(e.timestamp())) for s, e in merged]


def calendar_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a summary of events in the calendar by type and impact."""
    return con.execute("""
        SELECT
            type,
            impact,
            COUNT(*) as count,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM macro_events
        GROUP BY type, impact
        ORDER BY count DESC
    """).fetchdf()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

MACRO_COUNTRIES = ["US"]


def run_pipeline(
    data_dir: Path,
    years: int = 2,
    countries: list[str] | None = None,
    high_impact_only: bool = False,
    budget: int | None = None,
) -> None:
    """Fetch historical macro events and store in the ticks DuckDB."""
    from src.data.ingest_ticks import _init_db

    data_dir = Path(data_dir)
    db_path = data_dir / "ticks.duckdb"
    con = _init_db(db_path)
    init_calendar_table(con)

    client = EODHDClient(budget=budget)
    target_countries = countries or MACRO_COUNTRIES

    total = 0
    for country in target_countries:
        n = fetch_and_store_events(
            client, con, country=country, years=years,
            high_impact_only=high_impact_only,
        )
        total += n

    logger.info("=" * 60)
    logger.info("MACRO CALENDAR COMPLETE")
    logger.info("  Total events:     %6d", total)
    logger.info("  API calls made:   %6d", client.calls_made)

    summary = calendar_summary(con)
    if not summary.empty:
        logger.info("  Top event types:")
        for _, row in summary.head(10).iterrows():
            logger.info("    %-25s  %s impact  %4d events", row["type"], row["impact"], row["count"])

    con.close()
    logger.info("  DB: %s", db_path)
    logger.info("=" * 60)
