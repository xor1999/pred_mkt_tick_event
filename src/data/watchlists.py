"""Tiered ticker watchlists for data ingestion.

Three modes control what gets fetched at tick-level resolution:

  macro   -- ~50-80 macro-sensitive names: major indices, sector ETFs,
             bond ETFs, rate-sensitive stocks, volatility.
             Cost: ~12k credits, ~10 GB, ~30 min.

  full    -- Entire liquid universe (stocks >$1B + liquid ETFs).
             Cost: ~170k credits, ~100-150 GB, runs over 2 days.

  custom  -- User-provided ticker list via --ticker flags.

The 1-min bar pipeline always runs on the full universe.
Tick ingestion uses the mode to decide what to fetch.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Macro-sensitive watchlist
# ---------------------------------------------------------------------------

MACRO_INDEX_ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow 30
    "XLF",   # Financials (most rate-sensitive)
    "XLK",   # Tech
    "XLE",   # Energy
    "XLU",   # Utilities (rate-sensitive)
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLV",   # Healthcare
    "XLB",   # Materials
    "XLRE",  # Real Estate (rate-sensitive)
    "XLC",   # Communication Services
]

MACRO_BOND_ETFS = [
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "SHY",   # 1-3 Year Treasury
    "TIP",   # TIPS (inflation-linked)
    "HYG",   # High Yield Corporate
    "LQD",   # Investment Grade Corporate
    "AGG",   # US Aggregate Bond
    "BND",   # Total Bond Market
]

MACRO_VOLATILITY = [
    "VXX",   # VIX Short-Term Futures
    "UVXY",  # 1.5x VIX Short-Term
    "SVXY",  # -0.5x VIX Short-Term
]

MACRO_COMMODITY_FX = [
    "GLD",   # Gold (inflation hedge)
    "SLV",   # Silver
    "USO",   # Oil
    "UUP",   # US Dollar Index
    "FXE",   # Euro
    "FXY",   # Yen
    "FXB",   # British Pound
]

MACRO_RATE_SENSITIVE_STOCKS = [
    # Big banks (Fed rate decisions)
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    # Insurance (rate-sensitive)
    "BRK-B", "MET", "PRU",
    # REITs (rate-sensitive)
    "AMT", "PLD", "EQIX",
    # Homebuilders (housing data)
    "DHI", "LEN", "PHM",
    # Consumer bellwethers (CPI, retail sales)
    "WMT", "AMZN", "COST", "TGT", "HD",
    # Mega-cap tech (market-moving)
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA",
]

MACRO_WATCHLIST: list[str] = sorted(set(
    MACRO_INDEX_ETFS
    + MACRO_BOND_ETFS
    + MACRO_VOLATILITY
    + MACRO_COMMODITY_FX
    + MACRO_RATE_SENSITIVE_STOCKS
))

# ---------------------------------------------------------------------------
# Liquid ETF list (for universe builder)
# ---------------------------------------------------------------------------

LIQUID_ETFS: list[str] = sorted(set(
    MACRO_INDEX_ETFS
    + MACRO_BOND_ETFS
    + MACRO_VOLATILITY
    + MACRO_COMMODITY_FX
    + [
        "EEM",   # Emerging Markets
        "EFA",   # EAFE (Developed ex-US)
        "VWO",   # Emerging Markets (Vanguard)
        "VEA",   # Developed Markets (Vanguard)
        "IEMG",  # Emerging Markets (iShares)
        "ARKK",  # ARK Innovation
        "SMH",   # Semiconductors
        "XBI",   # Biotech
        "KRE",   # Regional Banks
        "ITB",   # Homebuilders
        "KWEB",  # China Internet
        "EWJ",   # Japan
        "EWZ",   # Brazil
        "SOXL",  # 3x Semiconductors
        "TQQQ",  # 3x Nasdaq
    ]
))


def get_watchlist(mode: str, custom: list[str] | None = None) -> list[str]:
    """Return the ticker list for a given mode."""
    if mode == "macro":
        return MACRO_WATCHLIST
    elif mode == "full":
        return MACRO_WATCHLIST + [e for e in LIQUID_ETFS if e not in MACRO_WATCHLIST]
    elif mode == "custom":
        if not custom:
            raise ValueError("mode='custom' requires a non-empty ticker list")
        return [t.split(".")[0] for t in custom]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'macro', 'full', or 'custom'.")


def estimate_cost(n_tickers: int, years: int = 2, chunk_days: int = 30) -> dict:
    """Estimate API credits, disk, and runtime for tick ingestion."""
    n_chunks = (years * 365) // chunk_days + 1
    api_calls = n_tickers * n_chunks
    credits = api_calls * 10

    ticks_per_ticker = 30_000 * 252 * years
    total_ticks = n_tickers * ticks_per_ticker
    disk_gb = total_ticks * 12 / 1e9

    runtime_min = api_calls / 5 / 60

    return {
        "tickers": n_tickers,
        "api_calls": api_calls,
        "credits": credits,
        "credits_per_day_needed": credits,
        "est_disk_gb": round(disk_gb, 1),
        "est_runtime_min": round(runtime_min, 1),
        "est_total_ticks_billions": round(total_ticks / 1e9, 2),
    }
