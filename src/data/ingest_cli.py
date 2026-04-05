"""CLI entry point for all data ingestion pipelines.

Subcommands (1-min bars):
    telonex  -- Prediction-market 1-min bars (Polymarket via Telonex SDK)
    eodhd    -- Equities 1-min intraday OHLCV (US via EODHD)
    kalshi   -- Kalshi macro contracts (disabled by default -- US-only API access)
    all      -- Run eodhd + telonex bar pipelines sequentially

Subcommands (tick-by-tick):
    ticks          -- Raw tick data: EODHD US equities
    event-ticks    -- Ticks only around macro events (CPI, NFP, FOMC, GDP, ...)

Subcommand (calendar):
    calendar       -- Fetch historical macro event calendar from EODHD

Usage examples:
    python -m src.data.ingest_cli eodhd
    python -m src.data.ingest_cli calendar -v
    python -m src.data.ingest_cli event-ticks --category CPI --category FOMC
    python -m src.data.ingest_cli ticks --ticker AAPL
    python -m src.data.ingest_cli all --years 2 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(log_dir: Path, verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_fmt, datefmt="%H:%M:%S"))

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "ingestion.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


# -- Subcommand handlers ---------------------------------------------------

def _cmd_telonex(args: argparse.Namespace) -> None:
    from src.data.ingest_telonex import run_pipeline
    run_pipeline(
        data_dir=Path(args.data_dir) / "telonex",
        years=args.years,
        exchange=getattr(args, "exchange", "polymarket"),
        slugs=getattr(args, "slugs", None),
        skip_catalog=getattr(args, "skip_catalog", False),
        macro_only=not getattr(args, "all_markets", False),
    )


def _cmd_eodhd(args: argparse.Namespace) -> None:
    from src.data.ingest_eodhd import run_pipeline
    run_pipeline(
        data_dir=Path(args.data_dir) / "eodhd",
        years=args.years,
        tickers=getattr(args, "tickers", None),
        budget=getattr(args, "budget", None),
        skip_universe=getattr(args, "skip_universe", False),
    )


def _cmd_kalshi(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Kalshi API is restricted to US-based users. "
        "If you are outside the US, this will fail. "
        "Use 'telonex' for Polymarket data instead."
    )
    from src.data.ingest_kalshi import run_pipeline
    run_pipeline(
        data_dir=Path(args.data_dir) / "kalshi",
        years=args.years,
        tickers=getattr(args, "tickers", None),
        skip_catalog=getattr(args, "skip_catalog", False),
        macro_only=not getattr(args, "all_events", False),
    )


def _cmd_ticks(args: argparse.Namespace) -> None:
    from src.data.ingest_ticks import run_pipeline
    run_pipeline(
        data_dir=Path(args.data_dir),
        years=args.years,
        source=getattr(args, "source", "eodhd"),
        mode=getattr(args, "mode", "macro"),
        eodhd_tickers=getattr(args, "eodhd_tickers", None),
        budget=getattr(args, "budget", None),
        skip_universe=getattr(args, "skip_universe", False),
    )


def _cmd_calendar(args: argparse.Namespace) -> None:
    from src.data.macro_calendar import run_pipeline
    run_pipeline(
        data_dir=Path(args.data_dir),
        years=args.years,
        countries=getattr(args, "countries", None),
        high_impact_only=getattr(args, "high_impact_only", False),
        budget=getattr(args, "budget", None),
    )


def _cmd_event_ticks(args: argparse.Namespace) -> None:
    from src.data.ingest_ticks import run_event_window_pipeline
    run_event_window_pipeline(
        data_dir=Path(args.data_dir),
        years=args.years,
        source=getattr(args, "source", "eodhd"),
        mode=getattr(args, "mode", "macro"),
        eodhd_tickers=getattr(args, "eodhd_tickers", None),
        budget=getattr(args, "budget", None),
        skip_universe=getattr(args, "skip_universe", False),
        categories=getattr(args, "categories", None),
        before_hours=getattr(args, "before", 72.0),
        after_hours=getattr(args, "after", 72.0),
    )


def _cmd_all(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    for name, handler in [("telonex", _cmd_telonex), ("eodhd", _cmd_eodhd)]:
        logger.info("=" * 60)
        logger.info("Starting %s pipeline ...", name)
        logger.info("=" * 60)
        try:
            handler(args)
        except Exception as exc:
            logger.error("%s pipeline failed: %s", name, exc)


# -- CLI definition ---------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root output directory (default: data/raw)",
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Lookback window in years (default: 2)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug-level logging",
    )


def _add_tick_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["macro", "full", "custom"], default="macro",
                       help="Ticker universe mode (default: macro)")
    parser.add_argument("--source", choices=["eodhd"], default="eodhd",
                       help="Tick source (default: eodhd)")
    parser.add_argument("--ticker", action="append", dest="eodhd_tickers", metavar="SYM",
                       help="EODHD: ingest specific US tickers (e.g. --ticker AAPL)")
    parser.add_argument("--budget", type=int, default=None,
                       help="Max EODHD API credits for this run (default: unlimited)")
    parser.add_argument("--skip-universe", action="store_true",
                       help="Reuse existing eodhd/universe.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Data ingestion: equities (EODHD) + prediction markets (Telonex/Polymarket)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- telonex --
    p_tel = sub.add_parser("telonex", help="Telonex/Polymarket prediction-market 1-min bars")
    _add_common_args(p_tel)
    p_tel.add_argument("--exchange", default="polymarket", help="Telonex exchange (default: polymarket)")
    p_tel.add_argument("--slug", action="append", dest="slugs", metavar="SLUG",
                       help="Only ingest specific slug(s). Repeatable.")
    p_tel.add_argument("--skip-catalog", action="store_true",
                       help="Reuse existing market catalog")
    p_tel.add_argument("--all-markets", action="store_true",
                       help="Ingest all Polymarket markets, not just macro-relevant ones")

    # -- eodhd --
    p_eod = sub.add_parser("eodhd", help="EODHD US equities 1-min intraday OHLCV")
    _add_common_args(p_eod)
    p_eod.add_argument("--ticker", action="append", dest="tickers", metavar="TICKER.US",
                       help="Skip universe build; ingest specific tickers (e.g. --ticker AAPL.US)")
    p_eod.add_argument("--budget", type=int, default=None,
                       help="Max API credits for this run (default: unlimited)")
    p_eod.add_argument("--skip-universe", action="store_true",
                       help="Reuse existing universe.parquet")

    # -- kalshi (kept but marked as US-only) --
    p_kal = sub.add_parser("kalshi",
                           help="[US-ONLY] Kalshi macro prediction-market 1-min bars (requires US API access)")
    _add_common_args(p_kal)
    p_kal.add_argument("--ticker", action="append", dest="tickers", metavar="TICKER",
                       help="Only ingest specific contract ticker(s). Repeatable.")
    p_kal.add_argument("--skip-catalog", action="store_true",
                       help="Reuse existing event/market catalog")
    p_kal.add_argument("--all-events", action="store_true",
                       help="Ingest all events, not just macro-relevant ones")

    # -- ticks (continuous) --
    p_tck = sub.add_parser("ticks", help="Tick-by-tick: EODHD US equities (full lookback)")
    _add_common_args(p_tck)
    _add_tick_source_args(p_tck)

    # -- calendar --
    p_cal = sub.add_parser("calendar", help="Fetch historical macro event calendar from EODHD")
    _add_common_args(p_cal)
    p_cal.add_argument("--country", action="append", dest="countries", metavar="CC",
                       help="ISO alpha-2 country codes (e.g. --country US --country GB). Default: US.")
    p_cal.add_argument("--high-impact-only", action="store_true",
                       help="Only store High-impact events")
    p_cal.add_argument("--budget", type=int, default=None,
                       help="Max EODHD API credits (default: unlimited)")

    # -- event-ticks --
    p_evt = sub.add_parser("event-ticks",
                           help="Tick-by-tick: only around macro events (CPI, NFP, FOMC, ...)")
    _add_common_args(p_evt)
    _add_tick_source_args(p_evt)
    p_evt.add_argument("--category", action="append", dest="categories", metavar="CAT",
                       help="Event categories to window around (e.g. --category CPI --category FOMC)")
    p_evt.add_argument("--before", type=float, default=72.0,
                       help="Hours before event to start window (default: 72)")
    p_evt.add_argument("--after", type=float, default=72.0,
                       help="Hours after event to end window (default: 72)")

    # -- all --
    p_all = sub.add_parser("all", help="Run eodhd + telonex bar pipelines sequentially")
    _add_common_args(p_all)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    _setup_logging(data_dir, verbose=args.verbose)

    dispatch = {
        "telonex": _cmd_telonex,
        "eodhd": _cmd_eodhd,
        "kalshi": _cmd_kalshi,
        "ticks": _cmd_ticks,
        "calendar": _cmd_calendar,
        "event-ticks": _cmd_event_ticks,
        "all": _cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
