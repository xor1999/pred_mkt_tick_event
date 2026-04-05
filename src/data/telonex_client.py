"""Telonex SDK wrapper -- fetches prediction-market fill data and market catalog.

The Telonex SDK handles its own concurrency (default 5 parallel downloads),
file caching (skips already-downloaded files), and chunking internally.
We avoid adding any artificial throttling on top.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


def _get_sdk():
    """Import and return the telonex SDK module (lazy)."""
    try:
        import telonex
        return telonex
    except ImportError as exc:
        raise ImportError(
            "The `telonex` package is required for data ingestion. "
            "Install it with: pip install telonex"
        ) from exc


class TelonexClient:
    """Wrapper around the Telonex Python SDK.

    No artificial rate limiting -- the SDK manages concurrency internally
    (default 5 parallel downloads) and caches files on disk.

    Provides:
    - ``list_markets()``  -- catalog of available prediction markets.
    - ``get_fills()``     -- raw fill-level DataFrame for a market/outcome.
    - ``get_ohlcv_1m()``  -- convenience: fetch fills and resample to 1-min bars.
    """

    def __init__(
        self,
        api_key: str | None = None,
        exchange: str = "polymarket",
        download_dir: str = "./data/raw/telonex/_cache",
    ):
        self.api_key = api_key or cfg.TELONEX_API_KEY
        self.exchange = exchange
        self.download_dir = download_dir
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    # -- SDK calls --------------------------------------------------------

    def list_markets(self, download_dir: str | None = None) -> pd.DataFrame:
        """Download the full market catalog and return as a DataFrame.

        The Telonex SDK downloads a large Parquet file (~872k markets).
        We read it with pyarrow column-selectively to avoid OOM.
        """
        sdk = _get_sdk()
        import pyarrow.parquet as pq

        dl_dir = download_dir or self.download_dir
        logger.info("Downloading Polymarket catalog via Telonex SDK ...")
        path = sdk.download_markets(
            exchange=self.exchange,
            download_dir=dl_dir,
            verbose=False,
        )
        self._calls_made += 1

        # Read only the columns we need (full file is too large for pandas)
        keep_cols = [
            "slug", "event_title", "question", "category",
            "outcome_0", "outcome_1",
            "onchain_fills_from", "onchain_fills_to",
        ]
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in keep_cols if c in available]
        df = pf.read(columns=cols).to_pandas()

        logger.info("Catalog: %d total markets", len(df))
        return df

    def get_fills(
        self,
        slug: str,
        outcome: str,
        from_date: str,
        to_date: str,
        concurrency: int = 20,
    ) -> pd.DataFrame:
        """Fetch fill-level data for a single market outcome over the full date range.

        Uses the SDK's download() directly to control concurrency.
        Downloaded parquet files are cached in download_dir.

        Returns DataFrame with DatetimeIndex and columns [price, amount].
        """
        sdk = _get_sdk()
        import pyarrow.parquet as pq

        try:
            files = sdk.download(
                api_key=self.api_key,
                exchange=self.exchange,
                channel="onchain_fills",
                slug=slug,
                outcome=outcome,
                from_date=from_date,
                to_date=to_date,
                download_dir=self.download_dir,
                concurrency=concurrency,
                verbose=False,
            )
        except Exception:
            files = []
        finally:
            self._calls_made += 1

        if not files:
            return pd.DataFrame(columns=["timestamp", "price", "amount"])

        # Read and concatenate all downloaded parquet files
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                pass

        if not frames:
            return pd.DataFrame(columns=["timestamp", "price", "amount"])

        df = pd.concat(frames, ignore_index=True)

        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "amount"])

        if "block_timestamp_us" in df.columns:
            df["timestamp"] = pd.to_datetime(df["block_timestamp_us"], unit="us", utc=True)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            raise KeyError(
                f"No recognised timestamp column in Telonex response. "
                f"Columns: {list(df.columns)}"
            )

        # Cast price/amount from string to numeric (Telonex stores as object)
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.sort_values("timestamp").set_index("timestamp")
        keep = [c for c in ["price", "amount"] if c in df.columns]
        return df[keep]

    def get_ohlcv_1m(
        self,
        slug: str,
        outcome: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Fetch fills and resample into 1-minute OHLCV bars."""
        fills = self.get_fills(slug, outcome, from_date, to_date)
        if fills.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "trade_count"]
            )
        return resample_fills_to_1m(fills)


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def resample_fills_to_1m(fills: pd.DataFrame) -> pd.DataFrame:
    """Resample tick-level fills into 1-minute OHLCV bars."""
    price = fills["price"]
    amount = fills["amount"] if "amount" in fills.columns else pd.Series(0, index=fills.index)

    ohlcv = pd.DataFrame({
        "open": price.resample("1min").first(),
        "high": price.resample("1min").max(),
        "low": price.resample("1min").min(),
        "close": price.resample("1min").last(),
        "volume": amount.resample("1min").sum(),
        "trade_count": price.resample("1min").count(),
    })

    ohlcv["close"] = ohlcv["close"].ffill()
    for col in ("open", "high", "low"):
        ohlcv[col] = ohlcv[col].fillna(ohlcv["close"])
    ohlcv["volume"] = ohlcv["volume"].fillna(0)
    ohlcv["trade_count"] = ohlcv["trade_count"].fillna(0).astype(int)

    return ohlcv
