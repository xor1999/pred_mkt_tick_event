"""EODHD API client -- fetches equities intraday and daily OHLCV.

Adapted for 1-minute intraday bars with rate limiting, retry logic,
and budget tracking.  The EODHD intraday endpoint returns at most
~120 days of 1-min data per request, so the ingestion pipeline chunks
the 3-year window accordingly.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from io import StringIO
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as cfg

logger = logging.getLogger(__name__)

BASE_URL = "https://eodhd.com/api"

EXCHANGES: dict[str, str] = {
    "US":    "NYSE + NASDAQ + ARCA + AMEX",
    "LSE":   "London Stock Exchange",
    "XETRA": "XETRA (Germany primary)",
    "PA":    "Euronext Paris",
    "AS":    "Euronext Amsterdam",
    "BR":    "Euronext Brussels",
    "LS":    "Euronext Lisbon",
    "SW":    "SIX Swiss Exchange",
    "VI":    "Vienna Exchange",
    "HE":    "Helsinki (Nasdaq Nordic)",
    "ST":    "Stockholm (Nasdaq Nordic)",
    "CO":    "Copenhagen (Nasdaq Nordic)",
    "OL":    "Oslo Bors",
    "IC":    "Iceland Exchange",
}


class BudgetExhausted(Exception):
    """Raised when the API call budget for this run is used up."""

    def __init__(self, calls_made: int, budget: int):
        self.calls_made = calls_made
        self.budget = budget
        super().__init__(
            f"API budget exhausted: {calls_made}/{budget} calls used. "
            "Re-run to continue from where you left off."
        )


class EODHDClient:
    """Thin wrapper around the EODHD REST API with rate limiting and budget tracking."""

    ENDPOINT_COSTS: dict[str, int] = {
        "exchange-symbol-list": 1,
        "eod-bulk-last-day": 1,
        "fundamentals": 10,
        "eod": 1,
        "screener": 5,
        "intraday": 5,
        "ticks": 10,
        "economic-events": 1,
    }

    def __init__(
        self,
        api_key: str | None = None,
        calls_per_sec: float = 5.0,
        calls_per_min: int = 1000,
        budget: int | None = None,
    ):
        self.api_key = api_key or cfg.EODHD_API_KEY
        if not self.api_key:
            raise ValueError(
                "EODHD_API_KEY not set. Add it to your .env file."
            )
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._min_interval = 1.0 / calls_per_sec
        self._calls_per_min = calls_per_min
        self._last_ts = 0.0
        self._calls_made = 0
        self._budget = budget
        self._lock = threading.Lock()
        self._call_window: deque[tuple[float, int]] = deque()

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def remaining(self) -> int | None:
        if self._budget is None:
            return None
        return max(0, self._budget - self._calls_made)

    def _cost_for_endpoint(self, endpoint: str) -> int:
        base = endpoint.split("/")[0]
        return self.ENDPOINT_COSTS.get(base, 1)

    def _throttle(self, cost: int = 1) -> None:
        now = time.time()
        elapsed = now - self._last_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
            now = time.time()

        cutoff = now - 60.0
        while self._call_window and self._call_window[0][0] <= cutoff:
            self._call_window.popleft()

        window_total = sum(c for _, c in self._call_window)
        while window_total + cost > self._calls_per_min:
            oldest_ts = self._call_window[0][0]
            sleep_time = oldest_ts - cutoff + 0.1
            if sleep_time > 0:
                logger.debug(
                    "Rate limit: %d + %d > %d/min -- sleeping %.1fs",
                    window_total, cost, self._calls_per_min, sleep_time,
                )
                time.sleep(sleep_time)
            now = time.time()
            cutoff = now - 60.0
            while self._call_window and self._call_window[0][0] <= cutoff:
                self._call_window.popleft()
            window_total = sum(c for _, c in self._call_window)

    def _get(
        self, endpoint: str, params: dict | None = None, cost: int | None = None,
    ) -> requests.Response:
        with self._lock:
            if cost is None:
                cost = self._cost_for_endpoint(endpoint)
            if self._budget is not None and self._calls_made + cost > self._budget:
                raise BudgetExhausted(self._calls_made, self._budget)
            self._throttle(cost)
            params = params or {}
            params["api_token"] = self.api_key
            url = f"{BASE_URL}/{endpoint}"
            resp = self.session.get(url, params=params, timeout=60)
            now = time.time()
            self._last_ts = now
            self._calls_made += cost
            self._call_window.append((now, cost))
            resp.raise_for_status()
            return resp

    def _get_json(self, endpoint: str, params: dict | None = None) -> Any:
        params = params or {}
        params["fmt"] = "json"
        return self._get(endpoint, params).json()

    def _get_csv(self, endpoint: str, params: dict | None = None) -> pd.DataFrame:
        params = params or {}
        params["fmt"] = "csv"
        resp = self._get(endpoint, params)
        text = resp.text.strip()
        if not text:
            return pd.DataFrame()
        return pd.read_csv(StringIO(text))

    # -- bulk EOD (last trading day) for liquidity screening ---------------

    def get_bulk_eod(self, exchange: str, date: str | None = None) -> pd.DataFrame:
        """Bulk last-day EOD prices for all tickers on *exchange*."""
        params: dict = {}
        if date:
            params["date"] = date
        return self._get_csv(f"eod-bulk-last-day/{exchange}", params)

    # -- exchange symbols -------------------------------------------------

    def get_exchange_symbols(self, exchange: str) -> pd.DataFrame:
        """Return all instruments listed on *exchange*."""
        df = self._get_csv(f"exchange-symbol-list/{exchange}")
        if df.empty:
            return df
        df["exchange_code"] = exchange
        return df

    # -- intraday OHLCV ---------------------------------------------------

    def get_intraday(
        self,
        ticker: str,
        exchange: str,
        from_ts: int,
        to_ts: int,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Fetch intraday bars between Unix timestamps *from_ts* .. *to_ts*.

        EODHD returns at most ~120 calendar days of 1-min data per request.
        The caller is responsible for chunking longer windows.
        """
        params: dict = {"interval": interval, "from": from_ts, "to": to_ts}
        df = self._get_csv(f"intraday/{ticker}.{exchange}", params)
        if not df.empty:
            df.columns = df.columns.str.lower()
        return df

    # -- tick data (US stocks only) ----------------------------------------

    def get_ticks(
        self,
        ticker: str,
        from_ts: int,
        to_ts: int,
        limit: int = 0,
    ) -> list[dict]:
        """Fetch tick-by-tick trade data for a US stock.

        Parameters
        ----------
        ticker : symbol name (e.g. "AAPL" or "AAPL.US").
        from_ts / to_ts : Unix timestamps in seconds (UTC).
        limit : max ticks to return (0 = no limit).

        Returns a list of tick dicts with keys:
            ts (ms timestamp), price, shares, mkt, sub_mkt, sl, seq.

        EODHD returns columnar JSON (arrays per field); this method
        unpacks it into a list of row-dicts for consistency.
        """
        sym = ticker.split(".")[0]
        params: dict[str, Any] = {"s": sym, "from": from_ts, "to": to_ts}
        if limit > 0:
            params["limit"] = limit
        data = self._get_json("ticks", params)

        if not isinstance(data, dict) or "ts" not in data:
            return []

        n = len(data["ts"])
        if n == 0:
            return []

        fields = list(data.keys())
        rows: list[dict] = []
        for i in range(n):
            rows.append({f: data[f][i] for f in fields})
        return rows

    # -- historical daily EOD (per ticker) --------------------------------

    def get_eod_historical(
        self,
        ticker: str,
        exchange: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV history for a single ticker."""
        params: dict = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get_csv(f"eod/{ticker}.{exchange}", params)

    # -- screener (market-cap filtering) ----------------------------------

    def get_screener(
        self,
        filters: list,
        sort: str = "market_capitalization.desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        import json as _json
        params = {
            "filters": _json.dumps(filters),
            "sort": sort,
            "limit": limit,
            "offset": offset,
        }
        data = self._get_json("screener", params)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        return []

    # -- economic events ---------------------------------------------------

    def get_economic_events(
        self,
        country: str,
        from_date: str,
        to_date: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch economic events for a country (ISO-3166 alpha-2 code)."""
        params: dict = {
            "country": country,
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "offset": offset,
        }
        data = self._get_json("economic-events", params)
        if isinstance(data, list):
            return data
        return []

    # -- API usage --------------------------------------------------------

    def get_api_usage(self) -> dict:
        return self._get_json("user")
