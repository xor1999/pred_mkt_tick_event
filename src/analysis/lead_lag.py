"""Lead-lag detection between prediction markets and equities.

Three approaches, from most to least sophisticated:

1. **Path signatures** (primary): form a 2D path from aligned
   (pred_mkt_probability, equity_return) series around each macro event.
   The level-2 signature cross-terms (signed area) directly encode
   lead-lag directionality.  Higher levels capture nonlinear patterns.

2. **Cross-correlation**: standard lagged correlation at 1-min resolution.
   Quick baseline but assumes linearity and fixed lag.

3. **Granger causality**: VAR-based test for whether past pred_mkt values
   help predict future equity returns (and vice versa).

All functions expect aligned, regularly-sampled DataFrames with a
DatetimeIndex at 1-min (or finer) resolution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Path signature lead-lag
# ---------------------------------------------------------------------------

def compute_path_signature(
    pred_mkt: pd.Series,
    equity: pd.Series,
    depth: int = 4,
    normalise: bool = True,
) -> np.ndarray:
    """Compute the truncated path signature of the 2D path (pred_mkt, equity).

    Parameters
    ----------
    pred_mkt : prediction market probability series (aligned, no NaNs).
    equity : equity return or price series (same index as pred_mkt).
    depth : truncation depth (2 = signed area only, 3+ = higher order).
    normalise : if True, rescale each channel to [0, 1] before computing.

    Returns
    -------
    1D array of signature coefficients.  For a 2D path at depth K, the
    signature has sum_{k=1}^{K} 2^k terms.
        depth=2: 6 terms  (2 level-1 + 4 level-2)
        depth=3: 14 terms (+ 8 level-3)
        depth=4: 30 terms (+ 16 level-4)

    The level-2 cross-terms (indices 2,3 for a 2D path) encode the
    signed area -- the core lead-lag signal.
    """
    try:
        import esig
    except ImportError:
        raise ImportError(
            "The `esig` package is required for path signatures. "
            "Install it with: pip install esig"
        )

    # Build the 2D path as (n_steps, 2) array
    x = pred_mkt.values.astype(np.float64)
    y = equity.values.astype(np.float64)

    if normalise:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)

    path = np.column_stack([x, y])
    sig = esig.stream2sig(path, depth)
    return sig


def signed_area(pred_mkt: pd.Series, equity: pd.Series) -> float:
    """Compute the signed area between two aligned series.

    This is the level-2 cross-term of the path signature, but computed
    directly via the trapezoidal rule for speed.

    Positive signed area => pred_mkt leads (moves before equity).
    Negative signed area => equity leads.
    Magnitude => strength of the lead.

    This is the single most important number for the thesis.
    """
    x = pred_mkt.values.astype(np.float64)
    y = equity.values.astype(np.float64)

    # Normalise to [0, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min)

    # Signed area = 0.5 * sum(x_i * dy_{i+1} - y_i * dx_{i+1})
    dx = np.diff(x)
    dy = np.diff(y)
    area = 0.5 * np.sum(x[:-1] * dy - y[:-1] * dx)
    return float(area)


def signature_lead_lag_single_event(
    pred_mkt: pd.Series,
    equity: pd.Series,
    depth: int = 4,
) -> dict[str, Any]:
    """Analyse lead-lag for a single event window using path signatures.

    Parameters
    ----------
    pred_mkt : prediction market probability, aligned 1-min index.
    equity : equity price or return, same index.
    depth : signature truncation depth.

    Returns
    -------
    Dict with:
        signed_area     : float -- positive = pred_mkt leads
        signature       : np.ndarray -- full truncated signature
        n_obs           : int -- number of aligned observations
    """
    # Drop NaNs from both, keeping alignment
    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()
    if len(aligned) < 10:
        return {"signed_area": np.nan, "signature": np.array([]), "n_obs": len(aligned)}

    sa = signed_area(aligned["pm"], aligned["eq"])
    sig = compute_path_signature(aligned["pm"], aligned["eq"], depth=depth)

    return {
        "signed_area": sa,
        "signature": sig,
        "n_obs": len(aligned),
    }


def signature_lead_lag_multi_event(
    events: pd.DataFrame,
    pred_mkt_bars: dict[str, pd.DataFrame],
    equity_bars: dict[str, pd.DataFrame],
    pm_slug: str,
    equity_ticker: str,
    before: pd.Timedelta = pd.Timedelta("72h"),
    after: pd.Timedelta = pd.Timedelta("72h"),
    depth: int = 4,
) -> pd.DataFrame:
    """Compute lead-lag signatures across multiple macro events.

    Parameters
    ----------
    events : DataFrame with at least a 'date' column (UTC datetime).
    pred_mkt_bars : dict mapping slug to DataFrame with 'close' column.
    equity_bars : dict mapping ticker to DataFrame with 'close' column.
    pm_slug : which prediction market slug to use.
    equity_ticker : which equity ticker to use.
    before / after : window around each event.
    depth : signature truncation depth.

    Returns
    -------
    DataFrame with one row per event, columns:
        event_date, event_type, signed_area, signature, n_obs
    """
    if pm_slug not in pred_mkt_bars or equity_ticker not in equity_bars:
        logger.warning("Missing data for %s or %s", pm_slug, equity_ticker)
        return pd.DataFrame()

    pm_df = pred_mkt_bars[pm_slug]
    eq_df = equity_bars[equity_ticker]

    results = []
    for _, event in events.iterrows():
        event_dt = pd.to_datetime(event["date"], utc=True)
        window_start = event_dt - before
        window_end = event_dt + after

        pm_window = pm_df.loc[window_start:window_end, "close"]
        eq_window = eq_df.loc[window_start:window_end, "close"]

        result = signature_lead_lag_single_event(pm_window, eq_window, depth=depth)
        result["event_date"] = event_dt
        result["event_type"] = event.get("type", "")
        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Cross-correlation baseline
# ---------------------------------------------------------------------------

def lagged_cross_correlation(
    pred_mkt: pd.Series,
    equity: pd.Series,
    max_lag: int = 60,
) -> pd.Series:
    """Compute cross-correlation at lags from -max_lag to +max_lag.

    Positive lags = pred_mkt leads equity by that many bars.
    The lag with the highest absolute correlation indicates the lead-lag.

    Parameters
    ----------
    pred_mkt : prediction market returns or price changes.
    equity : equity returns or price changes.
    max_lag : maximum lag in bars (e.g., 60 = +/- 60 minutes).

    Returns
    -------
    Series indexed by lag, values are correlation coefficients.
    """
    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()
    pm = aligned["pm"].values
    eq = aligned["eq"].values

    # Standardise
    pm = (pm - pm.mean()) / (pm.std() + 1e-12)
    eq = (eq - eq.mean()) / (eq.std() + 1e-12)

    n = len(pm)
    lags = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag >= 0:
            c = np.mean(pm[:n - lag] * eq[lag:]) if lag < n else 0.0
        else:
            c = np.mean(pm[-lag:] * eq[:n + lag]) if -lag < n else 0.0
        corrs.append(c)

    return pd.Series(corrs, index=list(lags), name="cross_corr")


def optimal_lag(
    pred_mkt: pd.Series,
    equity: pd.Series,
    max_lag: int = 60,
) -> dict[str, Any]:
    """Find the lag that maximises cross-correlation.

    Returns dict with:
        optimal_lag : int -- positive = pred_mkt leads
        max_corr    : float -- correlation at that lag
        corr_series : pd.Series -- full correlation function
    """
    corr = lagged_cross_correlation(pred_mkt, equity, max_lag)
    best_lag = int(corr.abs().idxmax())
    return {
        "optimal_lag": best_lag,
        "max_corr": float(corr.loc[best_lag]),
        "corr_series": corr,
    }


# ---------------------------------------------------------------------------
# 3. Granger causality baseline
# ---------------------------------------------------------------------------

def granger_causality(
    pred_mkt: pd.Series,
    equity: pd.Series,
    max_lag: int = 10,
) -> dict[str, Any]:
    """Pairwise Granger causality test in both directions.

    Tests:
      1. Does past pred_mkt help predict future equity? (lead)
      2. Does past equity help predict future pred_mkt? (lag)

    Uses statsmodels grangercausalitytests.

    Returns dict with:
        pm_causes_eq : dict -- {lag: (F-stat, p-value)} for each tested lag
        eq_causes_pm : dict -- same, reversed direction
        best_lag_pm_eq : int -- lag with lowest p-value for pm -> eq
        best_pval_pm_eq : float
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        raise ImportError(
            "The `statsmodels` package is required for Granger causality. "
            "Install it with: pip install statsmodels"
        )

    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()
    if len(aligned) < max_lag * 3:
        return {
            "pm_causes_eq": {}, "eq_causes_pm": {},
            "best_lag_pm_eq": np.nan, "best_pval_pm_eq": np.nan,
        }

    # statsmodels expects [effect, cause] column order
    # Test: pm -> eq (does pm Granger-cause eq?)
    pm_eq_results = grangercausalitytests(
        aligned[["eq", "pm"]], maxlag=max_lag, verbose=False,
    )
    pm_eq = {
        lag: (
            round(pm_eq_results[lag][0]["ssr_ftest"][0], 4),
            round(pm_eq_results[lag][0]["ssr_ftest"][1], 6),
        )
        for lag in pm_eq_results
    }

    # Test: eq -> pm
    eq_pm_results = grangercausalitytests(
        aligned[["pm", "eq"]], maxlag=max_lag, verbose=False,
    )
    eq_pm = {
        lag: (
            round(eq_pm_results[lag][0]["ssr_ftest"][0], 4),
            round(eq_pm_results[lag][0]["ssr_ftest"][1], 6),
        )
        for lag in eq_pm_results
    }

    # Find best lag for pm -> eq
    best_lag = min(pm_eq, key=lambda k: pm_eq[k][1])
    best_pval = pm_eq[best_lag][1]

    return {
        "pm_causes_eq": pm_eq,
        "eq_causes_pm": eq_pm,
        "best_lag_pm_eq": best_lag,
        "best_pval_pm_eq": best_pval,
    }


# ---------------------------------------------------------------------------
# 4. Convenience: run all methods on one event window
# ---------------------------------------------------------------------------

def analyse_event_window(
    pred_mkt: pd.Series,
    equity: pd.Series,
    sig_depth: int = 4,
    xcorr_max_lag: int = 60,
    granger_max_lag: int = 10,
) -> dict[str, Any]:
    """Run all lead-lag methods on a single event window.

    Parameters
    ----------
    pred_mkt : prediction market close prices (1-min, aligned).
    equity : equity close prices (1-min, aligned).

    Returns
    -------
    Dict with results from all three methods.
    """
    results: dict[str, Any] = {}

    # Signed area (pure numpy -- always available)
    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()
    results["n_obs"] = len(aligned)
    if len(aligned) >= 10:
        results["signed_area"] = signed_area(aligned["pm"], aligned["eq"])
    else:
        results["signed_area"] = np.nan

    # Path signature (requires esig -- optional)
    try:
        sig = compute_path_signature(aligned["pm"], aligned["eq"], depth=sig_depth)
        results["signature"] = sig
    except ImportError:
        results["signature"] = np.array([])
    except Exception as exc:
        logger.debug("Path signature failed: %s", exc)
        results["signature"] = np.array([])

    # Cross-correlation (pure numpy -- always available)
    pm_ret = pred_mkt.diff().dropna()
    eq_ret = equity.pct_change().dropna()
    xcorr = optimal_lag(pm_ret, eq_ret, max_lag=xcorr_max_lag)
    results["xcorr_optimal_lag"] = xcorr["optimal_lag"]
    results["xcorr_max_corr"] = xcorr["max_corr"]

    # Granger causality (requires statsmodels -- optional)
    try:
        gc = granger_causality(pm_ret, eq_ret, max_lag=granger_max_lag)
        results["granger_best_lag"] = gc["best_lag_pm_eq"]
        results["granger_best_pval"] = gc["best_pval_pm_eq"]
    except ImportError:
        results["granger_best_lag"] = np.nan
        results["granger_best_pval"] = np.nan
    except Exception as exc:
        logger.warning("Granger causality failed: %s", exc)
        results["granger_best_lag"] = np.nan
        results["granger_best_pval"] = np.nan

    return results
