"""Topological data analysis for regime detection in lead-lag dynamics.

Uses persistent homology on sliding windows of the joint
(pred_mkt, equity) time series to detect structural changes in
the lead-lag relationship.

Core idea: the "shape" of the 2D point cloud (pred_mkt_return,
equity_return) changes between regimes.  In a strong lead-lag regime,
the cloud is elongated along an off-diagonal direction (pred moves,
then equity follows -- creating a systematic loop).  When the lead-lag
breaks down, this structure collapses.

Persistent homology captures these shapes as persistence diagrams:
  - H0 (connected components): clustering structure in returns.
  - H1 (loops): cyclic patterns = lead-lag loops.

Changes in persistence diagrams across sliding windows signal regime
shifts -- moments when the lead-lag either strengthens or disappears.

Requires: giotto-tda (pip install giotto-tda)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_giotto():
    try:
        import gtda
        return gtda
    except ImportError:
        raise ImportError(
            "The `giotto-tda` package is required for TDA. "
            "Install it with: pip install giotto-tda"
        )


# ---------------------------------------------------------------------------
# 1. Time-delay embedding
# ---------------------------------------------------------------------------

def time_delay_embedding(
    series: pd.Series,
    dimension: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Embed a 1D time series into higher dimensions via Takens embedding.

    Parameters
    ----------
    series : 1D time series.
    dimension : embedding dimension (default 3).
    delay : time delay in steps (default 1 = consecutive).

    Returns
    -------
    (n_points, dimension) array of embedded points.
    """
    x = series.values.astype(np.float64)
    n = len(x)
    max_idx = n - (dimension - 1) * delay
    if max_idx <= 0:
        return np.empty((0, dimension))

    indices = np.arange(max_idx)
    embedded = np.column_stack([x[indices + i * delay] for i in range(dimension)])
    return embedded


def joint_point_cloud(
    pred_mkt: pd.Series,
    equity: pd.Series,
    use_returns: bool = True,
) -> np.ndarray:
    """Build a 2D point cloud from aligned pred_mkt and equity series.

    Parameters
    ----------
    pred_mkt : prediction market probability series.
    equity : equity price series.
    use_returns : if True, convert to returns/diffs first.

    Returns
    -------
    (n_points, 2) array.
    """
    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()

    if use_returns:
        pm = aligned["pm"].diff().dropna().values
        eq = aligned["eq"].pct_change().dropna().values
        n = min(len(pm), len(eq))
        pm, eq = pm[:n], eq[:n]
    else:
        pm = aligned["pm"].values
        eq = aligned["eq"].values

    return np.column_stack([pm, eq])


# ---------------------------------------------------------------------------
# 2. Persistent homology
# ---------------------------------------------------------------------------

def compute_persistence(
    point_cloud: np.ndarray,
    max_dimension: int = 1,
) -> np.ndarray:
    """Compute the persistence diagram of a point cloud.

    Parameters
    ----------
    point_cloud : (n_points, n_features) array.
    max_dimension : max homology dimension (0=components, 1=loops).

    Returns
    -------
    Persistence diagram as (n_features, 3) array where columns are
    (birth, death, dimension).
    """
    _check_giotto()
    from gtda.homology import VietorisRipsPersistence

    # giotto expects (n_samples, n_points, n_features) for fit_transform
    # but for a single point cloud we reshape
    vr = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dimension + 1)),
        n_jobs=-1,
    )
    diagrams = vr.fit_transform(point_cloud[np.newaxis, :, :])
    return diagrams[0]  # single diagram


def persistence_summary(diagram: np.ndarray) -> dict[str, Any]:
    """Extract summary statistics from a persistence diagram.

    Returns
    -------
    Dict with:
        n_h0 : number of H0 features (connected components)
        n_h1 : number of H1 features (loops)
        max_persistence_h0 : longest-lived component
        max_persistence_h1 : longest-lived loop (the lead-lag signal)
        total_persistence_h1 : sum of all H1 lifetimes
        mean_persistence_h1 : average H1 lifetime
    """
    lifetimes = diagram[:, 1] - diagram[:, 0]
    dims = diagram[:, 2]

    h0_mask = dims == 0
    h1_mask = dims == 1

    h0_life = lifetimes[h0_mask]
    h1_life = lifetimes[h1_mask]

    # Filter out infinite persistence (the single connected component)
    h0_life_finite = h0_life[np.isfinite(h0_life)]
    h1_life_finite = h1_life[np.isfinite(h1_life)]

    return {
        "n_h0": int(h0_mask.sum()),
        "n_h1": int(h1_mask.sum()),
        "max_persistence_h0": float(h0_life_finite.max()) if len(h0_life_finite) > 0 else 0.0,
        "max_persistence_h1": float(h1_life_finite.max()) if len(h1_life_finite) > 0 else 0.0,
        "total_persistence_h1": float(h1_life_finite.sum()),
        "mean_persistence_h1": float(h1_life_finite.mean()) if len(h1_life_finite) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Sliding window persistence for regime detection
# ---------------------------------------------------------------------------

def sliding_window_persistence(
    pred_mkt: pd.Series,
    equity: pd.Series,
    window_size: int = 120,
    step_size: int = 30,
    max_dimension: int = 1,
    use_returns: bool = True,
) -> pd.DataFrame:
    """Compute persistence summaries over sliding windows.

    This tracks how the topological structure of the (pred_mkt, equity)
    joint dynamics evolves over time.  A spike in H1 persistence
    indicates a period with strong cyclic (lead-lag) structure.

    Parameters
    ----------
    pred_mkt : prediction market probability series (1-min).
    equity : equity price series (1-min).
    window_size : window size in bars (default 120 = 2 hours).
    step_size : step between windows in bars (default 30 = 30 min).
    max_dimension : max homology dimension.
    use_returns : convert to returns before embedding.

    Returns
    -------
    DataFrame indexed by window center timestamp, with columns:
        n_h0, n_h1, max_persistence_h0, max_persistence_h1,
        total_persistence_h1, mean_persistence_h1
    """
    aligned = pd.DataFrame({"pm": pred_mkt, "eq": equity}).dropna()
    n = len(aligned)
    if n < window_size:
        return pd.DataFrame()

    results = []
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = aligned.iloc[start:end]
        center_time = window.index[window_size // 2]

        cloud = joint_point_cloud(window["pm"], window["eq"], use_returns=use_returns)
        if len(cloud) < 10:
            continue

        try:
            diagram = compute_persistence(cloud, max_dimension=max_dimension)
            summary = persistence_summary(diagram)
            summary["timestamp"] = center_time
            results.append(summary)
        except Exception as exc:
            logger.debug("Window at %s failed: %s", center_time, exc)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# 4. Regime detection from persistence time series
# ---------------------------------------------------------------------------

def detect_regime_changes(
    persistence_ts: pd.DataFrame,
    h1_col: str = "max_persistence_h1",
    threshold_std: float = 2.0,
) -> pd.DataFrame:
    """Detect regime changes from a sliding-window persistence time series.

    A regime change is flagged when the H1 persistence (loop strength)
    deviates by more than `threshold_std` standard deviations from
    its rolling mean.

    Parameters
    ----------
    persistence_ts : output of sliding_window_persistence().
    h1_col : which H1 metric to use for detection.
    threshold_std : number of standard deviations for threshold.

    Returns
    -------
    DataFrame with columns:
        timestamp, h1_value, rolling_mean, rolling_std, z_score, is_regime_change
    """
    if persistence_ts.empty or h1_col not in persistence_ts.columns:
        return pd.DataFrame()

    h1 = persistence_ts[h1_col].copy()

    # Rolling stats (use expanding for first windows)
    rolling_mean = h1.expanding(min_periods=3).mean()
    rolling_std = h1.expanding(min_periods=3).std()

    z_score = (h1 - rolling_mean) / (rolling_std + 1e-12)

    result = pd.DataFrame({
        "h1_value": h1,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "z_score": z_score,
        "is_regime_change": z_score.abs() > threshold_std,
    })

    return result


# ---------------------------------------------------------------------------
# 5. Convenience: full TDA pipeline for one event window
# ---------------------------------------------------------------------------

def analyse_event_window_tda(
    pred_mkt: pd.Series,
    equity: pd.Series,
    window_size: int = 120,
    step_size: int = 30,
) -> dict[str, Any]:
    """Run TDA analysis on a single event window.

    Returns
    -------
    Dict with:
        persistence_ts : DataFrame of sliding-window persistence summaries
        regime_changes : DataFrame of detected regime change points
        overall_h1     : persistence summary for the entire window
    """
    # Overall persistence for the full window
    cloud = joint_point_cloud(pred_mkt, equity, use_returns=True)
    overall = {}
    if len(cloud) >= 10:
        try:
            diagram = compute_persistence(cloud, max_dimension=1)
            overall = persistence_summary(diagram)
        except Exception as exc:
            logger.warning("Overall persistence failed: %s", exc)

    # Sliding window
    pers_ts = sliding_window_persistence(
        pred_mkt, equity,
        window_size=window_size, step_size=step_size,
    )

    # Regime detection
    regime = detect_regime_changes(pers_ts) if not pers_ts.empty else pd.DataFrame()

    return {
        "persistence_ts": pers_ts,
        "regime_changes": regime,
        "overall_h1": overall,
    }
