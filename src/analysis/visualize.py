"""Visualization module for lead-lag analysis results.

Generates publication-quality figures and summary tables for:
  1. Event study plots (avg pred mkt vs equity around macro events)
  2. Signed area distributions (histogram of lead-lag direction)
  3. Cross-correlation functions (lag structure)
  4. Lead-lag summary tables by event type
  5. TDA persistence diagrams and regime timelines

All figures are saved to `results/figures/` and tables to `results/tables/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


# ---------------------------------------------------------------------------
# 1. Event study: average price paths around macro events
# ---------------------------------------------------------------------------

def plot_event_study(
    pm_windows: dict[str, pd.DataFrame],
    eq_windows: dict[str, pd.DataFrame],
    event_type: str = "CPI",
    minutes_before: int = 120,
    minutes_after: int = 240,
    save: bool = True,
) -> Any:
    """Plot average pred mkt probability and equity return around events.

    Parameters
    ----------
    pm_windows : dict mapping event_id to DataFrame with 'close' column,
                 indexed relative to event time (minutes).
    eq_windows : same for equity, with returns or normalised prices.
    event_type : label for the title.

    Returns the matplotlib figure.
    """
    plt = _get_plt()

    # Average across events
    all_pm = pd.DataFrame({k: v["close"] for k, v in pm_windows.items()})
    all_eq = pd.DataFrame({k: v["close"] for k, v in eq_windows.items()})

    pm_mean = all_pm.mean(axis=1)
    pm_std = all_pm.std(axis=1)
    eq_mean = all_eq.mean(axis=1)
    eq_std = all_eq.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Pred market
    ax1.plot(pm_mean.index, pm_mean, color="#1f77b4", linewidth=2, label="Mean probability")
    ax1.fill_between(pm_mean.index, pm_mean - pm_std, pm_mean + pm_std,
                     alpha=0.2, color="#1f77b4")
    ax1.axvline(0, color="red", linestyle="--", alpha=0.7, label="Event release")
    ax1.set_ylabel("Prediction Market Probability")
    ax1.set_title(f"Event Study: {event_type} -- Prediction Market vs Equity Response")
    ax1.legend(loc="upper left")

    # Equity
    ax2.plot(eq_mean.index, eq_mean, color="#2ca02c", linewidth=2, label="Mean return")
    ax2.fill_between(eq_mean.index, eq_mean - eq_std, eq_mean + eq_std,
                     alpha=0.2, color="#2ca02c")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.7, label="Event release")
    ax2.set_ylabel("Equity Cumulative Return")
    ax2.set_xlabel("Minutes Relative to Event")
    ax2.legend(loc="upper left")

    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / f"event_study_{event_type.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# 2. Signed area distribution
# ---------------------------------------------------------------------------

def plot_signed_area_distribution(
    results: pd.DataFrame,
    group_col: str = "event_type",
    save: bool = True,
) -> Any:
    """Histogram of signed areas, optionally grouped by event type.

    Positive = pred mkt leads equity.  A distribution shifted right
    confirms the thesis.

    Parameters
    ----------
    results : DataFrame with columns 'signed_area' and optionally group_col.
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_col in results.columns:
        groups = results.groupby(group_col)["signed_area"]
        for name, data in groups:
            data.dropna().hist(ax=ax, alpha=0.5, bins=30, label=name)
        ax.legend(title="Event Type")
    else:
        results["signed_area"].dropna().hist(ax=ax, bins=40, alpha=0.7, color="#1f77b4")

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Signed Area (positive = prediction market leads)")
    ax.set_ylabel("Count")
    ax.set_title("Lead-Lag Direction: Path Signature Signed Area Distribution")

    # Add annotation with stats
    sa = results["signed_area"].dropna()
    stats_text = (
        f"n = {len(sa)}\n"
        f"mean = {sa.mean():.4f}\n"
        f"median = {sa.median():.4f}\n"
        f"% positive = {(sa > 0).mean():.1%}"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / "signed_area_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# 3. Cross-correlation function
# ---------------------------------------------------------------------------

def plot_cross_correlation(
    corr_series: pd.Series,
    event_type: str = "All Events",
    save: bool = True,
) -> Any:
    """Plot the cross-correlation function with optimal lag highlighted.

    Parameters
    ----------
    corr_series : Series indexed by lag (minutes), values = correlation.
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(corr_series.index, corr_series.values, width=0.8,
           color=["#d62728" if v < 0 else "#1f77b4" for v in corr_series.values],
           alpha=0.7)

    # Highlight optimal lag
    best_lag = int(corr_series.abs().idxmax())
    best_val = corr_series.loc[best_lag]
    ax.axvline(best_lag, color="red", linestyle="--", alpha=0.8,
               label=f"Optimal lag = {best_lag} min (r = {best_val:.3f})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag (minutes, positive = pred mkt leads)")
    ax.set_ylabel("Cross-Correlation")
    ax.set_title(f"Cross-Correlation: Pred Market Returns vs Equity Returns -- {event_type}")
    ax.legend()

    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / f"cross_correlation_{event_type.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# 4. Lead-lag summary table by event type
# ---------------------------------------------------------------------------

def build_lead_lag_table(results: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of lead-lag statistics by event type.

    Parameters
    ----------
    results : DataFrame with columns:
        event_type, signed_area, xcorr_optimal_lag, xcorr_max_corr,
        granger_best_lag, granger_best_pval, n_obs

    Returns
    -------
    Summary DataFrame with one row per event type.
    """
    if results.empty:
        return pd.DataFrame()

    summary = results.groupby("event_type").agg(
        n_events=("signed_area", "count"),
        mean_signed_area=("signed_area", "mean"),
        median_signed_area=("signed_area", "median"),
        pct_pm_leads=("signed_area", lambda x: (x > 0).mean()),
        mean_xcorr_lag=("xcorr_optimal_lag", "mean"),
        mean_xcorr_r=("xcorr_max_corr", "mean"),
        mean_granger_lag=("granger_best_lag", "mean"),
        min_granger_pval=("granger_best_pval", "min"),
    ).round(4)

    summary = summary.sort_values("pct_pm_leads", ascending=False)
    return summary


def save_lead_lag_table(
    summary: pd.DataFrame,
    fmt: str = "both",
) -> None:
    """Save the lead-lag summary table as CSV and/or markdown.

    Parameters
    ----------
    fmt : "csv", "markdown", or "both".
    """
    _ensure_dirs()

    if fmt in ("csv", "both"):
        path = TABLE_DIR / "lead_lag_summary.csv"
        summary.to_csv(path)
        logger.info("Saved %s", path)

    if fmt in ("markdown", "both"):
        path = TABLE_DIR / "lead_lag_summary.md"
        with open(path, "w") as f:
            f.write("# Lead-Lag Summary by Event Type\n\n")
            f.write(summary.to_markdown())
            f.write("\n")
        logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# 5. TDA persistence diagram
# ---------------------------------------------------------------------------

def plot_persistence_diagram(
    diagram: np.ndarray,
    event_type: str = "All Events",
    save: bool = True,
) -> Any:
    """Plot a persistence diagram (birth vs death).

    Parameters
    ----------
    diagram : (n, 3) array with columns (birth, death, dimension).
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(7, 7))

    # Separate by dimension
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "H0 (components)", 1: "H1 (loops)"}

    for dim in [0, 1]:
        mask = diagram[:, 2] == dim
        pts = diagram[mask]
        # Filter out infinite death
        finite = pts[np.isfinite(pts[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], c=colors[dim],
                      alpha=0.6, s=30, label=labels[dim])

    # Diagonal
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(f"Persistence Diagram -- {event_type}")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / f"persistence_diagram_{event_type.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# 6. TDA regime timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    regime_df: pd.DataFrame,
    event_type: str = "All Events",
    save: bool = True,
) -> Any:
    """Plot H1 persistence over time with regime change points highlighted.

    Parameters
    ----------
    regime_df : output of tda.detect_regime_changes().
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(regime_df.index, regime_df["h1_value"],
            color="#1f77b4", linewidth=1, alpha=0.8, label="H1 persistence")
    ax.plot(regime_df.index, regime_df["rolling_mean"],
            color="gray", linestyle="--", linewidth=1, label="Rolling mean")

    # Shade regime changes
    changes = regime_df[regime_df["is_regime_change"]]
    if not changes.empty:
        ax.scatter(changes.index, changes["h1_value"],
                  color="red", s=50, zorder=5, label="Regime change")

    ax.set_xlabel("Time")
    ax.set_ylabel("H1 Max Persistence (loop strength)")
    ax.set_title(f"Lead-Lag Regime Detection via TDA -- {event_type}")
    ax.legend()

    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / f"regime_timeline_{event_type.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# 7. Signed area by event type (box plot)
# ---------------------------------------------------------------------------

def plot_signed_area_by_event(
    results: pd.DataFrame,
    save: bool = True,
) -> Any:
    """Box plot of signed area grouped by event type.

    Makes it immediately visible which event types show the strongest
    lead-lag and in which direction.
    """
    plt = _get_plt()

    if "event_type" not in results.columns:
        return None

    grouped = results.groupby("event_type")["signed_area"]
    types_ordered = grouped.median().sort_values(ascending=False).index

    data_to_plot = [results[results["event_type"] == t]["signed_area"].dropna() for t in types_ordered]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_to_plot, labels=types_ordered, vert=True, patch_artist=True)

    # Color boxes
    for i, patch in enumerate(bp["boxes"]):
        median_val = data_to_plot[i].median()
        patch.set_facecolor("#2ca02c" if median_val > 0 else "#d62728")
        patch.set_alpha(0.5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Signed Area")
    ax.set_xlabel("Event Type")
    ax.set_title("Lead-Lag Strength by Macro Event Type (positive = pred mkt leads)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        _ensure_dirs()
        path = FIG_DIR / "signed_area_by_event_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    return fig
