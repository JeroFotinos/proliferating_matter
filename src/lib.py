from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import (
    Lattice,
    SquareLattice,
    TriangularLattice,
    HoneycombLattice,
    SimulationResult,
)


# ============================================================
# Result conversion and aggregation
# ============================================================

def result_to_frame(
    result: SimulationResult,
    run_id: Optional[int] = None,
    extra_columns: Optional[dict] = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": result.times,
            "population": result.population,
            "density": result.density,
            "mean_radius": result.mean_radius,
            "equiv_radius": result.equiv_radius,
            "roughness": result.roughness,
            "n_active": result.n_active,
            "center_x": result.center_x,
            "center_y": result.center_y,
            "front_velocity": result.front_velocity,
        }
    )

    if run_id is not None:
        df["run_id"] = run_id

    if extra_columns is not None:
        for key, value in extra_columns.items():
            df[key] = value

    return df


def aggregate_results(
    results: list[SimulationResult],
    times: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Aggregate an ensemble of runs observed on the same time grid.

    Assumes all runs were recorded at the same observation times.
    """
    if len(results) == 0:
        raise ValueError("results must be non-empty.")

    reference_times = results[0].times
    for result in results[1:]:
        if not np.allclose(result.times, reference_times):
            raise ValueError("All results must share the same observation times.")

    pop = np.vstack([r.population for r in results])
    dens = np.vstack([r.density for r in results])
    rad = np.vstack([r.mean_radius for r in results])
    req = np.vstack([r.equiv_radius for r in results])
    rough = np.vstack([r.roughness for r in results])
    vel = np.vstack([r.front_velocity for r in results])

    survival = (pop > 0).astype(float)

    df = pd.DataFrame(
        {
            "time": reference_times,
            "mean_population": pop.mean(axis=0),
            "std_population": pop.std(axis=0, ddof=0),
            "survival_probability": survival.mean(axis=0),
            "mean_density": dens.mean(axis=0),
            "std_density": dens.std(axis=0, ddof=0),
            "mean_radius": np.nanmean(rad, axis=0),
            "std_radius": np.nanstd(rad, axis=0, ddof=0),
            "mean_equiv_radius": np.nanmean(req, axis=0),
            "mean_roughness": np.nanmean(rough, axis=0),
            "std_roughness": np.nanstd(rough, axis=0, ddof=0),
            "mean_front_velocity": np.nanmean(vel, axis=0),
            "std_front_velocity": np.nanstd(vel, axis=0, ddof=0),
        }
    )
    return df


# ============================================================
# Power-law fitting helpers
# ============================================================

def fit_power_law_exponent(
    times: np.ndarray,
    values: np.ndarray,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> dict:
    """
    Fit values ~ C * t^alpha by linear regression in log-log scale.

    Only positive finite time/value pairs are used.

    Parameters
    ----------
    times : array-like
        Time points corresponding to the observed values.
    values : array-like
        Observed values to fit.
    t_min : float, optional
        Minimum time to include in the fit. If None, no minimum is applied.
    t_max : float, optional
        Maximum time to include in the fit. If None, no maximum is applied.

    Returns
    -------
    dict
        A dictionary containing the fitted exponent 'alpha', the log of the
        prefactor 'log_C', the prefactor 'C', the R^2 of the log-log fit
        'r2_loglog', and the number of points used in the fit 'n_points'.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(times) & np.isfinite(values) & (times > 0.0) & (values > 0.0)

    if t_min is not None:
        mask &= times >= t_min
    if t_max is not None:
        mask &= times <= t_max

    t_fit = times[mask]
    y_fit = values[mask]

    if t_fit.size < 2:
        raise ValueError("Not enough valid points for a power-law fit.")

    log_t = np.log(t_fit)
    log_y = np.log(y_fit)

    slope, intercept = np.polyfit(log_t, log_y, deg=1)

    y_pred = slope * log_t + intercept
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": float(slope),
        "log_C": float(intercept),
        "C": float(np.exp(intercept)),
        "r2_loglog": float(r2),
        "n_points": int(t_fit.size),
    }


# ============================================================
# Plotting
# ============================================================

def plot_observables(
    result: SimulationResult,
    which: tuple[str, ...] = ("population", "mean_radius", "roughness", "front_velocity"),
    title_prefix: str = "",
) -> None:
    n_panels = len(which)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), squeeze=False)

    for ax, key in zip(axes[:, 0], which):
        values = getattr(result, key)
        ax.plot(result.times, values)
        ax.set_xlabel("Time")
        ax.set_ylabel(key)
        ax.set_title(f"{title_prefix}{key}")

    plt.tight_layout()
    plt.show()


def plot_ensemble_curve(
    agg_df: pd.DataFrame,
    y: str,
    y_std: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(agg_df["time"], agg_df[y], label=y)

    if y_std is not None:
        lower = agg_df[y] - agg_df[y_std]
        upper = agg_df[y] + agg_df[y_std]
        plt.fill_between(agg_df["time"], lower, upper, alpha=0.25)

    plt.xlabel("Time")
    plt.ylabel(y)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_snapshots(
    lattice: Lattice,
    result: SimulationResult,
    times_to_show: list[float],
    title_prefix: str = "",
    cmap: str = "binary",
) -> None:
    """
    Plot occupancy snapshots.

    Convention with cmap='binary':
    - 0 = empty -> white
    - 1 = occupied -> black
    """
    if result.snapshots is None:
        raise ValueError("This result does not contain snapshots. Run with store_snapshots=True.")

    n = len(times_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, ts in zip(axes[0], times_to_show):
        arr = lattice.reshape_state(result.snapshots[float(ts)]).astype(int)
        ax.imshow(
            arr,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{title_prefix}t = {ts:g}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.tight_layout()
    plt.show()