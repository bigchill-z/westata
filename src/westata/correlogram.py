"""Autocorrelation and partial-autocorrelation plots: Stata's ``ac`` and ``pac``.

    ac varname [, lags(#) generate(newvar) level(#) fft]
    pac varname [, lags(#) generate(newvar) level(#)]

Both DRAW a correlogram with a confidence band; neither returns data, so the numbers are recovered
from Stata's own ``generate(newvar)`` and from the ``corrgram`` table, which reports the same two
columns.

DEFINITIONS, each checked against a live Stata run on 200 observations of an AR(1) with phi = 0.6:

* ``ac`` is the ordinary autocorrelation, ``sum_{t>k}(x_t - xbar)(x_{t-k} - xbar) / sum (x - xbar)^2``
  -- all twelve lags matched to eight decimals.
* ``pac`` is NOT Yule-Walker. It is the LAST coefficient of an OLS regression of ``x_t`` on a
  CONSTANT plus the first ``k`` lags. Yule-Walker gave .42822423 at lag 1 where Stata prints
  .42937017; including the constant reproduces .42937017 exactly, and all twelve lags then match.
  This is the one detail that a plausible implementation gets wrong.
* the ``ac`` band is Bartlett's formula for MA(q)::

      se(k) = sqrt( (1 + 2 * sum_{j<k} rho_j^2) / n )

* the ``pac`` band is the constant ``1/sqrt(n)`` -- Stata's help says so outright.

The house shape applies again: ``ws.ac_table(series).plot()`` and ``ws.pac_table(series).plot()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError
from westata.plotting import make_figure

__all__ = [
    "ac_table",
    "pac_table",
    "AcTable",
    "PacTable",
]


def _as_series(data: Any, column: str | None = None) -> np.ndarray:
    """Accept a Series, a 1-D array, or a DataFrame plus a column name."""

    if isinstance(data, pd.DataFrame):
        if column is not None:
            if column not in data.columns:
                raise ConsistencyError(
                    f"ac: column {column!r} is not in the frame; available: {list(data.columns)}"
                )
            values = data[column].to_numpy()
        elif data.shape[1] == 1:
            values = data.iloc[:, 0].to_numpy()
        else:
            raise ConsistencyError(
                "ac: the frame has several columns; name one with column= "
                f"(available: {list(data.columns)})"
            )
    elif isinstance(data, pd.Series):
        values = data.to_numpy()
    else:
        values = np.asarray(data)

    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _autocorrelations(values: np.ndarray, nlags: int) -> np.ndarray:
    """``rho_k`` for k = 1..nlags, with the mean removed and the zero-lag sum in the denominator."""

    deviations = values - values.mean()
    denominator = float(deviations @ deviations)
    if denominator <= 0.0:
        raise ConsistencyError("ac: the series is constant, so it has no autocorrelation")
    total = deviations.size
    return np.array(
        [float(deviations[: total - k] @ deviations[k:]) / denominator for k in range(1, nlags + 1)]
    )


def _partial_autocorrelations(values: np.ndarray, nlags: int) -> np.ndarray:
    """The LAST OLS coefficient of ``x_t`` on a constant and the first ``k`` lags, for each k.

    NOT the Yule-Walker recursion: that gives .42822423 at lag 1 where Stata prints .42937017,
    because the regression carries an intercept and therefore a different denominator.
    """

    total = values.size
    if nlags >= total:
        raise ConsistencyError(
            f"pac: lags={nlags} leaves no observations on {total} values"
        )
    result = []
    for k in range(1, nlags + 1):
        rows = np.column_stack(
            [np.ones(total - k)] + [values[k - j : total - j] for j in range(1, k + 1)]
        )
        target = values[k:]
        coefficients, *_ = np.linalg.lstsq(rows, target, rcond=None)
        result.append(float(coefficients[-1]))
    return np.array(result)


def _draw(**kwargs: Any) -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "westata ac/pac plots need matplotlib: pip install 'westata[plot]'"
        ) from exc
    axis = kwargs.get("ax")
    if axis is not None:
        return axis.figure, axis
    figure, axis = make_figure(figsize=kwargs.get("figsize"))
    return figure, axis


def _stem_plot(
    axis: Any,
    lags: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    band: bool,
) -> None:
    """Stata's correlogram is a stem plot inside a shaded band.

    THE X AXIS IS AN INTEGER LAG INDEX, so matplotlib's automatic locator is wrong for it: with 20
    lags it produced `0.0 2.5 5.0 7.5 ... 22.5` -- half-steps that no lag occupies -- and it ran past
    the last lag. Stata labels whole lags only, so the ticks are forced to integers and the limits
    held to the data, with a small margin so the end stems are not clipped.
    """

    # Imported here: matplotlib stays an optional dependency and this module must import without it.
    from matplotlib.ticker import MaxNLocator

    axis.axhline(0.0, color="black", linewidth=0.8)
    if band:
        axis.fill_between(lags, lower, upper, color="tab:blue", alpha=0.18, linewidth=0)
        axis.plot(lags, upper, color="tab:blue", linewidth=0.8, alpha=0.7)
        axis.plot(lags, lower, color="tab:blue", linewidth=0.8, alpha=0.7)
    axis.vlines(lags, 0.0, values, color="tab:blue", linewidth=1.4)
    axis.plot(lags, values, linestyle="none", marker="o", markersize=4.5, color="tab:blue")

    # `integer=True` both drops the half-steps and keeps the labels whole; without it a 20-lag plot
    # labels 0.0, 2.5, 5.0 ... and the axis runs on to 22.5.
    first, last = int(lags.min()), int(lags.max())
    axis.set_xlim(first - 0.6, last + 0.6)
    # Set the LIMITS BEFORE the locator and let `prune="both"` drop anything outside the view --
    # without it a 0.4..20.6 window still emitted a 21 tick, past the last lag.
    axis.xaxis.set_major_locator(
        MaxNLocator(nbins="auto", integer=True, prune="both", min_n_ticks=3)
    )


@dataclass(frozen=True)
class AcTable:
    """Autocorrelations with Bartlett bands -- Stata's ``ac``."""

    lags: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float = 0.95

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lag": self.lags, "ac": self.values, "lower": self.lower, "upper": self.upper}
        )

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        band: bool = True,
    ) -> Any:
        """Draw the autocorrelation correlogram."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _stem_plot(axis, self.lags, self.values, self.lower, self.upper, band=band)
        axis.set_title(title or "Autocorrelations of the series", fontsize=10)
        axis.set_xlabel("Lag")
        axis.set_ylabel("Autocorrelation")
        if grid:
            axis.grid(True, alpha=0.3, linewidth=0.6)
        return figure


@dataclass(frozen=True)
class PacTable(AcTable):
    """Partial autocorrelations with ``1/sqrt(n)`` bands -- Stata's ``pac``."""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lag": self.lags, "pac": self.values, "lower": self.lower, "upper": self.upper}
        )

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        band: bool = True,
    ) -> Any:
        """Draw the partial-autocorrelation correlogram."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _stem_plot(axis, self.lags, self.values, self.lower, self.upper, band=band)
        axis.set_title(title or "Partial autocorrelations of the series", fontsize=10)
        axis.set_xlabel("Lag")
        axis.set_ylabel("Partial autocorrelation")
        if grid:
            axis.grid(True, alpha=0.3, linewidth=0.6)
        return figure


def _default_lags(values: np.ndarray, nlags: int | None) -> int:
    """Stata's default is ``min(floor(n/2) - 2, 40)``."""

    if nlags is not None:
        return int(nlags)
    return int(min(values.size // 2 - 2, 40))


def ac_table(data: Any, *, column: str | None = None, lags: int | None = None,
             level: float = 0.95) -> AcTable:
    """``ac``: autocorrelations with Bartlett confidence bands.

    ``level`` is a FRACTION (0.95), per the package convention; Stata spells it ``level(95)``.
    """

    values = _as_series(data, column)
    nlags = _default_lags(values, lags)
    if nlags < 1:
        raise ConsistencyError("ac: the series is too short for a correlogram")
    rho = _autocorrelations(values, nlags)

    from scipy import stats as _stats

    critical = float(_stats.norm.ppf(0.5 + level / 2.0))
    total = values.size
    standard_errors = []
    running = 0.0
    for index in range(nlags):
        # Bartlett: se(k)^2 = (1 + 2 * sum_{j<k} rho_j^2) / n
        standard_errors.append(np.sqrt((1.0 + 2.0 * running) / total))
        running += float(rho[index]) ** 2
    half = critical * np.asarray(standard_errors)

    return AcTable(
        lags=np.arange(1, nlags + 1),
        values=rho,
        lower=-half,
        upper=half,
        level=level,
    )


def pac_table(data: Any, *, column: str | None = None, lags: int | None = None,
              level: float = 0.95) -> PacTable:
    """``pac``: partial autocorrelations with a constant ``1/sqrt(n)`` band."""

    values = _as_series(data, column)
    nlags = _default_lags(values, lags)
    if nlags < 1:
        raise ConsistencyError("pac: the series is too short for a correlogram")
    prho = _partial_autocorrelations(values, nlags)

    from scipy import stats as _stats

    critical = float(_stats.norm.ppf(0.5 + level / 2.0))
    half = critical / np.sqrt(values.size)

    return PacTable(
        lags=np.arange(1, nlags + 1),
        values=prho,
        lower=np.full(nlags, -half),
        upper=np.full(nlags, half),
        level=level,
    )
