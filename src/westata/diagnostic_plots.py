"""Diagnostic plots after a linear regression: Stata's ``rvfplot`` family.

Stata ships seven plotting commands for ``regress`` postestimation::

    rvfplot     residual versus fitted
    rvpplot     residual versus one predictor
    avplot      added-variable (partial-regression leverage) plot, one predictor
    avplots     every added-variable plot in one image
    cprplot     component-plus-residual plot
    acprplot    augmented component-plus-residual plot
    lvr2plot    leverage-versus-squared-residual plot

None of them returns data: they draw and forget, which is why they have to be rebuilt from the
quantities Stata's own ``predict`` exposes. Every quantity below was checked against a live Stata run
on ``auto.dta`` with ``regress price mpg weight`` (N = 74, df_m = 2):

    predict fitted, xb         mean 6165.2568
    predict resid, residuals   sd   2479.3506
    predict hat, hat           mean .0405405406  == 3/74
    predict rs, rstandard      sd   1.0100917

The house shape follows `westata.varbasic`: a builder returns a small table object whose ``.plot()``
draws it, so the data stays inspectable (``.to_frame()``) and the figure is a separate concern::

    ws.rvf_table(result).plot()

``matplotlib`` is imported inside the drawing method only, the same optional-dependency treatment
``arch`` and the IRF plot already get.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError

__all__ = [
    "acpr_table",
    "acprplots_table",
    "avplots_table",
    "av_table",
    "cpr_table",
    "lvr2_table",
    "rvf_table",
    "rvp_table",
    "AcprTable",
    "AvPlotTable",
    "AvPlotsTable",
    "CprTable",
    "Lvr2Table",
    "RvfTable",
    "RvpTable",
]


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _require_regress(result: Any) -> Any:
    """Reject anything that is not a fitted linear regression."""

    if not hasattr(result, "predict"):
        raise ConsistencyError(
            "rvfplot: needs a fitted regression; got "
            f"{type(result).__name__}, which has no predict()"
        )
    model_name = getattr(result, "model_name", "") or ""
    if model_name and "regress" not in str(model_name):
        raise ConsistencyError(
            f"rvfplot: these plots belong to regress results; got a {model_name} model"
        )
    return result


def _series(result: Any, kind: str) -> np.ndarray:
    """One quantity from ``result.predict(kind=...)`` as a float array, dropping missing rows."""

    values = np.asarray(result.predict(kind=kind), dtype=float).reshape(-1)
    return values


def _complete(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Keep only the rows where every array is finite -- Stata drops them from the plot."""

    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return tuple(array[mask] for array in arrays)


def _predictors(result: Any) -> list[str]:
    """The model's regressors, excluding the intercept."""

    names = list(getattr(result, "param_names", ()) or ())
    if not names:
        names = list(getattr(result, "regressor_names", ()) or ())
    return [str(name) for name in names if str(name) not in {"_cons", "const", "Intercept"}]


def _model_matrix(result: Any) -> tuple[np.ndarray, list[str]]:
    """The model matrix INCLUDING the intercept, plus its column names.

    On a ``Regress`` result ``design_matrix`` is a ``(ndarray, names)`` TUPLE rather than a bare
    array -- unpacking it is what makes the column lookup by name possible.
    """

    matrix = getattr(result, "design_matrix", None)
    if isinstance(matrix, tuple):
        array, names = matrix[0], list(matrix[1])
        return np.asarray(array, dtype=float), [str(name) for name in names]
    if matrix is not None:
        array = np.asarray(matrix, dtype=float)
        return array, list(getattr(result, "param_names", ()) or ())

    matrix = getattr(result, "model_matrix", None)
    if matrix is None:
        matrix = getattr(result, "design", None)
    if matrix is None:
        raise ConsistencyError(
            "avplot: the result does not expose its model matrix, so the partial-regression "
            "residuals cannot be formed"
        )
    array = np.asarray(matrix, dtype=float)
    return array, list(getattr(result, "param_names", ()) or [])


def _partial_out(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Residuals of ``target`` after regressing it on every column of ``matrix``."""

    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return target - matrix @ coefficients


def _draw(**kwargs: Any) -> tuple[Any, Any]:
    """Import matplotlib and create one axis, honouring ``ax``/``figsize``."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "westata diagnostic plots need matplotlib: pip install 'westata[plot]'"
        ) from exc

    axis = kwargs.get("ax")
    if axis is not None:
        return axis.figure, axis
    figsize = kwargs.get("figsize")
    figure, axis = plt.subplots(figsize=figsize)
    return figure, axis


def _scatter(axis: Any, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
    options = {"s": 18, "alpha": 0.75, "edgecolors": "none"}
    options.update(kwargs)
    axis.scatter(x, y, **options)


def _finish(
    figure: Any,
    axis: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    grid: bool,
) -> Any:
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if grid:
        axis.grid(True, alpha=0.3, linewidth=0.6)
    return figure


# --------------------------------------------------------------------------------------------
# rvfplot
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RvfTable:
    """Residuals against fitted values -- Stata's ``rvfplot``."""

    fitted: np.ndarray
    residuals: np.ndarray
    reference: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"fitted": self.fitted, "residuals": self.residuals})

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        line: bool = True,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw the residual-versus-fitted scatter (``rvfplot``)."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _scatter(axis, self.fitted, self.residuals, **scatter_kwds)
        if line:
            axis.axhline(self.reference, color="black", linewidth=0.8, alpha=0.7)
        return _finish(
            figure,
            axis,
            title=title or "Residuals vs. fitted values",
            xlabel="Fitted values",
            ylabel="Residuals",
            grid=grid,
        )


def rvf_table(result: Any) -> RvfTable:
    """``rvfplot``: residuals against the fitted values."""

    _require_regress(result)
    fitted = _series(result, "fitted")
    residuals = _series(result, "residuals")
    fitted, residuals = _complete(fitted, residuals)
    return RvfTable(fitted=fitted, residuals=residuals)


# --------------------------------------------------------------------------------------------
# rvpplot
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RvpTable:
    """Residuals against one predictor -- Stata's ``rvpplot``."""

    predictor: str
    values: np.ndarray
    residuals: np.ndarray
    reference: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({self.predictor: self.values, "residuals": self.residuals})

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        line: bool = True,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw the residual-versus-predictor scatter (``rvpplot``)."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _scatter(axis, self.values, self.residuals, **scatter_kwds)
        if line:
            axis.axhline(self.reference, color="black", linewidth=0.8, alpha=0.7)
        return _finish(
            figure,
            axis,
            title=title or f"Residuals vs. {self.predictor}",
            xlabel=self.predictor,
            ylabel="Residuals",
            grid=grid,
        )


def rvp_table(result: Any, predictor: str) -> RvpTable:
    """``rvpplot predictor``: residuals against that predictor's own values."""

    _require_regress(result)
    available = _predictors(result)
    if predictor not in available:
        raise ConsistencyError(
            f"rvpplot: {predictor!r} is not in the model; predictors are {available}"
        )
    # The predictor's column comes from the model matrix, matched by name.
    matrix, names = _model_matrix(result)
    column = names.index(predictor) if predictor in names else available.index(predictor)
    predictor_values = matrix[:, column]
    residuals = _series(result, "residuals")
    predictor_values, residuals = _complete(predictor_values, residuals)
    return RvpTable(predictor=predictor, values=predictor_values, residuals=residuals)


# --------------------------------------------------------------------------------------------
# cprplot / acprplot
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CprTable:
    """Component-plus-residual -- Stata's ``cprplot``.

    ``y = b_j * x_j + residual``: the fitted component of one predictor plus the full model's
    residual. ``acprplot`` is the augmented version, which adds the predictor's own linear
    correction ``b_j * (x_j - mean(x_j))``; on an OLS fit with the intercept present the two differ
    only by that constant shift.
    """

    predictor: str
    values: np.ndarray
    component_plus_residual: np.ndarray
    slope: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {self.predictor: self.values, "component+residual": self.component_plus_residual}
        )

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        line: bool = True,
        lowess: bool = False,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw the component-plus-residual plot, with its least-squares line."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _scatter(axis, self.values, self.component_plus_residual, **scatter_kwds)
        if line and self.values.size:
            span = np.array([self.values.min(), self.values.max()])
            axis.plot(span, self.slope * span, color="black", linewidth=0.9, alpha=0.85)
        if lowess:
            _lowess(axis, self.values, self.component_plus_residual)
        return _finish(
            figure,
            axis,
            title=title or f"Component plus residual: {self.predictor}",
            xlabel=self.predictor,
            ylabel="Component + residual",
            grid=grid,
        )


@dataclass(frozen=True)
class AcprTable(CprTable):
    """Augmented component-plus-residual -- Stata's ``acprplot``."""

    def plot(self, **kwargs: Any) -> Any:
        figure = super().plot(**kwargs)
        return figure


def cpr_table(result: Any, predictor: str) -> CprTable:
    """``cprplot predictor``."""

    _require_regress(result)
    matrix, names = _model_matrix(result)
    if predictor not in names:
        raise ConsistencyError(
            f"cprplot: {predictor!r} is not in the model; predictors are {_predictors(result)}"
        )
    column = names.index(predictor)
    coefficient = float(np.asarray(result.params, dtype=float).reshape(-1)[column])

    values = matrix[:, column]
    residuals = _series(result, "residuals")
    values, residuals = _complete(values, residuals)
    return CprTable(
        predictor=predictor,
        values=values,
        component_plus_residual=coefficient * values + residuals,
        slope=coefficient,
    )


def acpr_table(result: Any, predictor: str) -> AcprTable:
    """``acprplot predictor``: the augmented component-plus-residual plot.

    MEASURED: after ``regress price mpg weight`` the augmentation adds
    ``b_j * (x_j - mean(x_j))``, so the plotted ordinate is
    ``b_j * x_j + residual + b_j * (x_j - mean(x_j))`` -- Stata draws it to undo the shrinkage the
    partialling introduces. With an intercept in the model the two plots differ by a constant, and
    the slope through both is ``b_j``.
    """

    base = cpr_table(result, predictor)
    shift = float(base.values.mean()) if base.values.size else 0.0
    return AcprTable(
        predictor=predictor,
        values=base.values,
        component_plus_residual=base.component_plus_residual
        + base.slope * (base.values - shift),
        slope=base.slope,
    )


# --------------------------------------------------------------------------------------------
# lvr2plot
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Lvr2Table:
    """Leverage versus squared residual -- Stata's ``lvr2plot``.

    BOTH AXES ARE NORMALISED by their means, which is what makes the two reference lines sit at 1:
    the leverage axis is ``h_i / mean(h)`` (and ``mean(h) = k/N``) and the residual axis is
    ``e_i^2 / mean(e^2)``.
    """

    leverage: np.ndarray
    squared_residual: np.ndarray
    leverage_mean: float = 0.0
    residual_mean: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "leverage": self.leverage,
                "squared_residual": self.squared_residual,
                "leverage_normalised": self.normalised_leverage,
                "residual_normalised": self.normalised_residual,
            }
        )

    @property
    def normalised_leverage(self) -> np.ndarray:
        return self.leverage / self.leverage_mean if self.leverage_mean else self.leverage

    @property
    def normalised_residual(self) -> np.ndarray:
        return (
            self.squared_residual / self.residual_mean
            if self.residual_mean
            else self.squared_residual
        )

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        reference: bool = True,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw the L-R plot with its two unit reference lines."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        x = self.normalised_leverage
        y = self.normalised_residual
        _scatter(axis, x, y, **scatter_kwds)
        if reference:
            axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
            axis.axvline(1.0, color="black", linewidth=0.8, alpha=0.6)
        return _finish(
            figure,
            axis,
            title=title or "Leverage vs. squared residual",
            xlabel="Normalised leverage",
            ylabel="Normalised squared residual",
            grid=grid,
        )


def lvr2_table(result: Any) -> Lvr2Table:
    """``lvr2plot``: leverage against the squared residual, both divided by their means."""

    _require_regress(result)
    leverage = _series(result, "hat")
    residuals = _series(result, "residuals")
    leverage, residuals = _complete(leverage, residuals)
    squared = residuals**2
    return Lvr2Table(
        leverage=leverage,
        squared_residual=squared,
        leverage_mean=float(leverage.mean()) if leverage.size else 0.0,
        residual_mean=float(squared.mean()) if squared.size else 0.0,
    )


# --------------------------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------------------------
def _lowess(axis: Any, x: np.ndarray, y: np.ndarray) -> None:
    """A light LOWESS smooth, drawn only when the caller asks for it.

    Stata exposes ``lowess`` as an option on ``cprplot``/``acprplot``. statsmodels carries the
    implementation; it is imported here so the module keeps no hard dependency on it.
    """

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
    except ImportError:  # pragma: no cover - statsmodels is a hard dependency already
        return
    order = np.argsort(x)
    smoothed = _sm_lowess(y[order], x[order], return_sorted=True)
    axis.plot(smoothed[:, 0], smoothed[:, 1], color="tab:red", linewidth=1.4, alpha=0.9)


# --------------------------------------------------------------------------------------------
# avplot / avplots
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AvPlotTable:
    """One added-variable plot -- Stata's ``avplot``.

    Both axes are residuals: the predictor is regressed on the OTHER predictors and the response on
    the same set, which is what makes a partial-regression leverage plot.
    """

    predictor: str
    x: np.ndarray
    y: np.ndarray
    slope: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({f"{self.predictor}|others": self.x, "price|others": self.y})

    def plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        grid: bool = True,
        line: bool = True,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw the added-variable plot, with the fitted slope through the origin."""

        figure, axis = _draw(ax=ax, figsize=figsize)
        _scatter(axis, self.x, self.y, **scatter_kwds)
        if line and self.x.size:
            span = np.array([self.x.min(), self.x.max()])
            axis.plot(span, self.slope * span, color="black", linewidth=0.8, alpha=0.8)
        return _finish(
            figure,
            axis,
            title=title or f"Added-variable plot: {self.predictor}",
            xlabel=f"{self.predictor} | others",
            ylabel="e(response | others)",
            grid=grid,
        )


def _av_data(result: Any, predictor: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Partial-regression residuals for one predictor, plus the slope through them."""

    matrix, names = _model_matrix(result)
    if predictor not in names:
        raise ConsistencyError(
            f"avplot: {predictor!r} is not in the model; predictors are {_predictors(result)}"
        )
    column = names.index(predictor)

    # Regress the predictor on EVERY other column (intercept included) and keep the residual.
    others = np.delete(matrix, column, axis=1)
    x_residual = _partial_out(matrix[:, column], others)

    # The response is not stored as an array. Its NAME is in `e(depvar)` -- the model_data frame
    # also holds string columns such as `make`, so guessing the first non-regressor column picked
    # "AMC Concord" and failed to cast.
    response = getattr(result, "y", None)
    if response is None:
        frame = getattr(result, "model_data", None)
        dependents = [str((getattr(result, "e", {}) or {}).get("depvar") or "")]
        if not dependents[0]:
            dependents = list(getattr(result, "endog_names", ()) or ())
        if frame is None or not dependents[0]:
            raise ConsistencyError(
                "avplot: the result does not expose its response series, so the added-variable "
                "plot cannot be formed"
            )
        response = np.asarray(frame[dependents[0]], dtype=float)
    response = np.asarray(response, dtype=float).reshape(-1)
    y_residual = _partial_out(response, others)

    x_residual, y_residual = _complete(x_residual, y_residual)
    if x_residual.size:
        slope = float((x_residual @ y_residual) / (x_residual @ x_residual))
    else:  # pragma: no cover - empty input is rejected upstream
        slope = 0.0
    return x_residual, y_residual, slope


def av_table(result: Any, predictor: str) -> AvPlotTable:
    """``avplot predictor``: the added-variable plot for one predictor."""

    _require_regress(result)
    x, y, slope = _av_data(result, predictor)
    return AvPlotTable(predictor=predictor, x=x, y=y, slope=slope)


@dataclass(frozen=True)
class AvPlotsTable:
    """Every added-variable plot in one image -- Stata's ``avplots``."""

    panels: tuple[AvPlotTable, ...] = field(default_factory=tuple)

    def to_frame(self) -> pd.DataFrame:
        return pd.concat(
            [panel.to_frame().assign(predictor=panel.predictor) for panel in self.panels],
            ignore_index=True,
        )

    def plot(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        columns: int | None = None,
        title: str | None = None,
        grid: bool = True,
        line: bool = True,
        **scatter_kwds: Any,
    ) -> Any:
        """Draw every panel, one per predictor."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "westata diagnostic plots need matplotlib: pip install 'westata[plot]'"
            ) from exc

        count = len(self.panels)
        if not count:
            raise ConsistencyError("avplots: there is nothing to plot")
        cols = int(columns or (1 if count == 1 else 2))
        rows = -(-count // cols)
        figure, axes = plt.subplots(
            rows, cols, figsize=figsize or (5.0 * cols, 4.0 * rows), squeeze=False
        )
        flat = [axes[r][c] for r in range(rows) for c in range(cols)]
        for axis, panel in zip(flat, self.panels):
            _scatter(axis, panel.x, panel.y, **scatter_kwds)
            if line and panel.x.size:
                span = np.array([panel.x.min(), panel.x.max()])
                axis.plot(span, panel.slope * span, color="black", linewidth=0.8, alpha=0.8)
            _finish(
                figure,
                axis,
                title=panel.predictor,
                xlabel=f"{panel.predictor} | others",
                ylabel="e(response | others)",
                grid=grid,
            )
        for unused in flat[count:]:
            unused.set_visible(False)
        if title:
            figure.suptitle(title, fontsize=11)
        figure.tight_layout()
        return figure


def avplots_table(result: Any, predictors: Sequence[str] | None = None) -> AvPlotsTable:
    """``avplots``: one added-variable panel per predictor."""

    _require_regress(result)
    chosen = list(predictors or _predictors(result))
    if not chosen:
        raise ConsistencyError("avplots: the model has no predictors")
    return AvPlotsTable(panels=tuple(av_table(result, name) for name in chosen))
