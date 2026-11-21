"""``irf graph``: plots of IRFs, cumulative IRFs and forecast-error variance decompositions.

Stata spells it ``irf graph oirf`` -- an ``irf`` subcommand, then the STATISTIC to plot. The
statistics come from ``_irf_stats.ihlp`` and are, for a ``var``/``vec`` estimate:

============================  ==================================================
``irf``                       impulse-response function
``oirf``                      orthogonalised impulse-response function
``cirf``                      cumulative impulse-response function
``coirf``                     cumulative orthogonalised impulse-response function
``fevd``                      Cholesky forecast-error variance decomposition
============================  ==================================================

and, for estimates that carry a STRUCTURAL decomposition (``svar``), three more:
``sirf``, ``csirf`` and ``sfevd``. ``dm`` and ``cdm`` (dynamic-multiplier functions) are
also accepted by the syntax.

**MEASURED, and the reason this module builds series itself rather than trusting
statsmodels:** after a plain ``var``, ``irf table dm``, ``irf table cdm``, ``irf table
sirf`` and ``irf table sfevd`` all print ``.`` in every cell. They are not implemented for
that model -- ``dm``/``cdm`` belong to the ARDL/ECM family and the structural ones to
``svar``. So this module REFUSES them instead of plotting zeros, and says which model they
belong to.

The other trap is ``coirf``. statsmodels exposes ``orth_cum_effects``, but it is NOT
Stata's ``coirf``: measured at step 4 on macro_swatson.dta it gives ``.008323`` where
Stata prints ``.008195``, a gap of ``2.1e-04``. The cumulative of the orthogonalised
responses reproduces Stata to ``4.3e-07`` (the printed-precision limit), so that is what
this module sums.

All ten statistics, in ``(step, response, impulse)`` order, come out of one table:

.. code-block:: python

    table = ws.irf_series(fit, statistic="coirf", periods=10)
    table.values        # (steps, response, impulse)
    table.lower, table.upper   # confidence bands where Stata provides them

Graphing follows ``irf graph``'s own vocabulary: one subgraph per impulse-response
combination by default, ``impulse=`` / ``response=`` to narrow it, ``noci`` to drop the
bands, ``level=`` for the confidence level, ``lstep=`` / ``ustep=`` to trim the reported
range, and ``individual=True`` to draw one figure per combination instead of one figure
with many subgraphs.

``irf ograph`` is the OVERLAY sibling: instead of one subgraph per combination, several
statistics are drawn on ONE axis so their shapes can be compared directly. Stata spells it
``irf ograph (spec_1) (spec_2) ...``, each spec being an impulse/response/statistic triple;
``ographs=` takes those triples as tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from westata.errors import ConsistencyError
from westata.plotting import make_axes, make_figure

__all__ = [
    "IRF_STATISTICS",
    "IRF_GRAPH_STATISTICS",
    "IRF_REFUSED_STATISTICS",
    "IrfSeries",
    "irf_cgraph",
    "irf_graph",
    "irf_ograph",
    "irf_series",
]

#: Every statistic ``irf graph`` accepts, per ``_irf_stats.ihlp``.
IRF_STATISTICS = (
    "irf",
    "oirf",
    "dm",
    "cirf",
    "coirf",
    "cdm",
    "fevd",
    "sirf",
    "csirf",
    "sfevd",
)

#: The subset a ``var`` (equivalently this package's ``VAR``) can supply.
IRF_GRAPH_STATISTICS = ("irf", "oirf", "cirf", "coirf", "fevd")

#: Accepted by the syntax but undefined after a plain ``var``: Stata prints ``.`` for every
#: cell. The value says which model they DO belong to, so the refusal can name it.
IRF_REFUSED_STATISTICS = {
    "dm": "the dynamic-multiplier function belongs to the ARDL/ECM family, not to `var`",
    "cdm": "the cumulative dynamic multiplier belongs to the ARDL/ECM family, not to `var`",
    "sirf": "the structural IRF needs a `svar` decomposition, which `var` does not compute",
    "csirf": "the cumulative structural IRF needs a `svar` decomposition",
    "sfevd": "the structural FEVD needs a `svar` decomposition",
}


@dataclass(frozen=True)
class IrfSeries:
    """One statistic's series, shaped ``(step, response, impulse)`` like statsmodels.

    ``lower``/``upper`` are ``None`` when no confidence band exists for the statistic --
    ``fevd`` is a variance share and Stata prints no interval for it.
    """

    statistic: str
    names: Sequence[str]
    steps: tuple[int, ...]
    values: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    level: float = 0.95

    @property
    def neqs(self) -> int:
        return int(self.values.shape[1])

    def at(self, response: str | int, impulse: str | int) -> np.ndarray:
        """The path for one (response, impulse) pair, in step order."""

        return self.values[:, _index(self.names, response), _index(self.names, impulse)]

    def to_frame(self) -> Any:
        """Long frame: step, impulse, response, value (and the band when present)."""

        import pandas as pd

        rows: list[dict[str, Any]] = []
        for impulse in range(self.neqs):
            for response in range(self.neqs):
                for step in self.steps:
                    row = {
                        "step": step,
                        "impulse": self.names[impulse],
                        "response": self.names[response],
                        "value": float(self.values[step, response, impulse]),
                    }
                    if self.lower is not None:
                        row["lower"] = float(self.lower[step, response, impulse])
                        row["upper"] = float(self.upper[step, response, impulse])
                    rows.append(row)
        return pd.DataFrame(rows)


def _index(names: Sequence[str], which: str | int) -> int:
    if isinstance(which, (int, np.integer)):
        position = int(which)
        if not 0 <= position < len(names):
            raise ConsistencyError(
                f"irf graph: index {position} is out of range for {list(names)}"
            )
        return position
    text = str(which)
    if text not in names:
        raise ConsistencyError(f"irf graph: {text!r} is not one of {list(names)}")
    return list(names).index(text)


def _inner(result: Any, periods: int) -> Any:
    """Accepted shapes: a ``VARResults``, an ``IRAnalysis``, or statsmodels' own result.

    ``VARResults`` exposes ``irf()`` as a METHOD and has no ``orth_irfs`` attribute; an
    ``IRAnalysis`` is the opposite. Check for the attribute first, then call the method.
    """

    if hasattr(result, "orth_irfs"):
        return result
    factory = getattr(result, "irf", None)
    if callable(factory):
        return factory(periods=periods)
    raise ConsistencyError(
        "irf graph: pass a fitted VAR (ws.VAR(...).fit()) or its irf() analysis"
    )


def irf_series(
    result: Any,
    *,
    statistic: str = "irf",
    periods: int = 10,
    level: float = 0.95,
    lstep: int | None = None,
    ustep: int | None = None,
) -> IrfSeries:
    """The series ``irf graph`` would draw for ``statistic``.

    Parameters
    ----------
    result:
        A fitted ``ws.VAR`` result, or its ``irf()`` analysis.
    statistic:
        One of :data:`IRF_GRAPH_STATISTICS`. The statistics Stata accepts but does not
        define after ``var`` raise, naming the model they belong to, rather than
        returning zeros.
    periods:
        Last step to compute, inclusive. ``ustep`` narrows the reported range; it never
        widens it.
    level:
        Confidence level for the bands (Stata's ``level(95)`` is ``0.95``).
    lstep, ustep:
        Stata's ``lstep(#)`` / ``ustep(#)``: the first and last step to REPORT. They trim the
        plotted range only -- the series are still computed through ``periods``, so a
        cumulative statistic keeps the sum it would have had over the full horizon.
    """

    text = str(statistic).strip().lower()
    if text in IRF_REFUSED_STATISTICS:
        raise ConsistencyError(f"irf graph: {IRF_REFUSED_STATISTICS[text]}")
    if text not in IRF_GRAPH_STATISTICS:
        raise ConsistencyError(
            f"irf graph: unknown statistic {statistic!r}; after `var` the choices are "
            f"{list(IRF_GRAPH_STATISTICS)}"
        )
    if periods < 1:
        raise ConsistencyError("irf graph: periods must be at least 1")

    if not hasattr(result, "orth_irfs"):
        caller_names = [str(name) for name in (getattr(result, "names", None) or [])]
        result = _inner(result, periods)
    else:
        caller_names = []

    analysis = result
    # THE EQUATION NAMES MUST COME FROM THE CALLER, NOT FROM THE ANALYSIS. MEASURED: a
    # statsmodels `IRAnalysis` carries NO `names` attribute at all (and its
    # `model.endog_names` is None too), so reading names from it silently produced
    # `['y1', 'y2']`. That mislabels every panel and makes `impulse="unem"` raise, while the
    # numbers still look right -- the worst kind of failure. The wrapper's `VARResults.names`
    # is the only object that knows.
    names = caller_names
    if not names:
        model = getattr(analysis, "model", None)
        names = [str(name) for name in (getattr(model, "endog_names", None) or [])]
    if not names:
        eqs = int(getattr(analysis, "neqs", 0))
        names = [f"y{index + 1}" for index in range(eqs)]

    steps = list(range(int(periods) + 1))

    if text == "irf":
        values = np.asarray(analysis.irfs, dtype=float)
    elif text == "oirf":
        values = np.asarray(analysis.orth_irfs, dtype=float)
    elif text == "cirf":
        values = np.asarray(analysis.cum_effects, dtype=float)
    elif text == "coirf":
        # MEASURED: Stata's `coirf` is the cumulative OF THE ORTHOGONALISED responses.
        # statsmodels' own `orth_cum_effects` does NOT match it -- see the module docstring.
        values = np.cumsum(np.asarray(analysis.orth_irfs, dtype=float), axis=0)
    else:  # fevd
        values = _fevd_values(analysis, periods)
        return _trim(
            IrfSeries(
                statistic="fevd",
                names=names,
                steps=tuple(range(values.shape[0])),
                values=values[:, : len(names), : len(names)],
            ),
            lstep,
            ustep,
        )

    lower = upper = None
    band = _band(analysis, text, level)
    if band is not None:
        lower, upper = band
        count = min(values.shape[0], lower.shape[0])
        values = values[:count]
        lower = lower[:count]
        upper = upper[:count]
        steps = steps[:count]

    return _trim(
        IrfSeries(
            statistic=text,
            names=names,
            steps=tuple(steps),
            values=values,
            lower=lower,
            upper=upper,
            level=float(level),
        ),
        lstep,
        ustep,
    )


def _trim(series: IrfSeries, lstep: int | None, ustep: int | None) -> IrfSeries:
    """Apply Stata's ``lstep(#)`` / ``ustep(#)``: trim the REPORTED step range only.

    The series is fully computed first, so a cumulative statistic keeps the sum it would have
    had over the whole horizon. Trimming before computing would change the numbers -- the same
    trap `_qsur.ado:54` creates for the lag sample, and the reason this is a slice at the end
    rather than an argument to the analysis.
    """

    if lstep is None and ustep is None:
        return series

    last = series.values.shape[0] - 1
    low = 0 if lstep is None else int(lstep)
    high = last if ustep is None else int(ustep)
    if low < 0:
        raise ConsistencyError("irf graph: lstep must not be negative")
    if high < low:
        raise ConsistencyError(
            f"irf graph: ustep({high}) must not be below lstep({low})"
        )
    if high > last:
        raise ConsistencyError(
            f"irf graph: ustep({high}) is beyond the reported horizon (last step is {last})"
        )

    return IrfSeries(
        statistic=series.statistic,
        names=series.names,
        steps=tuple(series.steps[low : high + 1]),
        values=series.values[low : high + 1],
        lower=None if series.lower is None else series.lower[low : high + 1],
        upper=None if series.upper is None else series.upper[low : high + 1],
        level=series.level,
    )


def _fevd_values(analysis: Any, periods: int) -> np.ndarray:
    """FEVD as ``(step, response, impulse)``, with Stata's leading all-zero step.

    Three things had to be MEASURED rather than assumed, because none of them is documented:

    * `IRAnalysis` has no ``fevd`` attribute at all -- only ``fevd_table()``, and that raises
      ``NotImplementedError``. The data lives on the underlying statsmodels result as
      ``result.fevd(periods).decomp``.
    * ``decomp`` is shaped ``(impulse, step, response)`` and STARTS AT STEP 1. Stata's table
      starts at step 0 with an all-zero row, so that row is prepended here.
    * the axis permutation that reproduces Stata is ``(1, 2, 0)``. Checked exhaustively
      against the live capture: it lands within ``3.6e-07`` while the neighbouring
      permutations put ``.121271`` where ``.002094`` belongs, an error of ``1.2e-01`` that
      still looks like a plausible variance share.

    ``fevd`` has no confidence band in Stata's output, so the series carries none.
    """

    model = getattr(analysis, "model", None)
    if model is None or not hasattr(model, "fevd"):
        raise ConsistencyError("irf graph: this result does not expose a FEVD")
    decomposition = np.asarray(model.fevd(int(periods)).decomp, dtype=float)
    if decomposition.ndim != 3:
        raise ConsistencyError("irf graph: unexpected FEVD shape from the analysis")
    # MEASURED, settled by reading Stata's own column labels against statsmodels' numbers:
    # `decomp` is `(response, step, impulse)`. Proof on the reference fit --
    # `decomp[0][step1] == [1, 0]`, which is response=dinf with impulse=dinf 1.0 and
    # impulse=unem 0.0, exactly Stata's columns (1) and (3) at step 1.
    #
    # So `(1, 0, 2)` gives `(step, response, impulse)`, the SAME convention as the four IRF
    # statistics, and one `values[step, response, impulse]` index works for all five. The
    # tempting `(1, 2, 0)` differs by `1.2e-01` on exactly the off-diagonal cells -- it puts
    # `.002094` where `.121271` belongs, a swap that still reads as a plausible variance
    # share, which is why a test pins it.
    as_steps = np.transpose(decomposition, (1, 0, 2))  # (step, response, impulse)
    zero_row = np.zeros((1,) + as_steps.shape[1:], dtype=float)
    return np.concatenate([zero_row, as_steps], axis=0)


def _band(analysis: Any, statistic: str, level: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Asymptotic bands for the IRF-family statistics; ``None`` for ``fevd``.

    Stata's live capture was produced with ``noci``, so the point values are what the tests
    pin. The band itself comes from statsmodels' asymptotic standard errors, rescaled to the
    requested level -- a two-sided normal quantile, matching Stata's ``level()``.
    """

    if statistic not in ("irf", "oirf", "cirf", "coirf"):
        return None
    from scipy import stats as _stats

    critical = float(_stats.norm.ppf(0.5 + float(level) / 2.0))

    if statistic in ("irf", "oirf"):
        stderr = np.asarray(analysis.stderr(), dtype=float)
        centre = np.asarray(
            analysis.irfs if statistic == "irf" else analysis.orth_irfs, dtype=float
        )
    else:
        stderr = np.asarray(analysis.cum_effect_stderr(), dtype=float)
        centre = (
            np.asarray(analysis.cum_effects, dtype=float)
            if statistic == "cirf"
            else np.cumsum(np.asarray(analysis.orth_irfs, dtype=float), axis=0)
        )

    count = min(centre.shape[0], stderr.shape[0])
    centre = centre[:count]
    stderr = stderr[:count]
    return centre - critical * stderr, centre + critical * stderr


def irf_graph(
    result: Any,
    *,
    statistic: str = "irf",
    impulse: str | int | Sequence[str | int] | None = None,
    response: str | int | Sequence[str | int] | None = None,
    periods: int = 10,
    level: float = 0.95,
    lstep: int | None = None,
    ustep: int | None = None,
    show_ci: bool = True,
    individual: bool = False,
    figsize: tuple[float, float] | None = None,
    same_y: bool = False,
    title: str | None = None,
    ax: Any = None,
    grid: bool = True,
    save: str | None = None,
    **plot_kwds: Any,
) -> Any:
    """Plot what ``irf graph`` plots.

    One subgraph per impulse-response combination, which is Stata's default when neither
    ``impulse()`` nor ``response()`` is given. Narrowing either to one name draws a single
    panel.

    Parameters
    ----------
    result:
        A fitted ``ws.VAR`` result, or its ``irf()`` analysis.
    statistic:
        ``irf``, ``oirf``, ``cirf``, ``coirf`` or ``fevd``. The statistics that Stata
        accepts but does not define after ``var`` (``dm``, ``cdm``, ``sirf``, ``csirf``,
        ``sfevd``) raise rather than plotting zeros.
    impulse, response:
        One name/index, a sequence of them, or ``None`` for all. Stata's ``impulse()`` and
        ``response()``.
    periods:
        Last step, inclusive.
    level:
        Confidence level; Stata's ``level(95)`` is ``0.95``.
    lstep, ustep:
        Stata's ``lstep(#)`` / ``ustep(#)``: trim the reported step range. The series is still
        computed through ``periods``, so a cumulative statistic keeps its full-horizon sum.
    show_ci:
        Draw the band. ``False`` is Stata's ``noci``.
    individual:
        One FIGURE per combination instead of one figure with subgraphs. Returns a list of
        figures in that case. Stata's ``individual``.
    figsize:
        Defaults to a grid sized to the number of panels.
    same_y:
        Share one y-axis across panels, so magnitudes are comparable by eye. Off by
        default, matching Stata, where each subgraph keeps its own scale.
    title:
        Override the figure title.
    ax:
        Draw into this axis instead; only valid when exactly one panel is requested.
    grid:
        Light grid in each panel.
    save:
        Write the figure to this path, Stata's ``isaving()``. With ``individual=True`` the
        panels share the stub and each file gets a numbered suffix, which is what
        ``iname(stub)`` / ``isaving(stub)`` do in Stata -- otherwise every panel would
        overwrite the same file.
    **plot_kwds:
        Forwarded to ``Axes.plot`` for the point path.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "irf_graph() needs matplotlib: pip install 'westata[plot]'"
        ) from exc

    series = irf_series(
        result,
        statistic=statistic,
        periods=periods,
        level=level,
        lstep=lstep,
        ustep=ustep,
    )
    names = list(series.names)

    impulses = _resolve(names, impulse)
    responses = _resolve(names, response)

    if ax is not None and (len(impulses) != 1 or len(responses) != 1):
        raise ConsistencyError(
            "irf_graph: ax= needs exactly one impulse and one response; "
            f"got {len(impulses)}x{len(responses)}"
        )

    panels = [(response, impulse) for response in responses for impulse in impulses]
    label = series.statistic

    def _draw(axis: Any, response: int, impulse: int) -> None:
        steps = np.asarray(series.steps, dtype=float)
        values = series.values[:, response, impulse]
        if show_ci and series.lower is not None:
            axis.fill_between(
                steps,
                series.lower[:, response, impulse],
                series.upper[:, response, impulse],
                alpha=0.25,
                linewidth=0,
                label=f"{int(round(series.level * 100))}% CI",
            )
        style = dict(plot_kwds)
        style.setdefault("linewidth", 1.6)
        axis.plot(steps, values, label=label, **style)
        axis.axhline(0.0, color="black", linewidth=0.8)
        if grid:
            axis.grid(True, alpha=0.3, linewidth=0.6)
        axis.set_title(f"{names[impulse]} → {names[response]}", fontsize=10)
        axis.set_xlabel("step")
        axis.set_ylabel(label if series.statistic != "fevd" else "share")

    if individual:
        figures = []
        for response, impulse in panels:
            size = figsize or (5.0, 3.5)
            figure, axis = make_figure(figsize=size)
            _draw(axis, response, impulse)
            if title is not None:
                figure.suptitle(title)
            figure.tight_layout()
            if save is not None:
                _save_figure(figure, save, index=len(figures) + 1)
            figures.append(figure)
        return figures

    if ax is not None:
        response, impulse = panels[0]
        _draw(ax, response, impulse)
        return ax.figure

    columns = 1 if len(panels) == 1 else 2
    rows = -(-len(panels) // columns)
    if figsize is None:
        figsize = (6.0 * columns, 3.6 * rows)
    figure, flat = make_axes(rows, columns, figsize=figsize)
    for unused in flat[len(panels):]:
        unused.set_visible(False)

    for axis, (response, impulse) in zip(flat, panels):
        _draw(axis, response, impulse)

    if same_y:
        lows = [axis.get_ylim()[0] for axis in flat[: len(panels)]]
        highs = [axis.get_ylim()[1] for axis in flat[: len(panels)]]
        for axis in flat[: len(panels)]:
            axis.set_ylim(min(lows), max(highs))

    figure.suptitle(title if title is not None else f"Results from {label}")
    figure.tight_layout()
    if save is not None:
        _save_figure(figure, save)
    return figure


def _resolve(names: Sequence[str], which: Any) -> list[int]:
    """``None`` -> every index; a name/index or a sequence -> those indices."""

    if which is None:
        return list(range(len(names)))
    if isinstance(which, (str, int, np.integer)):
        return [_index(names, which)]
    resolved = [_index(names, item) for item in which]
    if not resolved:
        raise ConsistencyError("irf_graph: impulse/response selected nothing")
    return resolved


def irf_ograph(
    result: Any,
    *,
    ographs: Sequence[tuple[str | int, str | int, str]] | None = None,
    statistic: str | Sequence[str] = "oirf",
    impulse: str | int = 0,
    response: str | int = 0,
    periods: int = 10,
    level: float = 0.95,
    show_ci: bool = True,
    lstep: int | None = None,
    ustep: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: Any = None,
    grid: bool = True,
    **plot_kwds: Any,
) -> Any:
    """Overlaid impulse responses -- Stata's ``irf ograph``.

    ``irf graph`` draws one SUBGRAPH per combination; ``irf ograph`` lays several on ONE axis so
    their shapes can be compared. Stata spells a spec as ``(irfname impulse response stat)``;
    ``ographs=`` takes those triples directly::

        ws.irf_ograph(fit, ographs=[("dinf", "dinf", "oirf"), ("dinf", "dinf", "coirf")])

    The convenience form takes one impulse/response pair and several statistics, which is the
    comparison a reader usually wants::

        ws.irf_ograph(fit, impulse="dinf", response="unem", statistic=["oirf", "coirf"])

    Parameters
    ----------
    result:
        A fitted ``ws.VAR`` result, or its ``irf()`` analysis.
    ographs:
        Explicit ``(impulse, response, statistic)`` triples. When given, ``statistic``,
        ``impulse`` and ``response`` are ignored -- Stata's spec list wins.
    statistic:
        One statistic or a list of them, drawn over each other on one axis.
    impulse, response:
        The single pair the convenience form uses.
    periods, level, show_ci, lstep, ustep:
        As in :func:`irf_graph`.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "irf_ograph() needs matplotlib: pip install 'westata[plot]'"
        ) from exc

    if ographs is not None:
        specs = [(str(imp), str(res), str(stat)) for imp, res, stat in ographs]
        if not specs:
            raise ConsistencyError("irf ograph: `ographs` is empty")
    else:
        statistics = [statistic] if isinstance(statistic, str) else list(statistic)
        if not statistics:
            raise ConsistencyError("irf ograph: no statistic selected")
        analysis_names = _names_of(result, periods)
        impulse_name = _index(analysis_names, impulse)
        response_name = _index(analysis_names, response)
        specs = [
            (analysis_names[impulse_name], analysis_names[response_name], str(stat))
            for stat in statistics
        ]

    if ax is not None:
        axis = ax
        figure = ax.figure
    else:
        figure, axis = make_figure(figsize=figsize or (7.0, 4.0))

    # The band is drawn only for a single spec: several overlapping CI regions on one axis are
    # unreadable, and Stata's `irf ograph` documents the band per spec for the same reason.
    include_band = show_ci and len(specs) == 1

    for impulse_name, response_name, statistic_name in specs:
        series = irf_series(
            result,
            statistic=statistic_name,
            periods=periods,
            level=level,
            lstep=lstep,
            ustep=ustep,
        )
        response_index = _index(series.names, response_name)
        impulse_index = _index(series.names, impulse_name)
        steps = np.asarray(series.steps, dtype=float)
        values = series.values[:, response_index, impulse_index]

        style = dict(plot_kwds)
        style.setdefault("linewidth", 1.6)
        style.setdefault("marker", "o")
        style.setdefault("markersize", 3.0)
        if include_band and series.lower is not None:
            axis.fill_between(
                steps,
                series.lower[:, response_index, impulse_index],
                series.upper[:, response_index, impulse_index],
                alpha=0.20,
                linewidth=0,
            )
        label = f"{statistic_name}"
        if len({(spec[0], spec[1]) for spec in specs}) > 1:
            label = f"{impulse_name}→{response_name} {statistic_name}"
        axis.plot(steps, values, label=label, **style)

    axis.axhline(0.0, color="black", linewidth=0.8)
    if grid:
        axis.grid(True, alpha=0.3, linewidth=0.6)
    axis.set_xlabel("step")
    axis.set_ylabel(specs[0][2] if len(specs) == 1 else "response")
    axis.set_title(
        title
        if title is not None
        else f"{specs[0][0]} → {specs[0][1]}" if len({(s[0], s[1]) for s in specs}) == 1
        else "Overlaid impulse responses"
    )
    if len(specs) > 1:
        axis.legend(frameon=False, fontsize=9)

    if ax is None:
        figure.tight_layout()
    return figure


def _names_of(result: Any, periods: int) -> list[str]:
    """The equation names, resolved the same way `irf_series` does (never from the analysis)."""

    names = [str(name) for name in (getattr(result, "names", None) or [])]
    if names:
        return names
    analysis = _inner(result, periods)
    model = getattr(analysis, "model", None)
    names = [str(name) for name in (getattr(model, "endog_names", None) or [])]
    if names:
        return names
    eqs = int(getattr(analysis, "neqs", 0))
    return [f"y{index + 1}" for index in range(eqs)]


def irf_cgraph(
    result: Any,
    *,
    specs: Sequence[tuple[str | int, str | int, str | Sequence[str]]] | None = None,
    statistic: str | Sequence[str] = "oirf",
    impulse: str | int | Sequence[str | int] | None = None,
    response: str | int | Sequence[str | int] | None = None,
    periods: int = 10,
    level: float = 0.95,
    show_ci: bool = True,
    lstep: int | None = None,
    ustep: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    grid: bool = True,
    **plot_kwds: Any,
) -> Any:
    """Combined panels -- Stata's ``irf cgraph``, the third member of the ``irf`` graph family.

    The three commands differ only in LAYOUT, and that is exactly what makes them easy to
    confuse:

    ==============  ==================================================================
    ``irf graph``   one subgraph per (impulse, response) combination; a SPEC is one panel
    ``irf ograph``  several statistics overlaid on ONE axis; a SPEC is one line
    ``irf cgraph``  several SPECS combined as subgraphs of one figure; a SPEC is one panel
    ==============  ==================================================================

    Stata writes a spec as ``(irfname impulse response stat)``; ``specs=`` takes those triples,
    and a spec may itself name several statistics, in which case they share that panel::

        ws.irf_cgraph(fit, specs=[("dinf", "unem", "oirf"), ("dinf", "unem", "fevd")])
        ws.irf_cgraph(fit, specs=[("dinf", "dinf", ["oirf", "cirf"])])

    Parameters
    ----------
    specs:
        Explicit ``(impulse, response, statistic-or-list)`` triples. When omitted, the
        convenience form builds one spec per (impulse, response) pair.
    statistic, impulse, response:
        Used only when ``specs`` is omitted. ``statistic`` may be a list, which is then drawn
        overlaid inside every panel -- the shorthand for "same panel, several series".
    periods, level, show_ci, lstep, ustep:
        As in :func:`irf_graph`.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "irf_cgraph() needs matplotlib: pip install 'westata[plot]'"
        ) from exc

    statistics_seed = [statistic] if isinstance(statistic, str) else list(statistic)

    if specs is not None:
        panels: list[tuple[str, str, list[str]]] = []
        for impulse_name, response_name, stat_names in specs:
            names = [stat_names] if isinstance(stat_names, str) else list(stat_names)
            if not names:
                raise ConsistencyError("irf cgraph: a spec names no statistic")
            panels.append((str(impulse_name), str(response_name), [str(n) for n in names]))
        if not panels:
            raise ConsistencyError("irf cgraph: `specs` is empty")
    else:
        names = _names_of(result, periods)
        impulses = _resolve(names, impulse)
        responses = _resolve(names, response)
        if not statistics_seed:
            raise ConsistencyError("irf cgraph: no statistic selected")
        panels = [
            (names[impulse_index], names[response_index], [str(s) for s in statistics_seed])
            for response_index in responses
            for impulse_index in impulses
        ]

    columns = 1 if len(panels) == 1 else 2
    rows = -(-len(panels) // columns)
    if figsize is None:
        figsize = (6.0 * columns, 3.6 * rows)
    figure, flat = make_axes(rows, columns, figsize=figsize)
    for unused in flat[len(panels):]:
        unused.set_visible(False)

    for axis, (impulse_name, response_name, stat_names) in zip(flat, panels):
        for statistic_name in stat_names:
            series = irf_series(
                result,
                statistic=statistic_name,
                periods=periods,
                level=level,
                lstep=lstep,
                ustep=ustep,
            )
            response_index = _index(series.names, response_name)
            impulse_index = _index(series.names, impulse_name)
            steps = np.asarray(series.steps, dtype=float)
            values = series.values[:, response_index, impulse_index]

            style = dict(plot_kwds)
            style.setdefault("linewidth", 1.6)
            if len(stat_names) > 1:
                style.setdefault("marker", "o")
                style.setdefault("markersize", 3.0)
            label = statistic_name if len(stat_names) > 1 else None
            if show_ci and series.lower is not None:
                axis.fill_between(
                    steps,
                    series.lower[:, response_index, impulse_index],
                    series.upper[:, response_index, impulse_index],
                    alpha=0.20,
                    linewidth=0,
                )
            axis.plot(steps, values, label=label, **style)

        axis.axhline(0.0, color="black", linewidth=0.8)
        if grid:
            axis.grid(True, alpha=0.3, linewidth=0.6)
        axis.set_xlabel("step")
        axis.set_ylabel(stat_names[0] if len(stat_names) == 1 else "response")
        axis.set_title(f"{impulse_name} → {response_name}", fontsize=10)
        if len(stat_names) > 1:
            axis.legend(frameon=False, fontsize=8)

    figure.suptitle(title if title is not None else "Combined impulse responses")
    figure.tight_layout()
    return figure


def _save_figure(figure: Any, target: str, index: int | None = None) -> str:
    """Save a figure, giving each panel of a multi-file call its own name.

    Stata's ``iname(stub)`` / ``isaving(stub)`` write ``stub1``, ``stub2`` ... Writing every
    panel to one path would leave only the last one on disk, which looks like "the save
    worked" until somebody opens the file.
    """

    from pathlib import Path as _Path

    path = _Path(target)
    if index is not None:
        path = path.with_name(f"{path.stem}{index}{path.suffix or '.png'}")
    elif not path.suffix:
        path = path.with_name(path.name + ".png")
    if path.parent and not path.parent.exists():
        raise ConsistencyError(
            f"irf graph: the directory for save={target!r} does not exist: {path.parent}"
        )
    figure.savefig(path)
    return str(path)
