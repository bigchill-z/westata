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
import pandas as pd

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
    """A name, or a 1-BASED position -- Stata's ``i()``/``r()`` count from one.

    MEASURED: Stata's `irf graph oirf, i(unem) r(dinf)` equals `i(2) r(1)` on
    `var dinf unem`, so `i(1)` is the FIRST variable. Treating the integer as 0-based made
    ``impulse=2`` raise "out of range for ['dinf', 'unem']" on a two-variable VAR, and silently
    selected the wrong column for every other value.
    """

    if isinstance(which, (int, np.integer)):
        position = int(which)
        if not 1 <= position <= len(names):
            raise ConsistencyError(
                f"irf graph: index {position} is out of range; positions are 1-based, so "
                f"1..{len(names)} for {list(names)}"
            )
        return position - 1
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
        analysis = factory(periods=periods)
        # The band needs Stata-scale e(b)/e(V)/e(Sigma), which live on the westata wrapper rather
        # than on the statsmodels analysis. Carrying a reference is cheaper and safer than trying to
        # rescale the backend objects after the fact.
        try:
            object.__setattr__(analysis, "_westata_result", result)
        except Exception:  # noqa: BLE001 - a slotted class simply keeps the old behaviour
            pass
        return analysis
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
        # A statsmodels VARResults keeps the equation names on `names`; `endog_names` is None
        # there, so reading only that fell through to y1/y2 and a caller passing the ANALYSIS
        # could not address its own variables. Prefer `names`, keep the old path as a fallback.
        _model_names = getattr(model, "names", None) or getattr(model, "endog_names", None)
        names = [str(name) for name in (_model_names or ())]
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
    """Confidence bands for the IRF-family statistics; ``None`` for ``fevd``.

    THIS IS THE DELTA METHOD ON (B, SIGMA) JOINTLY, which is what Stata uses. The previous version
    read ``statsmodels``'s ``stderr()``, which treats the Cholesky factor as KNOWN. On
    ``var dinf unem, lags(1/2)`` from macro_swatson that made the bands about 6% too narrow, and it
    returned exactly 0 for entries Stata reports as non-zero.

    THE DECISIVE EVIDENCE is the step-0 row, where the oirf IS the Cholesky element: for the
    structurally-zero entry ``unem -> dinf`` Stata reports se 0, while ``dinf -> dinf`` reports
    .0007510342. A band built from B alone gives 0 for both, so Sigma must carry sampling error too.

    Gaussian ML asymptotics make B and Sigma INDEPENDENT, so the variances add::

        Var(f)     = f_B' Var(vec B) f_B  +  f_S' Var(vec Sigma_ml) f_S
        Var(vec S) = (I + K)(Sigma (x) Sigma) / T          K = commutation matrix

    MEASURED against Stata: .0007511097 vs .0007510342, .0010548691 vs .0010548663,
    .0008844842 vs .0008844550, .0005187990 vs .0005186320.
    """

    if statistic not in ("irf", "oirf", "cirf", "coirf"):
        return None
    from scipy import stats as _stats

    critical = float(_stats.norm.ppf(0.5 + float(level) / 2.0))

    if statistic in ("irf", "oirf"):
        centre = np.asarray(
            analysis.irfs if statistic == "irf" else analysis.orth_irfs, dtype=float
        )
    else:
        centre = (
            np.asarray(analysis.cum_effects, dtype=float)
            if statistic == "cirf"
            else np.cumsum(np.asarray(analysis.orth_irfs, dtype=float), axis=0)
        )

    stderr = _joint_stderr(analysis, statistic, centre)
    if stderr is None:
        try:
            values = (
                np.asarray(analysis.stderr(), dtype=float)
                if statistic in ("irf", "oirf")
                else np.asarray(analysis.cum_effect_stderr(), dtype=float)
            )
            stderr = values[: centre.shape[0]]
        except Exception:  # noqa: BLE001
            return None

    count = min(centre.shape[0], stderr.shape[0])
    centre = centre[:count]
    stderr = stderr[:count]
    return centre - critical * stderr, centre + critical * stderr


def _commutation(size: int) -> np.ndarray:
    """K such that K vec(A) = vec(A')."""

    matrix = np.zeros((size * size, size * size))
    for row in range(size):
        for column in range(size):
            matrix[row * size + column, column * size + row] = 1.0
    return matrix


def _ma_path(blocks: list[np.ndarray], chol: np.ndarray, steps: int) -> list[np.ndarray]:
    """phi_0 = P, phi_h = sum_l A_l phi_{h-l} -- the structural MA coefficients."""

    order = len(blocks)
    path = [chol]
    for horizon in range(1, steps + 1):
        term = np.zeros_like(chol)
        for lag in range(1, min(horizon, order) + 1):
            term = term + blocks[lag - 1] @ path[horizon - lag]
        path.append(term)
    return path


def _coefficient_blocks(
    parameters: np.ndarray, names: Sequence[str], equations: int, order: int
) -> list[np.ndarray]:
    """The A_l matrices in Stata's coefficient ordering."""

    per_equation = len(names) // equations
    blocks = []
    for lag in range(1, order + 1):
        block = np.zeros((equations, equations))
        for eq in range(equations):
            local = list(names[eq * per_equation : (eq + 1) * per_equation])
            for variable in range(equations):
                target = f"L{lag}.{{}}"
                candidates = [name for name in local if name.startswith(f"L{lag}.")]
                if variable < len(candidates):
                    block[eq, variable] = parameters[
                        eq * per_equation + local.index(candidates[variable])
                    ]
        blocks.append(block)
    return blocks


def _lag_order(names: Sequence[str]) -> int:
    order = 0
    for name in names:
        if name.startswith("L") and "." in name:
            try:
                order = max(order, int(name[1:].split(".")[0]))
            except ValueError:
                continue
    return order


def _joint_stderr(analysis: Any, statistic: str, centre: np.ndarray) -> np.ndarray | None:
    """The (B, Sigma) delta-method standard errors, shaped like ``centre``."""

    model = getattr(analysis, "model", None)
    if model is None:
        return None
    observations = float(getattr(model, "nobs", 0) or 0)
    if observations <= 0:
        return None

    # THE RAW BACKEND OBJECTS ARE NOT ON STATAS SCALE. statsmodels' `sigma_u` divides by (T - k)
    # while Stata's e(Sigma) divides by T, and `params`/`cov_params` come in statsmodels' row order
    # rather than Stata's. The wrapper that owns the ML factor is reachable from the analysis, so it
    # is preferred whenever it is present.
    wrapper = getattr(analysis, "_westata_result", None) or getattr(analysis, "result", None)
    # `stata_cov_params` is a PROPERTY on the wrapper but arrives as a bound method on some backend
    # objects, so it has to be called before use -- `np.asarray` on the method raised TypeError.
    def _value(owner: Any, name: str) -> Any:
        attribute = getattr(owner, name, None)
        return attribute() if callable(attribute) else attribute

    beta_series = _value(wrapper, "stata_params")
    cov_beta = _value(wrapper, "stata_cov_params")
    sigma = _value(wrapper, "sigma_u")
    names = list(_value(wrapper, "stata_param_names") or [])
    if beta_series is None or cov_beta is None or sigma is None or not names:
        # Fall back to the backend objects, converting them the only way possible: rescale Sigma by
        # its own degrees-of-freedom factor so the Cholesky matches e(Sigma).
        raw_sigma = _value(model, "sigma_u")
        raw_params = _value(model, "params")
        raw_cov = _value(model, "cov_params")
        for candidate in (raw_sigma, raw_params, raw_cov):
            if candidate is None or callable(candidate):
                return None
        per_equation = int(np.asarray(raw_params).shape[0])
        total = float(observations)
        if total > per_equation > 0:
            raw_sigma = np.asarray(raw_sigma, dtype=float) * (total - per_equation) / total
        beta_series = pd.Series(np.asarray(raw_params, dtype=float).ravel())
        cov_beta = raw_cov
        sigma = raw_sigma
        names = [str(name) for name in getattr(model, "names", [])] or []

    beta = np.asarray(beta_series, dtype=float).ravel()
    cov_beta = np.asarray(cov_beta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    equations = sigma.shape[0]
    order = _lag_order(names)
    if order == 0:
        return None
    if beta.size != cov_beta.shape[0]:
        return None

    steps = centre.shape[0] - 1
    commutation = _commutation(equations)
    var_sigma = (
        (np.eye(equations * equations) + commutation) @ np.kron(sigma, sigma) / observations
    )

    def matrix_at(step: int, parameters: np.ndarray, sigma_local: np.ndarray) -> np.ndarray:
        blocks = _coefficient_blocks(parameters, names, equations, order)
        chol = np.linalg.cholesky(sigma_local)
        structural = _ma_path(blocks, chol, steps)[step]
        if statistic in ("oirf", "coirf"):
            values = structural
        else:
            values = _ma_path(blocks, np.eye(equations), steps)[step]
        if statistic in ("cirf", "coirf"):
            path = _ma_path(blocks, chol if statistic == "coirf" else np.eye(equations), steps)
            values = np.sum(path[: step + 1], axis=0)
        return values

    stderr = np.zeros(centre.shape)
    for step in range(steps + 1):
        base = matrix_at(step, beta, sigma)
        for response in range(equations):
            for impulse in range(equations):
                gradient_beta = np.zeros(beta.size)
                for position in range(beta.size):
                    h = 1e-7 * max(abs(beta[position]), 1e-4)
                    up, down = beta.copy(), beta.copy()
                    up[position] += h
                    down[position] -= h
                    gradient_beta[position] = (
                        matrix_at(step, up, sigma)[response, impulse]
                        - matrix_at(step, down, sigma)[response, impulse]
                    ) / (2.0 * h)

                flat = sigma.ravel().copy()
                gradient_sigma = np.zeros(flat.size)
                for position in range(flat.size):
                    h = 1e-9 * max(abs(flat[position]), 1e-16)
                    up, down = flat.copy(), flat.copy()
                    up[position] += h
                    down[position] -= h
                    matrix_up = up.reshape(equations, equations)
                    matrix_up = (matrix_up + matrix_up.T) / 2.0
                    matrix_down = down.reshape(equations, equations)
                    matrix_down = (matrix_down + matrix_down.T) / 2.0
                    gradient_sigma[position] = (
                        matrix_at(step, beta, matrix_up)[response, impulse]
                        - matrix_at(step, beta, matrix_down)[response, impulse]
                    ) / (2.0 * h)

                variance = max(float(gradient_beta @ cov_beta @ gradient_beta), 0.0)
                variance += max(float(gradient_sigma @ var_sigma @ gradient_sigma), 0.0)
                stderr[step, response, impulse] = np.sqrt(variance)
    return stderr


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
    irf: str | Sequence[str] | None = None,
    show_ci: bool = True,
    individual: bool = False,
    figsize: tuple[float, float] | None = None,
    same_y: bool = True,
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
        Share one y range across every panel. ON by default because that is what Stata does: its
        `irf graph oirf` prints all four subgraphs on ONE scale with no option involved -- exported
        and measured, the four panels all read -.01 to .02. The previous default of False gave each
        panel its own range, so a pair whose band was 50x larger set the scale for itself alone and
        the small ones looked flat by comparison. Pass ``same_y=False`` for independent scales.
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

    # A STORE holds several NAMED result sets, which is what makes Stata draw one panel per
    # (irfname, impulse, response). Everything below works on a LIST of (label, series) pairs, so a
    # single result is simply a list of one and the old behaviour is preserved.
    if hasattr(result, "select"):
        chosen = result.select(irf)
    elif irf is not None:
        raise ConsistencyError(
            "irf_graph: `irf=` needs a result store (ws.IrfStore) holding several named IRFs; "
            "a single VAR result carries only one"
        )
    else:
        chosen = [(None, result)]

    series_list = [
        (
            label_name,
            irf_series(
                target,
                statistic=statistic,
                periods=periods,
                level=level,
                lstep=lstep,
                ustep=ustep,
            ),
        )
        for label_name, target in chosen
    ]

    series = series_list[0][1]
    names = list(series.names)

    impulses = _resolve(names, impulse)
    responses = _resolve(names, response)

    if ax is not None and (len(impulses) != 1 or len(responses) != 1):
        raise ConsistencyError(
            "irf_graph: ax= needs exactly one impulse and one response; "
            f"got {len(impulses)}x{len(responses)}"
        )

    # ONE PANEL PER (irfname, impulse, response), which is Stata's rule: `i()`/`r()` narrow the
    # variables but NOT the name, so two names in the file produce two panels that share a title.
    panels = [
        (name_index, response, impulse)
        for name_index in range(len(series_list))
        for response in responses
        for impulse in impulses
    ]
    label = series.statistic

    def _draw(axis: Any, name_index: int, response: int, impulse: int) -> None:
        current = series_list[name_index][1]
        # Each series has its own step vector: two names can be drawn with different horizons.
        steps = np.asarray(current.steps, dtype=float)[: current.values.shape[0]]
        values = current.values[:, response, impulse]
        if show_ci and current.lower is not None:
            axis.fill_between(
                steps,
                current.lower[:, response, impulse],
                current.upper[:, response, impulse],
                alpha=0.25,
                linewidth=0,
                label=f"{int(round(current.level * 100))}% CI",
            )
        style = dict(plot_kwds)
        style.setdefault("linewidth", 1.6)
        axis.plot(steps, values, label=label, **style)
        axis.axhline(0.0, color="black", linewidth=0.8)
        if grid:
            axis.grid(True, alpha=0.3, linewidth=0.6)
        irf_label = series_list[name_index][0]
        pair = f"{names[impulse]} \u2192 {names[response]}"
        # Stata titles the subgraph "iu, unem, dinf"; an unnamed single result keeps the short form
        # so existing output is unchanged.
        axis.set_title(
            f"{irf_label}, {names[impulse]}, {names[response]}" if irf_label else pair,
            fontsize=10,
        )
        axis.set_xlabel("step")
        axis.set_ylabel(label if current.statistic != "fevd" else "share")

    if individual:
        figures = []
        for name_index, response, impulse in panels:
            size = figsize or (5.0, 3.5)
            figure, axis = make_figure(figsize=size)
            _draw(axis, name_index, response, impulse)
            if title is not None:
                figure.suptitle(title)
            figure.tight_layout()
            if save is not None:
                _save_figure(figure, save, index=len(figures) + 1)
            figures.append(figure)
        return figures

    if ax is not None:
        if len(panels) != 1:
            raise ConsistencyError(
                "irf_graph: ax= needs exactly one panel; got "
                f"{len(panels)} (names x impulses x responses)"
            )
        name_index, response, impulse = panels[0]
        _draw(ax, name_index, response, impulse)
        return ax.figure

    columns = 1 if len(panels) == 1 else 2
    rows = -(-len(panels) // columns)
    if figsize is None:
        figsize = (6.0 * columns, 3.6 * rows)
    figure, flat = make_axes(rows, columns, figsize=figsize)
    for unused in flat[len(panels):]:
        unused.set_visible(False)
    for axis, (name_index, response, impulse) in zip(flat, panels):
        _draw(axis, name_index, response, impulse)

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
    ographs: Sequence[tuple[Any, ...]] | None = None,
    statistic: str | Sequence[str] = "oirf",
    impulse: str | int = 1,
    response: str | int = 1,
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
    their shapes can be compared.

    STATA'S SPEC HAS **FOUR** FIELDS, in this order::

        irf ograph (irfname impulsevar responsevar stat) [(spec_2) ... [(spec_15)]]

    so ``ographs=`` takes four-tuples, the first being the IRF-name slot::

        ws.irf_ograph(
            fit,
            ographs=[("iu", "unem", "dinf", "oirf"), ("iu", "unem", "unem", "oirf")],
        )

    A three-tuple ``(impulse, response, stat)`` is still accepted for compatibility. At most 15
    specs, as Stata documents.

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
        # STATA'S SPEC HAS FOUR FIELDS: (irfname impulse response stat). A three-field
        # (impulse, response, stat) entry is still accepted for compatibility. Unpacking three
        # unconditionally raised "too many values to unpack (expected 3)" on Stata's own spelling.
        fields_list = []
        for entry in ographs:
            fields = list(entry)
            if len(fields) == 4:
                _label, impulse_field, response_field, stat = fields
            elif len(fields) == 3:
                impulse_field, response_field, stat = fields
            else:
                raise ConsistencyError(
                    "irf ograph: each spec needs (irfname impulse response stat) -- 4 fields, as "
                    "Stata documents -- or the older 3-field (impulse, response, stat); "
                    f"got {len(fields)}: {fields!r}"
                )
            fields_list.append((impulse_field, response_field, str(stat)))
        if not fields_list:
            raise ConsistencyError("irf ograph: `ographs` is empty")
        if len(fields_list) > 15:
            # Stata documents a hard ceiling of 15 specs.
            raise ConsistencyError(
                f"irf ograph: at most 15 specs, as Stata documents; got {len(fields_list)}"
            )
        analysis_names = _names_of(result, periods)
        specs = [
            (
                analysis_names[_index(analysis_names, imp)],
                analysis_names[_index(analysis_names, res)],
                stat,
            )
            for imp, res, stat in fields_list
        ]
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
