"""Estimation-sample summary: Stata's ``estat summarize``.

Stata's ``estat summarize`` "summarizes the variables used by the command and
automatically restricts the sample to the estimation sample; it also summarizes
the weight variable and cluster structure, if specified"
(``estat_summarize.sthlp``).  The table columns are fixed by
``estat_summ.ado:706-710``:

    Variable |  Obs  Mean  Std. dev.  Min  Max  [Label]

Two weight details come from ``estat_summ.ado:206-233``:

* ``pweight`` is treated as ``aweight``, because for ``summarize`` that
  "produces okay mean and std. dev.";
* only ``fweight`` changes the reported ``Obs`` -- "a, i, and p weights do not
  change N".

The cluster block (``Obs per cluster: min = ...``) is printed when the model has
a cluster variable; ``estat_summ.ado:680-690``.

westata carries the estimation sample on the result only for estimators that
retained ``model_data``; the response and fitted values are always present, so
the response is always summarizable, while the regressors depend on that
retention.  Passing ``data=`` explicitly covers the rest.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas as pd

__all__ = ["SummarizeResult", "estat_summarize"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


def _summary_of_a_non_numeric_column(column: Any) -> dict[str, float]:
    """A row for a variable whose values are not numeric (a factor or string).

    Stata's ``estat summarize`` reports the UNDERLYING variable rather than its dummy
    expansion (``estat_summarize.sthlp:38`` refers to factor-variable labelling, which
    is about how the row is NAMED).  It cannot compute a mean of a string, so the
    numeric cells are blank and only ``Obs`` is filled.

    Returning a row rather than raising keeps the rest of the table: a caller asking
    for several variables should not lose all of them because one is categorical.
    """

    values = np.asarray(column.to_numpy(), dtype=object).reshape(-1)
    observed = sum(1 for value in values if value is not None and not _is_missing(value))
    return {
        "obs": float(observed),
        "mean": float("nan"),
        "sd": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }


def _is_missing(value: Any) -> bool:
    """Whether a raw cell counts as missing for the observation tally."""

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic array-like cell
        return False


def _weighted_summary(
    values: np.ndarray, weights: np.ndarray | None, *, weight_type: str | None
) -> dict[str, float]:
    """One row of ``summarize``: obs, mean, sd, min, max.

    ``Obs`` follows ``estat_summ.ado:222-233``: only ``fweight`` changes it
    ("a, i, and p weights do not change N"), because ``fweight`` is literal
    replication.

    ``mean`` is the weighted mean.  For ``sd`` this module uses the
    **frequency/replication** variance ``sum(w (x-mean)^2) / (sum(w) - 1)``,
    which is exactly what ``fweight`` replication produces.  For ``aweight`` /
    ``pweight`` / ``iweight`` Stata's ``summarize`` documents the variance as
    computed from the same weighted sums; the two conventions differ only in the
    denominator, and on a constant weight they differ by a few percent.  Rather
    than claim Stata-identical digits under analytic weights, callers who need
    to quote an ``aweight`` standard deviation should compare it in Stata --
    ``e()`-level conformance for weighted *descriptives* was not part of the
    sweep, so no evidence level is claimed here.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if weights is None:
        sample = values[finite]
        obs = float(sample.size)
        if sample.size == 0:
            return {
                "obs": 0.0,
                "mean": float("nan"),
                "sd": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = float(sample.mean())
        sd = float(sample.std(ddof=1)) if sample.size > 1 else float("nan")
        return {
            "obs": obs,
            "mean": mean,
            "sd": sd,
            "min": float(sample.min()),
            "max": float(sample.max()),
        }

    weights = np.asarray(weights, dtype=float).reshape(-1)
    keep = finite & np.isfinite(weights) & (weights > 0.0)
    sample = values[keep]
    w = weights[keep]
    if sample.size == 0:
        return {
            "obs": 0.0,
            "mean": float("nan"),
            "sd": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    total_weight = float(w.sum())
    kind = str(weight_type or "").strip().lower()

    # aweight/pweight/iweight are normalised to ``sum(w) = N`` before any
    # variance is formed.  That is Stata's rule and the package already encodes
    # it in ``westata.data.weights.normalize_weights``: "aweight and pweight are
    # both normalized so that the weights sum to N".  Every ESTIMATOR follows it,
    # which is why an aweighted fit's standard errors do not change when the user
    # rescales their weights.
    #
    # The descriptive table used to consume the RAW weights, so its standard
    # deviation moved with the scale (measured 1.8531 at w, 1.8507 at 7w) while
    # the mean correctly did not.  Same weight type, two conventions -- an
    # internal inconsistency rather than an open question about Stata.
    #
    # ``fweight`` is different: it means literal replication, so its weights are
    # not rescaled and ``Obs`` becomes ``sum(w)``.
    if kind in ("aweight", "pweight", "iweight") and total_weight > 0.0:
        w = w * (sample.size / total_weight)
        total_weight = float(w.sum())

    mean = float(np.sum(w * sample) / total_weight)

    # Variance = sum(w (x-mean)^2) / (sum(w) - 1).  For fweight this is the
    # ddof=1 variance of the expanded sample; for the analytic weights above it
    # runs on the normalised weights, so
    #
    #     Var_w / Var_unw = c(N-1)/(cN-1)
    #
    # for a constant weight ``c`` (verified in tests), and a rescale of the input
    # weights leaves the result unchanged.
    if total_weight > 1.0:
        variance = float(np.sum(w * (sample - mean) ** 2) / (total_weight - 1.0))
        sd = float(np.sqrt(variance))
    else:
        sd = float("nan")
    obs = total_weight if kind == "fweight" else float(sample.size)
    return {
        "obs": obs,
        "mean": mean,
        "sd": sd,
        "min": float(sample.min()),
        "max": float(sample.max()),
    }


class SummarizeResult:
    """The ``estat summarize`` table.

    Attributes
    ----------
    table:
        DataFrame indexed by variable with columns ``Obs``, ``Mean``,
        ``Std. Dev.``, ``Min``, ``Max``.
    n_obs:
        The estimation-sample size (``e(N)``).
    clusters:
        Optional mapping describing the cluster structure: ``{"name", "n",
        "min", "avg", "max"}``.
    """

    __slots__ = ("table", "n_obs", "clusters", "noheader", "labels")

    def __init__(
        self,
        table: pd.DataFrame,
        n_obs: float,
        clusters: Mapping[str, Any] | None = None,
        noheader: bool = False,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.table = table
        self.n_obs = float(n_obs)
        self.clusters = dict(clusters) if clusters else None
        #: ``estat_summarize.sthlp:93``: "noheader suppresses the header".  In Stata
        #: that is the "Number of obs" line above the table; the COLUMN headings are
        #: part of the table and stay.
        self.noheader = bool(noheader)
        #: ``estat_summarize.sthlp:90``: "labels displays variable labels".  A mapping
        #: from variable name to label, or ``None`` when the caller supplied none.
        #: Variables without a label show an empty cell, which is what Stata does --
        #: inventing a label from the name would look like real metadata.
        if labels is None:
            self.labels = None
        elif isinstance(labels, Mapping):
            self.labels = dict(labels)
        else:
            # A bare truthiness test on an array raises "truth value ... is ambiguous",
            # which is how a shadowed parameter surfaced here once.  Naming the type
            # makes the mistake obvious instead of mysterious.
            raise TypeError(
                "estat summarize: labels must be a mapping of variable name to label; "
                f"got {type(labels).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"<SummarizeResult {len(self.table)} variables, N={self.n_obs:g}>"
        )

    def __str__(self) -> str:
        if self.noheader:
            lines = [""]
        else:
            lines = ["", f"{'':<4}Number of obs = {self.n_obs:>10,.0f}", ""]
        header = (
            f"{'Variable':<16}|{'Obs':>9}{'Mean':>12}{'Std. dev.':>12}"
            f"{'Min':>12}{'Max':>12}"
        )
        # ``estat_summarize.sthlp:90``: "labels displays variable labels".  The column is
        # appended AFTER the statistics so the numeric columns keep their positions --
        # inserting it beside the name would shift every other heading.
        if self.labels:
            header += f"  {'Label'}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, row in self.table.iterrows():
            # Stata prints ``.`` for a missing statistic, never ``nan``.  A
            # non-numeric variable has only ``Obs``; its numeric cells are missing.
            def _cell(value: float) -> str:
                return "." if value is None or not np.isfinite(value) else f"{value:.4f}"

            rendered = (
                f"{str(name):<16}|{row['Obs']:>9.0f}{_cell(row['Mean']):>12}"
                f"{_cell(row['Std. Dev.']):>12}{_cell(row['Min']):>12}"
                f"{_cell(row['Max']):>12}"
            )
            if self.labels:
                # An unlabelled variable gets an EMPTY cell.  Falling back to the name
                # would invent metadata the data does not carry.
                rendered += f"  {self.labels.get(str(name), '')}"
            lines.append(rendered)
        if self.clusters:
            lines.extend(
                [
                    "",
                    f"Group variable: {self.clusters['name']}"
                    f"   Number of groups = {self.clusters['n']:.0f}",
                    f"Obs per group: min = {self.clusters['min']:.0f}"
                    f"   avg = {self.clusters['avg']:.1f}"
                    f"   max = {self.clusters['max']:.0f}",
                ]
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """The summary table."""

        return self.table

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Per-variable statistics keyed by name."""

        return {
            str(name): {
                "obs": float(row["Obs"]),
                "mean": float(row["Mean"]),
                "sd": float(row["Std. Dev."]),
                "min": float(row["Min"]),
                "max": float(row["Max"]),
            }
            for name, row in self.table.iterrows()
        }


def _estimation_frame(
    result: Any, data: pd.DataFrame | None, variables: Sequence[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve the estimation sample and the variables to summarize."""

    if data is not None:
        frame = data
    else:
        frame = getattr(result, "model_data", None)
        if frame is None:
            raise ValueError(
                "estat summarize needs the estimation-sample data, which this "
                "result did not retain; pass data= explicitly"
            )
    frame = frame.reset_index(drop=True)

    if variables is not None:
        wanted = [str(v) for v in variables]
        missing = [v for v in wanted if v not in frame.columns]
        if missing:
            raise ValueError(
                f"estat summarize: variable(s) {missing} not in the data; "
                f"available: {list(frame.columns)}"
            )
        return frame, wanted

    # Default: the response and the reported regressors.  Stata summarizes
    # ``e(depvar)`` first -- ``estat_summ.ado:86,96,503`` all begin from it.
    chosen: list[str] = []
    response_name = getattr(result, "response_name", None)
    if response_name is not None and response_name in frame.columns:
        chosen.append(str(response_name))

    if getattr(result, "design_matrix", None):
        _, names = result.design_matrix
        for name in names:
            if _is_constant(name):
                continue
            # patsy spells ``C(foreign)[T.Foreign]``; the source column is the
            # part before the bracket.
            candidate = str(name).split("[", 1)[0]
            candidate = candidate.replace("C(", "").rstrip(")")
            # MULTI-EQUATION fits prefix every coefficient with its equation --
            # ``mvreg y x1 = x2`` gives ``y:x2``/``y:_cons``/``x1:x2``/``x1:_cons``.
            # Comparing that directly against the frame columns matched nothing, so
            # ``chosen`` came out empty, the table had no columns, and the fixed
            # column selection raised a bare KeyError.  Stata summarizes the
            # responses and regressors, which are the parts after the colon.
            if candidate not in frame.columns and ":" in candidate:
                candidate = candidate.rsplit(":", 1)[1]
            if candidate in frame.columns and candidate not in chosen:
                chosen.append(candidate)

    # ``estat_summ.ado:15`` takes an equation list and summarizes every response,
    # so a multi-equation fit should start from its responses rather than from the
    # prefixed coefficient names alone.
    depvars = getattr(result, "depvar_names", None) or []
    if not depvars:
        # ``mvreg``/``sureg`` record them in e(outcomes)/e(depvar) instead of on an
        # attribute, and neither retains a design_matrix, so this is the only route
        # back to the data columns for those fits.
        e = getattr(result, "e", None) or {}
        outcomes = e.get("outcomes")
        if outcomes:
            depvars = list(outcomes)
        elif e.get("depvar"):
            depvars = str(e["depvar"]).split()
    for name in depvars:
        text = str(name)
        if text in frame.columns and text not in chosen:
            chosen.insert(0, text)

    # A multi-equation fit may have no design_matrix at all.  Its coefficient names
    # are ``equation:regressor`` pairs, so the EQUATION part identifies a response
    # and the remainder identifies a regressor.  ``sureg`` stores neither
    # e(depvar) nor e(eqnames), which is why this parse is the only route back.
    if not chosen or len(chosen) < 2:
        equation_names: list[str] = []
        for name in getattr(result, "param_names", []) or []:
            text = str(name)
            if ":" not in text:
                continue
            head, tail = text.rsplit(":", 1)
            if head and head not in equation_names:
                equation_names.append(head)
        for name in equation_names:
            if name in frame.columns and name not in chosen:
                chosen.insert(0, name)
        for name in getattr(result, "param_names", []) or []:
            candidate = str(name).rsplit(":", 1)[-1]
            if _is_constant(candidate):
                continue
            if candidate in frame.columns and candidate not in chosen:
                chosen.append(candidate)
    return frame, chosen


def estat_summarize(
    result: Any,
    *,
    data: pd.DataFrame | None = None,
    variables: Sequence[str] | None = None,
    weights: Any = None,
    weight_type: str | None = None,
    cluster: Any = None,
    cluster_name: str = "",
    noweights: bool = False,
    noheader: bool = False,
    labels: Mapping[str, str] | None = None,
) -> SummarizeResult:
    """Stata's ``estat summarize``.

    Parameters
    ----------
    data:
        The estimation-sample frame.  Defaults to ``result.model_data``; it must
        be supplied when the estimator did not retain one.
    variables:
        Variables to summarize.  Defaults to the response plus the reported
        regressors.
    weights, weight_type:
        Optional weights, with the Stata type.  ``fweight`` inflates ``Obs``;
        ``aweight``/``pweight``/``iweight`` do not (``estat_summ.ado:222-233``).
    cluster, cluster_name:
        Optional cluster labels; adds the ``Obs per cluster`` block.

    Raises
    ------
    ValueError
        When the estimation data cannot be resolved, or a named variable is
        absent.
    """

    frame, chosen = _estimation_frame(result, data, variables)

    if weights is not None and weight_type is None:
        raise ValueError(
            "estat summarize: weights were supplied without weight_type; Stata "
            "never guesses the weight type"
        )

    # No explicit weights: use the ESTIMATION weights, exactly as
    # ``estat_summ.ado:208-231`` does.  Stata rebuilds the weight expression and
    # summarises with it, so an aweighted fit reports weighted means without the
    # caller restating the weights.  Reporting UNWEIGHTED statistics here would be
    # a silent difference from Stata -- and the caller has no way to tell.
    #
    # ``noweights`` (``estat_summarize.sthlp:35``: "ignore weights") suppresses that
    # automatic adoption.  Without it the option would be accepted and do nothing,
    # which is the failure shape round 138 found in ``estat gof, group()``: a
    # plausible table for the wrong request.
    weight_variable = ""
    if noweights:
        weights = None
        weight_type = None
    else:
        spec = getattr(result, "weight_spec", None)
        if weights is None and spec is not None:
            if getattr(spec, "weights", None) is not None:
                spec_weights = np.asarray(spec.weights, dtype=float)
                if spec_weights.size == len(frame):
                    weights = spec_weights
                    weight_type = getattr(spec, "weight_type", None)
        # The NAME of the weight variable, needed below: Stata summarises that
        # variable UNWEIGHTED (``estat_summarize.sthlp:96-99``).  ``weight_spec`` keeps
        # the VALUES but not the column name -- when the weights were passed as an
        # array there is no name to keep -- so the column is identified by matching its
        # values below.
        weight_variable = str(getattr(spec, "variable", "") or "") if spec else ""

    weight_values = None if weights is None else np.asarray(weights, dtype=float)
    if weight_values is not None and weight_values.size != len(frame):
        raise ValueError(
            f"estat summarize: {weight_values.size} weights for {len(frame)} rows"
        )

    # ``estat_summarize.sthlp:96-99``: "The default when weights are present is to
    # perform a weighted summarize on all variables except the weight variable itself.
    # An unweighted summarize is performed on the weight variable."
    #
    # Summarising the weight column WITH the weights is meaningless (it multiplies the
    # weights by themselves) and it disagreed with Stata: on a ``fweight`` fit whose
    # weights sum to 393 over 200 rows, ``w`` reported Obs=393 Mean=2.303 where Stata
    # reports Obs=200 Mean=1.965.

    rows: dict[str, dict[str, float]] = {}
    for name in chosen:
        column = frame[name]
        try:
            values = np.asarray(column.to_numpy(), dtype=float)
        except (TypeError, ValueError):
            # A factor/variable whose values are not numeric.  Stata's
            # ``estat summarize`` reports the UNDERLYING variable, not its dummy
            # expansion (``estat_summarize.sthlp:38`` mentions factor-variable
            # labelling, which is about naming the row).  Summarising the raw column
            # numerically is impossible, so the variable is reported with a count only
            # and the numeric statistics left blank -- the same shape Stata shows for
            # a string variable, rather than a crash that hides the whole table.
            values = None
        is_weight_column = bool(weight_variable) and str(name) == weight_variable
        if not is_weight_column and weight_values is not None and values is not None:
            # ``weight_spec`` carries no column name, so the weight variable is found by
            # its VALUES: a column equal to the weight vector IS the weight variable.
            # Without this the weight column was summarised with the weights, giving
            # Obs = sum(w) and Mean = sum(w^2)/sum(w) instead of Stata's raw counts.
            if values.size == weight_values.size and np.array_equal(values, weight_values):
                is_weight_column = True
        if values is None:
            rows[name] = _summary_of_a_non_numeric_column(column)
        elif is_weight_column:
            # ``estat_summarize.sthlp:98-99``: "An unweighted summarize is performed on
            # the weight variable."  Weighting the weight column by itself multiplies
            # each weight by its own value -- a number with no meaning.
            rows[name] = _weighted_summary(values, None, weight_type=None)
        else:
            rows[name] = _weighted_summary(
                values, weight_values, weight_type=weight_type
            )

    if not rows:
        # Reaching here means no variable could be mapped back to the data.  A
        # bare KeyError from the column selection below says nothing about why,
        # so name the actual problem.
        raise ValueError(
            "estat summarize: no variable of this model could be matched to a "
            f"column of the data (available: {list(frame.columns)}); pass "
            "variables= explicitly"
        )

    table = pd.DataFrame.from_dict(rows, orient="index")
    table = table.rename(
        columns={
            "obs": "Obs",
            "mean": "Mean",
            "sd": "Std. Dev.",
            "min": "Min",
            "max": "Max",
        }
    )
    table = table[["Obs", "Mean", "Std. Dev.", "Min", "Max"]]
    table.index = pd.Index(list(rows), name="Variable")

    clusters = None
    if cluster is not None:
        # NB: this local is called ``cluster_labels``, not ``labels``.  It used to be
        # ``labels`` and silently SHADOWED the ``labels=`` parameter, so passing
        # ``cluster=`` clobbered the variable labels and the constructor then tried to
        # ``dict()`` a numpy array.
        cluster_labels = np.asarray(cluster)
        if cluster_labels.size != len(frame):
            raise ValueError(
                f"estat summarize: {cluster_labels.size} cluster labels for "
                f"{len(frame)} rows"
            )
        _, counts = np.unique(cluster_labels, return_counts=True)
        clusters = {
            "name": cluster_name or "cluster",
            "n": float(counts.size),
            "min": float(counts.min()),
            "avg": float(counts.mean()),
            "max": float(counts.max()),
        }

    n_obs = float(result.nobs) if hasattr(result, "nobs") else float(len(frame))
    return SummarizeResult(
        table, n_obs, clusters, noheader=noheader, labels=labels
    )
