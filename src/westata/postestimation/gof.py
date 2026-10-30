"""Goodness-of-fit tests: Stata's ``estat gof`` (Poisson only).

Stata's ``estat gof`` exists **only** for ``poisson``: ``poisson.ado:255`` records
``estat_cmd = "poisson_estat"``, and that program forwards ``gof`` to
``poisgof.ado``.  ``nbreg`` sets no such ``estat_cmd``, so there is no
``estat gof`` after a negative binomial fit.  Implementing one for ``nbreg``
would be inventing a command Stata does not have.

The formulas are transcribed from ``poisgof.ado`` (StataNow 19.5, v1.3.3):

.. code-block:: stata

    predict double `yhat' if e(sample), n
    gen double `elem' = `wt'*(`dv'-`yhat')^2/`yhat'
    summarize `elem' if e(sample), meanonly
    scalar `chi2_p' = r(sum)                       // Pearson

    gen double `P' = lngamma(1 + `dv') if e(sample)
    replace `P' = sum(`P'*`wt') if e(sample)
    scalar `llconst' = `P'[`lastobs']               // sum lngamma(1+y)
    replace `P' = cond(`dv'==0, 0, `dv'*(ln(`dv')-1)) if e(sample)
    replace `P' = sum(`P'*`wt') if e(sample)
    scalar `llperf' = `P'[`lastobs'] - `llconst'    // sum[y ln y - y]
    scalar `chi2_d' = -2*(e(ll) - `llperf')         // deviance

    local df = e(N) - e(df_m) - 1

Note the ``cond(y == 0, 0, ...)`` guard: ``y ln y`` is taken as 0 at ``y = 0``,
which is the standard convention, and skipping it would produce ``NaN`` for
every zero count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.special import gammaln

__all__ = ["GofResult", "estat_gof"]


class GofResult:
    """Stata's ``estat gof`` output for a Poisson fit.

    Attributes
    ----------
    deviance, deviance_pvalue:
        The likelihood-ratio (deviance) goodness-of-fit statistic and its
        upper-tail chi-square p-value.
    pearson, pearson_pvalue:
        The Pearson statistic and its p-value.
    df:
        ``e(N) - e(df_m) - 1``, shared by both tests.
    """

    __slots__ = (
        "deviance",
        "deviance_pvalue",
        "pearson",
        "pearson_pvalue",
        "df",
    )

    def __init__(
        self,
        *,
        deviance: float,
        deviance_pvalue: float,
        pearson: float,
        pearson_pvalue: float,
        df: float,
    ) -> None:
        self.deviance = float(deviance)
        self.deviance_pvalue = float(deviance_pvalue)
        self.pearson = float(pearson)
        self.pearson_pvalue = float(pearson_pvalue)
        self.df = float(df)

    def __repr__(self) -> str:
        return (
            f"<GofResult df={self.df:g} deviance={self.deviance:.6g} "
            f"pearson={self.pearson:.6g}>"
        )

    def __str__(self) -> str:
        # Layout follows poisgof.ado's display statements: two blocks, each a
        # statistic line and a ``Prob > chi2(df)`` line.
        df_label = f"{self.df:g}"
        prob_label = f"Prob > chi2({df_label})"
        return "\n".join(
            [
                "",
                f"{'Deviance goodness-of-fit':<34} = {self.deviance:>9.6g}",
                f"{prob_label:<34} =    {self.deviance_pvalue:>6.4f}",
                "",
                f"{'Pearson goodness-of-fit':<34} = {self.pearson:>9.6g}",
                f"{prob_label:<34} =    {self.pearson_pvalue:>6.4f}",
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names Stata returns."""

        return {
            "chi2_d": self.deviance,
            "chi2_p": self.pearson,
            "df": self.df,
            "p_d": self.deviance_pvalue,
            "p_p": self.pearson_pvalue,
        }


class LfitResult:
    """Stata's ``estat gof`` after ``logit``/``probit`` -- ``lfit.ado``.

    This is a DIFFERENT test from the Poisson one: ``lfit.ado:44-97`` collapses
    observations by covariate pattern and forms

        chi2 = sum_j (obs_j - exp_j)^2 / (exp_j (1 - exp_j / w_j))

    with ``df = m - k`` where ``m`` is the number of distinct covariate patterns
    and ``k`` the number of regressors (``lfit.ado:96``:
    ``ret scalar df = return(m) - `nvar'``).

    Attributes
    ----------
    chi2, pvalue:
        The Pearson statistic and its upper-tail chi-square p-value.
    df:
        ``m - k``.
    patterns:
        ``m``, the number of distinct covariate patterns.
    """

    __slots__ = ("chi2", "pvalue", "df", "patterns", "depvar", "model_label", "n_obs")

    def __init__(
        self,
        *,
        chi2: float,
        pvalue: float,
        df: float,
        patterns: int,
        depvar: str = "",
        model_label: str = "logistic",
        n_obs: int = 0,
    ) -> None:
        self.chi2 = float(chi2)
        self.pvalue = float(pvalue)
        self.df = float(df)
        self.patterns = int(patterns)
        #: ``lfit.ado:91``: ``ret scalar N = `obs'`` -- the observation count the test
        #: was built on.  It was missing from ``to_dict``, so a caller reading Stata's
        #: ``r(N)`` got nothing.
        self.n_obs = int(n_obs)
        #: ``lfit.ado:116-120`` prints the model kind and the response name:
        #:
        #:   Goodness-of-fit test after logistic model
        #:   Variable: yb
        #:
        #: The label is "logistic" for logit and "probit" for probit
        #: (``lfit.ado:117-119``).
        self.depvar = str(depvar)
        self.model_label = str(model_label)

    def __str__(self) -> str:
        header = f"Goodness-of-fit test after {self.model_label} model"
        if self.depvar:
            header += f"\nVariable: {self.depvar}"
        return (
            f"{header}\n"
            f"  Pearson chi2({self.df:g}) = {self.chi2:.4f}\n"
            f"  Prob > chi2        = {self.pvalue:.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names ``lfit.ado:91-98`` returns.

        All five: ``N`` (observations), ``m`` (classes or patterns), ``chi2``, ``df``
        and ``p``.  ``N`` used to be absent, so a caller following the ado got nothing
        where Stata returns a value.
        """

        return {
            "N": float(self.n_obs),
            "m": float(self.patterns),
            "chi2": self.chi2,
            "df": self.df,
            "p": self.pvalue,
        }


def _lfit_pearson(result: Any) -> LfitResult:
    """``lfit.ado:44-97`` -- the default (no ``group()``) branch.

    Collapse by covariate pattern, then form the Pearson chi-square.  The
    denominator ``exp_j (1 - exp_j / w_j)`` uses the pattern's total weight ``w_j``:

    * unweighted, ``w_j`` is the observation count, so the term is the familiar
      ``n_j p_j (1 - p_j)``;
    * ``fweight`` -- the type ``Logit``/``Probit`` accept -- replicates each row
      ``w_i`` times, so the pattern's weight is ``sum(w_i)`` over its rows.

    ``lfit_p.ado:123-128`` builds ``w`` exactly that way ("``gen double `w' `exp'``"
    with ``w = 1`` when no weight is given).
    """

    from scipy import stats as scipy_stats

    design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat gof needs the estimation design, which this result did not "
            "retain; refit with a from_formula constructor"
        )
    matrix = np.asarray(design[0], dtype=float)
    fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
    residual = np.asarray(result.resid, dtype=float).reshape(-1)
    observed = fitted + residual

    # A frequency-weighted fit expands NOBS to sum(w) while the design keeps one
    # row per observation, so the arrays disagree in length.  The test is defined
    # per covariate pattern, so the row-level design is what matters; taking the
    # first ``n_rows`` entries would be wrong, because the expansion interleaves
    # replicated rows.  Rebuild the per-row values from the design instead.
    n_rows = matrix.shape[0]
    if observed.size != n_rows:
        raise ValueError(
            "estat gof cannot reconstruct the per-row response for this result: "
            f"the design has {n_rows} rows but the residual vector has "
            f"{observed.size}.  A frequency-weighted fit expands the arrays to "
            "sum(w) while the design stays one row per observation, so the "
            "covariate-pattern grouping is not recoverable from these pieces; "
            "refit without freq_weights for estat gof."
        )

    # Group by covariate pattern: rows of the design that are identical.
    keys: dict[tuple[float, ...], list[int]] = {}
    for index in range(n_rows):
        key = tuple(np.round(matrix[index], 10))
        keys.setdefault(key, []).append(index)

    chi2 = 0.0
    patterns = 0
    for indices in keys.values():
        block = np.asarray(indices)
        # ``lfit.ado:67-72``: observed successes and expected successes summed
        # within the pattern, with ``w`` the pattern's total WEIGHT.
        weight = float(block.size)
        successes = float(np.sum(observed[block] != 0))
        expected = float(np.sum(fitted[block]))
        # ``lfit.ado:83-92``: ``m`` counts the observations of ``touse``, which
        # only requires ``w > 0`` -- it does NOT require the expected count to lie
        # in ``(0, w)``.  A pattern whose expected count falls outside that range
        # is skipped for the STATISTIC (``lfit.ado:85`` guards the term) but still
        # contributes to ``m``, and hence to ``df``.  Counting only the
        # contributing patterns understates the degrees of freedom whenever a
        # fitted probability reaches 0 or 1.
        patterns += 1
        if expected <= 0.0 or expected >= weight:
            continue
        denominator = expected * (1.0 - expected / weight)
        if denominator <= 0.0:  # pragma: no cover - covered by the guard above
            continue
        chi2 += (successes - expected) ** 2 / denominator

    # ``lfit.ado:96``: df = m - nvar, where nvar counts the regressors AND the
    # intercept (Getivars, ``lfit.ado:480-504``).
    df = float(patterns - matrix.shape[1])
    pvalue = float(scipy_stats.chi2.sf(chi2, df)) if df > 0 else float("nan")

    # ``lfit.ado:116-120`` labels the table with the model kind and the response.
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    model_label = "probit" if "probit" in model_name else "logistic"
    response_name = str(getattr(result, "response_name", "") or "")
    if not response_name:
        e_macros = getattr(result, "e", None) or {}
        response_name = str(e_macros.get("depvar", "") or "")
    return LfitResult(
        chi2=chi2,
        pvalue=pvalue,
        df=df,
        patterns=patterns,
        depvar=response_name,
        model_label=model_label,
        n_obs=int(observed.size),
    )


def _lfit_hosmer_lemeshow(result: Any, group: int) -> "LfitResult":
    """``estat gof, group(#)`` -- the Hosmer-Lemeshow test (``lfit.ado:26-41, 199-250``).

    Observations are sorted by the fitted probability and COLLAPSED into ``group``
    quantile classes; the Pearson statistic is then formed over those classes with
    ``df = group - 2`` (``lfit.ado:96``'s ``df = m - nvar`` where ``m`` is the number of
    classes and ``nvar`` is 2 for a model with an intercept).

    Grouping follows ``lfit.ado:252-281``'s ``Quantile``: with cumulative weight ``w``
    the scaled quantity is ``q = group * w / obs``, and a new class starts where ``q``
    reaches the next integer.  Implemented on the SORTED probabilities so ties stay
    together -- splitting equal fitted probabilities across classes would make the
    collapse depend on row order, which ``lfit.ado:265``'s ``sort p`` rules out.

    Groups that end up empty (possible with heavy ties) are dropped, matching
    ``lfit.ado``'s ``m = r(N)`` count over the collapsed rows.
    """

    from scipy import stats as scipy_stats

    if not isinstance(group, (int, np.integer)) or group < 2:
        raise ValueError(
            "estat gof, group(#) needs an integer of at least 2: the Hosmer-Lemeshow "
            f"test compares at least two quantile classes; got {group!r}"
        )

    fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
    residual = np.asarray(result.resid, dtype=float).reshape(-1)
    observed = fitted + residual
    n = observed.size
    if n == 0:
        raise ValueError("estat gof needs at least one observation")

    # Sort by the fitted probability; ties therefore stay adjacent.
    order = np.argsort(fitted, kind="mergesort")
    sorted_probability = fitted[order]
    sorted_observed = observed[order]

    # ``lfit.ado:268``: q = group * (cumulative obs) / obs.  Unweighted, the cumulative
    # weight of the i-th sorted row is i + 1.
    cumulative = np.arange(1, n + 1, dtype=float)
    scaled = group * cumulative / n

    # ``lfit.ado:268``: q = group * (cumulative weight) / obs.  A row belongs to class
    # ``floor(q) + 1``, capped at ``group`` (the last row has q == group exactly).
    # This is the definition the ado's ``round(q,1)`` comparisons encode, written
    # directly so no boundary can be off by one.
    classes = np.minimum(np.floor(scaled).astype(int) + 1, group)

    chi2 = 0.0
    used = 0
    for identifier in range(1, group + 1):
        mask = classes == identifier
        count = int(mask.sum())
        if count == 0:
            # An empty class contributes nothing and does not count towards ``m``.
            continue
        weight = float(count)
        successes = float(np.sum(sorted_observed[mask] != 0))
        expected = float(np.sum(sorted_probability[mask]))
        used += 1
        if expected <= 0.0 or expected >= weight:
            continue
        denominator = expected * (1.0 - expected / weight)
        if denominator > 0.0:
            chi2 += (successes - expected) ** 2 / denominator

    # ``lfit.ado:96``: df = m - nvar, and for the HL form nvar is 2 (the intercept plus
    # the grouping), which is why Stata prints ``df = group - 2``.
    df = float(used - 2)
    pvalue = float(scipy_stats.chi2.sf(chi2, df)) if df > 0 else float("nan")

    # ``lfit.ado:116-120`` prints the SAME header as the Pearson branch, so the model
    # label and the response name have to be carried through here too.  Hardcoding
    # "logistic" made a probit fit claim to be a logistic one.
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    model_label = "probit" if "probit" in model_name else "logistic"
    response_name = str(getattr(result, "response_name", "") or "")
    if not response_name:
        e_macros = getattr(result, "e", None) or {}
        response_name = str(e_macros.get("depvar", "") or "")
    return LfitResult(
        chi2=chi2,
        pvalue=pvalue,
        df=df,
        patterns=used,
        depvar=response_name,
        model_label=model_label,
        n_obs=int(n),
    )


def _require_poisson(result: Any) -> None:
    """``estat gof`` after ``poisson`` is ``poisgof``; after ``logit`` it is
    ``lfit``.  These are different tests, so dispatch on the estimator."""

    name = str(getattr(result, "model_name", "") or "").strip().lower()
    if name != "poisson":
        raise NotImplementedError(
            "estat gof after this estimator is not implemented: Stata dispatches "
            "it through the command's own estat program -- poisson uses "
            "poisgof.ado (a deviance test) and logit uses lfit.ado (a Pearson "
            "test over covariate patterns).  This result is "
            f"{name or 'unlabelled'}."
        )


def estat_gof(
    result: Any, *, y: Any = None, mu: Any = None, group: int | None = None
) -> "GofResult | LfitResult":
    """Stata's ``estat gof`` after ``poisson``.

    Parameters
    ----------
    result:
        A fitted :class:`~westata.estimators.discrete.Poisson` result.
    y, mu:
        The response and the fitted mean.  ``mu`` defaults to
        ``result.fittedvalues``; ``y`` defaults to ``mu + result.resid`` (which
        is exact for these fits).  Pass them explicitly when the result does not
        retain a usable residual.

    Returns
    -------
    GofResult
        ``str(...)`` prints Stata's two-block table; ``.to_dict()`` returns the
        ``r()`` names.

    Raises
    ------
    NotImplementedError
        For any non-Poisson result, because Stata has no such command there.
    """

    # ``logit_estat.ado:17-19`` dispatches ``gof`` to ``lfit``, which is a Pearson
    # test over covariate patterns -- not poisgof's deviance test.  Handle it
    # before the Poisson-only guard.
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name in ("logit", "probit"):
        if group is not None:
            return _lfit_hosmer_lemeshow(result, group)
        return _lfit_pearson(result)

    _require_poisson(result)

    if group is not None:
        # ``poisgof.ado:3`` parses only ``[, Pearson]`` -- there is no ``group()``
        # option, so Stata rejects the call with a syntax error.  Silently ignoring it
        # would run the UNCOLLAPSED deviance test while the caller believed they had
        # asked for a Hosmer-Lemeshow collapse.
        raise ValueError(
            "estat gof after poisson has no group() option: poisgof.ado:3 parses only "
            f"[, Pearson], so group({group}) is not a valid request here.  The "
            "Hosmer-Lemeshow collapse belongs to lfit (logit/probit)."
        )

    if mu is None:
        mu = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    if y is None:
        resid = np.asarray(result.resid, dtype=float).reshape(-1)
        if resid.size != mu.size:
            raise ValueError(
                "estat gof: the residual vector cannot be aligned with the "
                f"fitted mean ({resid.size} vs {mu.size}); pass y= explicitly"
            )
        y = mu + resid
    y = np.asarray(y, dtype=float).reshape(-1)

    if y.size != mu.size:
        raise ValueError(
            f"estat gof: y has {y.size} entries but mu has {mu.size}"
        )
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError(
            "estat gof: the fitted mean must be finite and strictly positive; "
            "the Pearson statistic divides by it"
        )

    # -- Pearson: sum (y - mu)^2 / mu ------------------------------------
    pearson = float(np.sum((y - mu) ** 2 / mu))

    # -- Deviance: -2 (ll - ll_perfect) ----------------------------------
    # ``poisgof.ado`` builds ll_perfect in two pieces:
    #     llconst = sum lngamma(1 + y)          (i.e. sum log(y!))
    #     P       = sum[ cond(y==0, 0, y*(ln y - 1)) ]
    #     llperf  = P - llconst
    # The lngamma term is NOT optional: leaving it out turns the perfect-fit
    # log-likelihood into a large positive number and the deviance into
    # nonsense.  (Caught by comparing against a hand transcription of the ado.)
    llf = float(result.e["ll"])
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log_y_minus_y = np.where(
            y == 0.0, 0.0, y * (np.log(np.where(y == 0.0, 1.0, y)) - 1.0)
        )
    ll_const = float(np.sum(gammaln(1.0 + y)))
    ll_perfect = float(np.sum(y_log_y_minus_y)) - ll_const
    deviance = float(-2.0 * (llf - ll_perfect))

    df = float(result.e["N"] - result.e["df_m"] - 1.0)

    return GofResult(
        deviance=deviance,
        deviance_pvalue=float(stats.chi2.sf(deviance, df)) if df > 0 else float("nan"),
        pearson=pearson,
        pearson_pvalue=float(stats.chi2.sf(pearson, df)) if df > 0 else float("nan"),
        df=df,
    )
