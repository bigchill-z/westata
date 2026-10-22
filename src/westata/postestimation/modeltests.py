"""Model-comparison tests: ``lrtest`` and ``hausman``.

``[R] lrtest`` compares two nested models by their log-likelihoods:

    LR = 2 (lnL_unrestricted - lnL_restricted)  ~  chi2(k_u - k_r)

``[R] hausman`` tests whether the efficient estimator is also consistent, using
the difference between a consistent and an efficient estimate:

    H = (b_c - b_e)' [V_c - V_e]^-1 (b_c - b_e)  ~  chi2(k)

Both tests are only meaningful under assumptions westata cannot verify from the
numbers alone, so this module checks what *is* checkable (nesting, sample size,
shared parameters) and says so explicitly when a precondition fails.

Stata conformance notes
-----------------------
``lrtest`` with ``df == 0``
    Two models with the same parameter count are the degenerate case of nesting.
    ``2 (lnL_u - lnL_r)`` is then numerically 0, and ``chi2(0)`` is a point mass
    at 0, whose survival function is 1.  ``scipy.stats.chi2.sf(0, 0)`` returns
    ``nan`` instead, so the p-value is pinned to 1.0 here rather than propagated.
``hausman`` and the constant
    Stata's ``hausman`` compares the **slope** coefficients and excludes the
    constant by default (``[R] hausman`` exposes ``constant`` as an opt-in
    option precisely because it is not comparable).  The reason is not cosmetic:
    a within-transformed intercept and a GLS intercept estimate different
    population quantities, so ``V_fe - V_re`` is indefinite as soon as the
    constant enters - verified on ordinary panels in the tests.  The constant is
    therefore dropped from the comparison; ``excluded`` continues to report only
    parameters that one model does not estimate at all.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["HausmanResult", "LRTestResult", "hausman", "lrtest"]

#: Names the constant can appear under in westata's results.
_CONSTANT_NAMES = frozenset({"Intercept", "_cons", "const"})


@dataclass(frozen=True)
class LRTestResult:
    """Likelihood-ratio test between a restricted and an unrestricted model.

    Attributes
    ----------
    statistic:
        ``LR = 2 (lnL_unrestricted - lnL_restricted)``.
    df:
        ``k_unrestricted - k_restricted``; 0 for two models of equal size.
    pvalue:
        ``P > chi2(df)``; exactly 1.0 when ``df`` is 0.
    ll_unrestricted, ll_restricted:
        The log-likelihoods that produced the statistic, kept so the number can
        be audited without re-fitting either model.
    """

    statistic: float
    df: float
    pvalue: float
    ll_unrestricted: float
    ll_restricted: float

    def __str__(self) -> str:
        return (
            f"Likelihood-ratio test\n"
            f"  LR chi2({self.df:g}) = {self.statistic:.2f}\n"
            f"  Prob > chi2     = {self.pvalue:.4f}"
        )


@dataclass(frozen=True)
class HausmanResult:
    """Hausman specification test.

    Attributes
    ----------
    statistic:
        ``(b_c - b_e)' [V_c - V_e]^-1 (b_c - b_e)``.
    df:
        Number of coefficients actually compared (constants excluded).
    pvalue:
        ``P > chi2(df)``.
    excluded:
        Parameters that are not present in both models, so could not be compared.
    """

    statistic: float
    df: float
    pvalue: float
    excluded: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        text = (
            f"Hausman test\n"
            f"  chi2({self.df:g}) = {self.statistic:.2f}\n"
            f"  Prob > chi2 = {self.pvalue:.4f}"
        )
        if self.excluded:
            text += f"\n  excluded (not in both models): {self.excluded}"
        return text


def _loglike(result: Any, *, role: str) -> float:
    """Read an auditable log-likelihood, refusing to invent one."""

    value = getattr(result, "llf", None)
    if value is None or not np.isfinite(value):
        # Several estimators populate e(ll) even where the property is missing.
        e = getattr(result, "e", None)
        if e is not None:
            value = e.get("ll")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConsistencyError(
            f"lrtest: the {role} model does not report a log-likelihood, so no "
            "likelihood-ratio test is possible"
        ) from None
    if not np.isfinite(value):
        raise ConsistencyError(
            f"lrtest: the {role} model's log-likelihood is not finite; the model "
            "may not have converged"
        )
    return value


def lrtest(unrestricted: Any, restricted: Any) -> LRTestResult:
    """Likelihood-ratio test that ``restricted`` is adequate (``[R] lrtest``).

    Parameters
    ----------
    unrestricted:
        The model with the larger parameter set.
    restricted:
        The nested model with fewer parameters.  Every coefficient name in
        ``restricted`` must also appear in ``unrestricted``.

    Raises
    ------
    ConsistencyError
        The models were fit on different samples, ``restricted`` estimates a
        parameter ``unrestricted`` does not, ``unrestricted`` has *fewer*
        parameters, or a log-likelihood is unavailable.

    Warns
    -----
    RuntimeWarning
        ``unrestricted`` has a lower log-likelihood than ``restricted``, which
        cannot happen for a genuinely nested pair.  The statistic is still
        computed (and is negative) rather than silently reported as 0.
    """

    if int(unrestricted.nobs) != int(restricted.nobs):
        raise ConsistencyError(
            "lrtest: the two models were fit on different samples "
            f"({unrestricted.nobs} vs {restricted.nobs} observations); a "
            "likelihood-ratio test requires the same observations"
        )

    unrestricted_names = list(unrestricted.param_names)
    restricted_names = list(restricted.param_names)
    unrestricted_set = set(unrestricted_names)
    missing = [name for name in restricted_names if name not in unrestricted_set]
    if missing:
        raise ConsistencyError(
            "lrtest: the models are not nested - the restricted model estimates "
            f"parameter(s) the unrestricted model does not: {missing}. "
            f"unrestricted={unrestricted_names}, restricted={restricted_names}"
        )

    k_unrestricted = len(unrestricted_names)
    k_restricted = len(restricted_names)
    if k_unrestricted < k_restricted:
        raise ConsistencyError(
            "lrtest: the 'unrestricted' model has fewer parameters than the "
            f"'restricted' one ({k_unrestricted} vs {k_restricted}); swap the "
            "two arguments"
        )

    ll_unrestricted = _loglike(unrestricted, role="unrestricted")
    ll_restricted = _loglike(restricted, role="restricted")

    if ll_unrestricted < ll_restricted:
        warnings.warn(
            "lrtest: the unrestricted model has a lower log-likelihood than the "
            "restricted one, which cannot happen for a genuinely nested pair fit "
            "on the same data; the models may not be comparable. The statistic is "
            "negative, which is reported rather than clamped to zero.",
            RuntimeWarning,
            stacklevel=2,
        )

    statistic = 2.0 * (ll_unrestricted - ll_restricted)
    df = float(k_unrestricted - k_restricted)
    # chi2(0) is a point mass at 0, so its survival function is exactly 1;
    # scipy returns nan for that degenerate call, hence the explicit branch.
    pvalue = 1.0 if df == 0 else float(stats.chi2.sf(statistic, df))

    return LRTestResult(
        statistic=float(statistic),
        df=df,
        pvalue=pvalue,
        ll_unrestricted=float(ll_unrestricted),
        ll_restricted=float(ll_restricted),
    )


def _comparable_names(consistent: Any, efficient: Any) -> tuple[list[str], list[str]]:
    """Return ``(comparable, excluded)`` coefficient names.

    ``comparable`` keeps the consistent model's order and drops the constant;
    ``excluded`` lists every name that only one of the two models estimates.
    """

    consistent_names = list(consistent.param_names)
    efficient_names = list(efficient.param_names)
    efficient_set = set(efficient_names)
    consistent_set = set(consistent_names)

    comparable = [
        name
        for name in consistent_names
        if name in efficient_set and name not in _CONSTANT_NAMES
    ]
    excluded: list[str] = []
    for name in consistent_names:
        if name not in efficient_set and name not in _CONSTANT_NAMES:
            excluded.append(name)
    for name in efficient_names:
        if name not in consistent_set and name not in _CONSTANT_NAMES:
            excluded.append(name)
    return comparable, excluded


def hausman(consistent: Any, efficient: Any) -> HausmanResult:
    """Hausman test of the efficient estimator's consistency (``[R] hausman``).

    Parameters
    ----------
    consistent:
        Consistent under both hypotheses but inefficient, e.g. ``xtreg, fe``.
    efficient:
        Efficient under the null but inconsistent under the alternative, e.g.
        ``xtreg, re``.

    Raises
    ------
    ConsistencyError
        The models were fit on different samples, or share no comparable slope
        coefficient.

    Warns
    -----
    RuntimeWarning
        ``V_consistent - V_efficient`` is not positive definite.  Stata then
        inverts with a generalised inverse and still reports, so this does the
        same with ``numpy.linalg.pinv``; the statistic is clamped at 0 because a
        pseudo-inverse can produce a tiny negative quadratic form.
    """

    comparable, excluded = _comparable_names(consistent, efficient)
    if not comparable:
        raise ConsistencyError(
            "hausman: the two models share no slope coefficients, so they cannot "
            "be compared. The constant is excluded by design because a "
            f"within-transformed and a GLS intercept estimate different "
            f"parameters. consistent={list(consistent.param_names)}, "
            f"efficient={list(efficient.param_names)}"
        )

    if int(consistent.nobs) != int(efficient.nobs):
        raise ConsistencyError(
            "hausman: the two models were fit on different samples "
            f"({consistent.nobs} vs {efficient.nobs} observations)"
        )

    consistent_names = list(consistent.param_names)
    efficient_names = list(efficient.param_names)
    consistent_index = [consistent_names.index(name) for name in comparable]
    efficient_index = [efficient_names.index(name) for name in comparable]

    consistent_params = np.asarray(consistent.params, dtype=float)
    efficient_params = np.asarray(efficient.params, dtype=float)
    consistent_cov = np.asarray(consistent.cov_params(), dtype=float)
    efficient_cov = np.asarray(efficient.cov_params(), dtype=float)

    difference = consistent_params[consistent_index] - efficient_params[efficient_index]
    difference_cov = (
        consistent_cov[np.ix_(consistent_index, consistent_index)]
        - efficient_cov[np.ix_(efficient_index, efficient_index)]
    )

    # Stata inverts with a generalised inverse when the difference is not
    # positive definite, and warns. Reproduce that rather than raising: a non-PD
    # difference is common in finite samples and Stata still reports a number.
    symmetric = (difference_cov + difference_cov.T) / 2.0
    try:
        positive_definite = bool(np.all(np.linalg.eigvalsh(symmetric) > 0.0))
    except np.linalg.LinAlgError:
        positive_definite = False

    if positive_definite:
        solved = np.linalg.solve(symmetric, difference)
    else:
        warnings.warn(
            "hausman: the difference V_consistent - V_efficient is not positive "
            "definite; using a generalised inverse as Stata does, so the test "
            "statistic may be unreliable",
            RuntimeWarning,
            stacklevel=2,
        )
        solved = np.linalg.pinv(symmetric) @ difference

    statistic = float(difference @ solved)
    # A generalised inverse can produce a tiny negative quadratic form.
    statistic = max(statistic, 0.0)
    df = float(len(comparable))

    return HausmanResult(
        statistic=statistic,
        df=df,
        pvalue=float(stats.chi2.sf(statistic, df)),
        excluded=excluded,
    )
