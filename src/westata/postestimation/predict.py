"""Typed prediction, targeting Stata's ``predict`` suboptions.

Each estimator family exposes the prediction types Stata offers for it:

======================  ==========================================
``xb``                  linear prediction ``x b``
``stdp``                standard error of the linear prediction
``stdf``                standard error of the forecast
``pr``                  predicted probability (binary models)
``mu`` / ``n``          predicted mean (count models)
``residuals``           ``y - fitted``
``e(a,b)``              expected value of ``y`` given censoring (tobit)
``ystar(a,b)``          expected value of the latent ``y*`` (tobit)
======================  ==========================================

The module is deliberately a thin, explicit dispatch: every supported type for
an estimator is listed in :data:`SUPPORTED_TYPES`, and requesting anything else
raises rather than guessing.  ``margins`` builds on this table, so the two
cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["SUPPORTED_TYPES", "predict", "supported_types"]

#: ``model_name -> tuple of valid predict types``.  Keys are the Stata command
#: names westata's ``StataResults.model_name`` carries.
#:
#: Every model that offers ``xb`` also offers the two prediction-standard-error
#: types ``stdp``/``stdf``: both are functions of the design matrix and the
#: parameter covariance alone, which every result carries.
SUPPORTED_TYPES: dict[str, tuple[str, ...]] = {
    "regress": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "ivregress 2sls": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "ivregress liml": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "ivregress gmm": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "reghdfe": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "xtreg, fe": ("xb", "stdp", "stdf", "residuals", "fitted", "ue"),
    "xtreg, re": ("xb", "stdp", "stdf", "residuals", "fitted", "ue"),
    "xtreg, be": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "xtreg, fd": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "logit": ("xb", "stdp", "pr", "residuals"),
    "probit": ("xb", "stdp", "pr", "residuals"),
    "poisson": ("xb", "stdp", "n", "residuals"),
    "nbreg": ("xb", "stdp", "n", "residuals"),
    "ologit": ("xb", "stdp", "pr", "residuals"),
    "oprobit": ("xb", "stdp", "pr", "residuals"),
    "tobit": ("xb", "stdp", "stdf", "e", "ystar", "residuals"),
    # --- P6 limited dependent variables ---
    # Stata: truncreg/intreg are continuous models, so they offer xb, stdp, stdf.
    "truncreg": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "intreg": ("xb", "stdp", "stdf", "fitted"),
    # heckman/etregress predict from the outcome equation.
    "heckman": ("xb", "stdp", "stdf", "fitted"),
    "etregress": ("xb", "stdp", "stdf", "fitted"),
    # --- P7 count models ---
    # Verified on StataNow 19.5: zip offers xb/n/stdp but rejects stdf with
    # r(198), the same split as poisson/nbreg.
    "zip": ("xb", "n", "stdp", "residuals"),
    "zinb": ("xb", "n", "stdp", "residuals"),
    "ztp": ("xb", "n", "stdp", "residuals"),
    "tnbreg": ("xb", "n", "stdp", "residuals"),
}


def supported_types(model_name: str) -> tuple[str, ...]:
    """Return the predict types available for ``model_name``."""

    try:
        return SUPPORTED_TYPES[model_name]
    except KeyError:
        raise UnsupportedError(
            f"predict: no prediction types registered for {model_name!r}; "
            f"known models: {sorted(SUPPORTED_TYPES)}"
        ) from None


def _predict_binary(result: Any, link: str, exog: np.ndarray) -> np.ndarray:
    """Linear prediction pushed through the model's link function."""

    xb = exog @ np.asarray(result.params, dtype=float)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-xb))
    from scipy import stats as _stats

    return _stats.norm.cdf(xb)


def _predict_tobit_e(
    result: Any, exog: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """E(y | a < y < b) -- the *conditional* mean, Stata's ``e(a,b)``.

    ``[R] tobit postestimation`` defines ``e(a,b)`` as ``E(y_j | a < y_j < b)``:
    the mean of the observed outcome *conditional on it falling inside the
    interval*.  For an observation whose linear index already lies inside
    (a, b), that conditional mean is the truncated-normal mean

        xb + sigma * (phi(z_a) - phi(z_b)) / (Phi(z_b) - Phi(z_a)).

    Note this is NOT the same as Stata's ``ystar(a,b)``, which is the
    unconditional expectation of the censored variable.
    """

    from scipy import stats as _stats

    params = np.asarray(result.params, dtype=float)
    beta = params[:-1]
    sigma = float(params[-1])
    xb = exog @ beta
    z_lower = (lower - xb) / sigma if lower is not None else None
    z_upper = (upper - xb) / sigma if upper is not None else None

    # Mass of the interval. Computed with the survival function so that the
    # upper tail keeps its precision (< 1 - 1e-16 would cancel to zero).
    if z_lower is None and z_upper is None:
        probability = np.ones_like(xb)
    elif z_lower is None:
        probability = _stats.norm.cdf(z_upper)
    elif z_upper is None:
        probability = _stats.norm.sf(z_lower)
    else:
        probability = _stats.norm.cdf(z_upper) - _stats.norm.cdf(z_lower)

    lower_pdf = 0.0 if z_lower is None else _stats.norm.pdf(z_lower)
    upper_pdf = 0.0 if z_upper is None else _stats.norm.pdf(z_upper)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = xb + sigma * (lower_pdf - upper_pdf) / probability
    fallback = lower if lower is not None else (upper if upper is not None else 0.0)
    return np.where(probability > 1e-300, value, fallback)


def _predict_tobit_ystar(
    result: Any, exog: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """E(y*) where y* = max{a, min{y, b}} -- Stata's ``ystar(a,b)``.

    The *unconditional* expectation of the censored variable, i.e. the familiar
    Tobit mean:

        E[y*] = a*Phi(z_a) + xb*(Phi(z_b) - Phi(z_a)) + b*(1 - Phi(z_b))
                + sigma*(phi(z_a) - phi(z_b))

    This is the quantity ``margins, predict(ystar(0,.))`` averages.  It matches
    Stata's ``e(a,b)`` output only when a < y < b for every observation, which is
    why the two are easy to confuse.
    """

    from scipy import stats as _stats

    params = np.asarray(result.params, dtype=float)
    beta = params[:-1]
    sigma = float(params[-1])
    xb = exog @ beta
    z_lower = (lower - xb) / sigma if lower is not None else None
    z_upper = (upper - xb) / sigma if upper is not None else None

    lower_cdf = 0.0 if z_lower is None else _stats.norm.cdf(z_lower)
    upper_cdf = 1.0 if z_upper is None else _stats.norm.cdf(z_upper)
    lower_pdf = 0.0 if z_lower is None else _stats.norm.pdf(z_lower)
    upper_pdf = 0.0 if z_upper is None else _stats.norm.pdf(z_upper)

    value = xb * (upper_cdf - lower_cdf) + sigma * (lower_pdf - upper_pdf)
    if lower is not None:
        value = value + lower * lower_cdf
    if upper is not None:
        value = value + upper * (1.0 - upper_cdf)
    return value


def _coefficient_block(result: Any, n_columns: int) -> np.ndarray:
    """Return the parameter indices that multiply the columns of ``exog``.

    For a plain model every parameter is a regressor and the block is the whole
    vector.  Maximum-likelihood estimators append *auxiliary* parameters that
    Stata reports with a leading slash and that never appear in a design
    matrix: ``/sigma`` for tobit, ``/lnalpha`` for nbreg, ``/cut1``, ``/cut2``,
    ... for the ordered models.  Those must be dropped before forming
    ``x' V x`` -- mixing a scale parameter into a coefficient would be a units
    error -- so ``stdp``/``stdf`` use only the coefficient block, together with
    the matching block of the covariance matrix.  This is exactly what Stata
    does: `predict, stdp` after `tobit` treats the coefficients as the estimand
    and ``sigma`` as a nuisance parameter.

    A mismatch between ``exog``'s width and the number of *regressors* is a
    caller error and raises rather than guessing which columns were meant.
    """

    params = np.asarray(result.params, dtype=float).reshape(-1)
    n_params = params.size
    names = [str(name) for name in getattr(result, "param_names", []) or []]

    if len(names) == n_params:
        block = np.array(
            [position for position, name in enumerate(names) if not name.startswith("/")],
            dtype=int,
        )
    else:
        block = np.arange(n_params)

    n_regressors = block.size
    if n_columns != n_regressors:
        raise ValueError(
            f"predict: exog has {n_columns} column(s) but the model has "
            f"{n_regressors} coefficient(s)"
            + (
                f" (design columns: {[names[i] for i in block]})"
                if len(names) == n_params
                else ""
            )
            + "; pass one column per regressor, i.e. the design matrix the "
            "model was fitted with (auxiliary parameters such as /sigma are not "
            "design columns)."
        )

    return block


def _residual_variance(result: Any) -> float:
    """Residual variance ``sigma^2`` used by ``stdf``.

    Follows the contract's Stata chain: ``e(rmse)`` squared when it is recorded,
    otherwise ``rss / N`` from the stored residuals.  A model that records
    neither is a genuine consistency problem -- falling back to, say,
    ``rss / (N - k)`` would silently report a different quantity than Stata --
    so it raises instead.

    Note for the non-Gaussian families: Stata's own ``stdf`` after ``logit`` /
    ``probit`` / ``poisson`` adds the *distribution-specific* innovation
    variance (``pi^2/3`` for logit, ``1`` for probit, ``mu_i`` for Poisson)
    rather than a residual variance, so those numbers differ from this
    implementation.  ``stdf`` is documented here as meaningful for models with
    an additive ``sigma^2`` -- ``regress``, ``ivregress``, ``reghdfe``,
    ``xtreg`` and ``tobit`` -- where it reproduces Stata exactly; for the
    binary/count families it reports ``x'Vx + rss/N`` on the response scale.
    """

    e = getattr(result, "e", None)
    raw_rmse = None
    if e is not None:
        getter = getattr(e, "get", None)
        raw_rmse = getter("rmse") if callable(getter) else getattr(e, "rmse", None)

    rmse = None if raw_rmse is None else float(np.asarray(raw_rmse, dtype=float))
    if rmse is not None and np.isfinite(rmse):
        return rmse**2

    resid = getattr(result, "resid", None)
    if resid is not None:
        resid_arr = np.asarray(resid, dtype=float).reshape(-1)
        nobs = getattr(result, "nobs", None)
        count = resid_arr.size
        if nobs is not None and int(nobs) != count:
            raise ConsistencyError(
                "predict: stdf needs a residual variance, but this result stores "
                f"{count} residuals against nobs={nobs}; the residuals cannot be "
                "used to recover the residual variance"
            )
        if count > 0 and np.all(np.isfinite(resid_arr)):
            rss = float(np.dot(resid_arr, resid_arr))
            if rss == 0.0:
                # An all-zero residual vector is a placeholder, not a perfect
                # fit: a genuinely perfect fit records e(rmse) == 0 and returns
                # through the branch above.  Dividing by N here would report
                # sigma2 == 0 and silently make stdf identical to stdp.
                raise ConsistencyError(
                    "predict: stdf cannot be computed for this result -- the "
                    "stored residuals are all zero (a placeholder, as for the "
                    "ordered models), so rss/N is not a residual variance. "
                    "Use kind='stdp' for the standard error of the linear "
                    "prediction."
                )
            return rss / float(count)

    raise ConsistencyError(
        "predict: stdf needs the residual variance, but this result records "
        "neither a finite e(rmse) nor usable residuals to compute rss/N from; "
        "refit with an estimator that stores e(rmse) or the residuals"
    )


def _predict_standard_error(
    result: Any, exog: np.ndarray, *, forecast: bool
) -> np.ndarray:
    """``sqrt(diag(X V X'))``, plus ``sigma^2`` for the forecast version."""

    exog = np.asarray(exog, dtype=float)
    if exog.ndim != 2:
        raise ValueError(
            f"predict: exog must be 2-dimensional (observations x regressors), "
            f"got {exog.ndim} dimension(s)"
        )

    block = _coefficient_block(result, exog.shape[1])

    cov = np.asarray(result.cov_params(), dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ConsistencyError(
            f"predict: the parameter covariance must be square, got shape {cov.shape}"
        )
    cov = cov[np.ix_(block, block)]
    if not np.all(np.isfinite(cov)):
        raise ConsistencyError(
            "predict: the parameter covariance contains non-finite entries; the "
            "model may be unidentified or the VCE failed"
        )

    # Guarantee symmetry: the quadratic form is only meaningful for a symmetric
    # matrix, and a sandwich estimator can carry 1-ulp asymmetry.
    variance = np.einsum("ij,jk,ik->i", exog, (cov + cov.T) / 2.0, exog)
    if not np.all(np.isfinite(variance)):
        raise ConsistencyError(
            "predict: x'Vx is not finite; check the design matrix for extreme "
            "values or non-finite entries"
        )

    tolerance = (
        -1e-10 * max(float(np.max(np.abs(variance))), 1.0) if variance.size else 0.0
    )
    if np.any(variance < tolerance):
        raise ConsistencyError(
            "predict: x'Vx is negative for at least one observation, so the "
            "covariance matrix is not positive semidefinite; the reported "
            "prediction standard errors would be undefined"
        )
    # Round-off can leave a genuinely zero variance slightly negative.
    variance = np.maximum(variance, 0.0)

    if forecast:
        variance = variance + _residual_variance(result)

    return np.sqrt(variance)


def predict(
    result: Any,
    exog: np.ndarray,
    kind: str = "xb",
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """Return a prediction of the requested ``kind`` for ``exog``.

    Parameters
    ----------
    exog:
        Design matrix to predict at.  Pass the estimation sample for the usual
        in-sample predictions, or new rows for out-of-sample work.
    kind:
        One of the types returned by :func:`supported_types`.  ``"stdp"`` and
        ``"stdf"`` are the prediction standard errors (``[R] predict``): the
        standard error of the linear prediction, and of the forecast, which adds
        the residual variance.
    lower, upper:
        Censoring limits, required by the ``tobit`` types ``e`` and ``ystar``.
        When omitted, ``result.e["lower"]``/``result.e["upper"]`` are used.
    """

    model_name = str(getattr(result, "model_name", ""))
    allowed = supported_types(model_name)
    normalized = str(kind).strip()
    if normalized not in allowed:
        raise UnsupportedError(
            f"predict: {model_name!r} supports {allowed}; got {kind!r}"
        )

    exog = np.asarray(exog, dtype=float)
    params = np.asarray(result.params, dtype=float)

    # Standard errors of the prediction come first: they consume the raw design
    # matrix and covariance, so they must not run through the ``xb`` branch.
    if normalized in ("stdp", "stdf"):
        return _predict_standard_error(result, exog, forecast=normalized == "stdf")

    if normalized in ("xb", "fitted"):
        return exog @ params[: exog.shape[1]]

    if normalized == "residuals":
        return np.asarray(result.resid, dtype=float).reshape(-1)

    if normalized in ("pr", "n"):
        link = "logit" if model_name in ("logit", "ologit") else "probit"
        if model_name in ("poisson", "nbreg"):
            return np.exp(exog @ params[: exog.shape[1]])
        return _predict_binary(result, link, exog)

    if model_name == "tobit" and normalized in ("e", "ystar"):
        lower_limit = (
            float(result.e.get("lower")) if lower is None and result.e.get("lower") is not None else lower
        )
        upper_limit = (
            float(result.e.get("upper")) if upper is None and result.e.get("upper") is not None else upper
        )
        if normalized == "e":
            return _predict_tobit_e(result, exog, lower_limit, upper_limit)
        return _predict_tobit_ystar(result, exog, lower_limit, upper_limit)

    raise UnsupportedError(
        f"predict: type {kind!r} is registered for {model_name!r} but has no "
        "implementation; this is a westata bug"
    )
