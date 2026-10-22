"""Typed prediction, targeting Stata's ``predict`` suboptions.

Each estimator family exposes the prediction types Stata offers for it:

======================  ==========================================
``xb``                  linear prediction ``x b``
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

from westata.errors import UnsupportedError

__all__ = ["SUPPORTED_TYPES", "predict", "supported_types"]

#: ``model_name -> tuple of valid predict types``.  Keys are the Stata command
#: names westata's ``StataResults.model_name`` carries.
SUPPORTED_TYPES: dict[str, tuple[str, ...]] = {
    "regress": ("xb", "residuals", "fitted"),
    "ivregress 2sls": ("xb", "residuals", "fitted"),
    "ivregress liml": ("xb", "residuals", "fitted"),
    "ivregress gmm": ("xb", "residuals", "fitted"),
    "reghdfe": ("xb", "residuals", "fitted"),
    "xtreg, fe": ("xb", "residuals", "fitted", "ue"),
    "xtreg, re": ("xb", "residuals", "fitted", "ue"),
    "xtreg, be": ("xb", "residuals", "fitted"),
    "xtreg, fd": ("xb", "residuals", "fitted"),
    "logit": ("xb", "pr", "residuals"),
    "probit": ("xb", "pr", "residuals"),
    "poisson": ("xb", "n", "residuals"),
    "nbreg": ("xb", "n", "residuals"),
    "ologit": ("xb", "pr", "residuals"),
    "oprobit": ("xb", "pr", "residuals"),
    "tobit": ("xb", "e", "ystar", "residuals"),
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
    """Expected value of the censored outcome (Stata ``predict, e(a,b)``).

    Following ``[R] tobit``:

        E[y|x] = a*Phi((a-xb)/s) + xb*(Phi((b-xb)/s) - Phi((a-xb)/s))
                 + b*(1 - Phi((b-xb)/s))
                 + s*(phi((a-xb)/s) - phi((b-xb)/s))
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


def _predict_tobit_ystar(
    result: Any, exog: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """Expected value of the latent outcome (Stata ``predict, ystar(a,b)``)."""

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

    return xb * (upper_cdf - lower_cdf) + sigma * (lower_pdf - upper_pdf)


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
        One of the types returned by :func:`supported_types`.
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
