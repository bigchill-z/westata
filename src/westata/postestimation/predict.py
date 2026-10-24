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

Model-side delegation
---------------------
A model whose predictions need parameters this module cannot interpret -- the
zero-inflated and truncated count models, whose mean depends on the inflation
equation as well as the count equation -- publishes a ``_predict_dispatch``
callable on its results object.  When that attribute exists it owns every
prediction type, including ``xb``/``stdp``: this module calls it and only falls
back to its own implementation for types the model declines.  Stata's
``predict`` after ``zip`` takes ``n``/``pr``/``xb``/``stdp`` but rejects
``stdf`` (verified on StataNow 19.5), which is precisely a per-model decision,
so it belongs to the model rather than to a hard-coded list here.
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


def model_predict_types(result: Any) -> tuple[str, ...] | None:
    """Prediction types a *model* declares for itself, if it does.

    Zero-inflated and truncated count models cannot be served by the generic
    table above: their mean is a function of both equations, and the set of
    types Stata accepts differs per command.  Such a result publishes
    ``_predict_types`` (a sequence, or a callable returning one); when present
    it is authoritative and :data:`SUPPORTED_TYPES` is not consulted.
    """

    declared = getattr(result, "_predict_types", None)
    if declared is None:
        return None
    if callable(declared):
        declared = declared()
    return tuple(str(kind) for kind in declared)


def _dispatch_prediction(
    result: Any,
    dispatch: Callable[..., Any],
    kind: str,
    exog: np.ndarray,
    *,
    lower: float | None,
    upper: float | None,
) -> np.ndarray | None:
    """Hand one prediction to the model's own ``_predict_dispatch``.

    The adapter is intentionally signature-tolerant: a model may spell the hook
    ``dispatch(kind, exog)``, ``dispatch(kind=..., exog=...)`` or accept the
    censoring limits as well.  Only parameter names the callable actually
    declares are passed, so a model does not have to accept keywords it does not
    use (and ``**kwargs`` models still receive everything).

    A ``None`` return means "the model does not implement this type after all"
    and lets the generic implementation below try, rather than forcing every
    model to duplicate the shared ``xb``/``stdp`` machinery.
    """

    import inspect

    try:
        signature = inspect.signature(dispatch)
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        signature = None

    candidate: dict[str, Any] = {
        "kind": kind,
        "exog": exog,
        "lower": lower,
        "upper": upper,
    }
    if signature is None:
        value = dispatch(kind, exog)
    else:
        parameters = signature.parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if accepts_var_kw:
            value = dispatch(**candidate)
        else:
            kwargs = {
                name: item for name, item in candidate.items() if name in parameters
            }
            if "kind" in kwargs or "exog" in kwargs:
                value = dispatch(**kwargs)
            else:
                # Positional-only spelling: ``(kind, exog)`` is the contract.
                value = dispatch(kind, exog)

    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def _predict_binary(result: Any, link: str, exog: np.ndarray) -> np.ndarray:
    """Linear prediction pushed through the model's link function."""

    xb = exog @ np.asarray(result.params, dtype=float)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-xb))
    from scipy import stats as _stats

    return _stats.norm.cdf(xb)


#: Response-scale family of a prediction type.  ``margins`` needs the first two
#: derivatives of the prediction with respect to the linear index to build the
#: delta-method standard error analytically; differencing a quantity as small as
#: an average marginal effect (3.5e-4 for ``margins, dydx(weight)`` after
#: ``logit``) loses most of its significant digits.  A family name means "the
#: prediction is a known smooth scalar function g(eta)"; ``None`` means the
#: module has no closed form and the caller must differentiate numerically.
#:
#: ``logit``:  g = 1/(1+e^-eta),  g' = p(1-p),      g'' = p(1-p)(1-2p)
#: ``probit``: g = Phi(eta),       g' = phi(eta),    g'' = -eta*phi(eta)
#: ``exp``:    g = e^eta,          g' = g'' = e^eta
#: ``identity``: g = eta,          g' = 1,           g'' = 0
#:
#: The ordered models are deliberately *absent*: their ``pr`` is
#: ``Lambda(kappa_1 - eta)``, whose derivative carries an extra minus sign and a
#: cut point, so treating it as the plain logit family would report the wrong
#: sign.  (Their prediction is not implemented either -- the ordered results do
#: not retain a design matrix -- so ``margins`` refuses them.)
_PREDICT_FAMILY: dict[str, dict[str, str]] = {
    "logit": {"pr": "logit"},
    "probit": {"pr": "probit"},
    "poisson": {"n": "exp"},
    "nbreg": {"n": "exp"},
}

_IDENTITY_KINDS = frozenset({"xb", "fitted"})


def prediction_family(result: Any, kind: str) -> str | None:
    """Name the response-scale family of ``kind`` for this model, if known.

    Returns one of ``"identity"``, ``"logit"``, ``"probit"``, ``"exp"``, or
    ``None`` when the prediction has no closed form in the linear index (the
    censored/truncated expectations ``e``/``ystar``, whose derivative depends on
    the censoring limits).
    """

    normalized = str(kind).strip()
    if normalized in _IDENTITY_KINDS:
        return "identity"
    model_name = str(getattr(result, "model_name", ""))
    return _PREDICT_FAMILY.get(model_name, {}).get(normalized)


def family_slopes(family: str, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(g'(eta), g''(eta))`` for a family named by :func:`prediction_family`."""

    from scipy import stats as _stats

    eta = np.asarray(eta, dtype=float)
    if family == "identity":
        ones = np.ones_like(eta)
        return ones, np.zeros_like(eta)
    if family == "logit":
        probability = 1.0 / (1.0 + np.exp(-eta))
        first = probability * (1.0 - probability)
        return first, first * (1.0 - 2.0 * probability)
    if family == "probit":
        density = _stats.norm.pdf(eta)
        return density, -eta * density
    if family == "exp":
        mean = np.exp(eta)
        return mean, mean
    raise ValueError(f"predict: unknown prediction family {family!r}")


#: Likelihood of the ordered models: ``ologit`` is logistic, ``oprobit`` normal.
_ORDERED_LINK: dict[str, str] = {"ologit": "logit", "oprobit": "probit"}


def ordered_structure(result: Any) -> tuple[list[int], list[int], str] | None:
    """``(slope positions, cut positions, link)`` for an ordered model.

    Detected from the *parameter names* (``/cut1``, ``/cut2``, ...), which is
    what makes an ordered model ordered rather than by a model list: any result
    that reports cut points has the ordered structure, and one that does not
    cannot be predicted this way.
    """

    names = [str(name) for name in getattr(result, "param_names", []) or []]
    cut_positions = [
        position for position, name in enumerate(names) if name.startswith("/cut")
    ]
    if not cut_positions:
        return None
    slope_positions = [position for position in range(len(names)) if position not in cut_positions]
    link = _ORDERED_LINK.get(str(getattr(result, "model_name", "")), "logit")
    return slope_positions, cut_positions, link


def _ordered_pieces(
    result: Any, exog: np.ndarray, *, outcome: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Category probability and its derivatives for an ordered model.

    Returns ``(probability, dP/deta, d2P/deta2, dP/dcut, d2P/(deta dcut))``,
    where the cut derivatives are ``K-1``-vectors broadcast against the
    observations.  ``outcome`` is Stata's ``outcome(#k)``; the default 1 is
    Stata's default for ``predict, pr`` (verified on StataNow 19.5:
    ``ologit rep78 weight mpg`` + ``predict, pr`` equals ``P(y=1)`` to 0.0e+00,
    and ``margins, dydx(weight)`` reproduces ``predict(pr outcome(1))``).
    """

    from scipy import stats as _stats

    structure = ordered_structure(result)
    if structure is None:
        return None
    slope_positions, cut_positions, link = structure
    params = np.asarray(result.params, dtype=float).reshape(-1)
    cuts = params[cut_positions]
    beta = params[slope_positions]
    eta = np.asarray(exog, dtype=float) @ beta

    n_cuts = cuts.size
    category = 1 if outcome is None else int(outcome)
    if not 1 <= category <= n_cuts + 1:
        raise UnsupportedError(
            f"predict: outcome(#{category}) is out of range for "
            f"{getattr(result, 'model_name', '')!r}, which has {n_cuts + 1} "
            f"categories (1..{n_cuts + 1})"
        )

    lower = -np.inf if category == 1 else float(cuts[category - 2])
    upper = np.inf if category == n_cuts + 1 else float(cuts[category - 1])

    def cdf(limit: float) -> np.ndarray:
        if np.isinf(limit):
            return np.ones_like(eta) if limit > 0 else np.zeros_like(eta)
        if link == "logit":
            return 1.0 / (1.0 + np.exp(-np.clip(limit - eta, -700.0, 700.0)))
        return _stats.norm.cdf(limit - eta)

    def pdf(limit: float) -> np.ndarray:
        if np.isinf(limit):
            return np.zeros_like(eta)
        shift = limit - eta
        if link == "logit":
            value = 1.0 / (1.0 + np.exp(-np.clip(shift, -700.0, 700.0)))
            return value * (1.0 - value)
        return _stats.norm.pdf(shift)

    def pdf_derivative(limit: float) -> np.ndarray:
        """``f'(limit - eta)`` with respect to the argument."""

        if np.isinf(limit):
            return np.zeros_like(eta)
        shift = limit - eta
        if link == "logit":
            value = 1.0 / (1.0 + np.exp(-np.clip(shift, -700.0, 700.0)))
            return value * (1.0 - value) * (1.0 - 2.0 * value)
        return -shift * _stats.norm.pdf(shift)

    probability = cdf(upper) - cdf(lower)
    # dP/deta = -f(upper - eta) + f(lower - eta)
    first = -pdf(upper) + pdf(lower)
    # d2P/deta2 = f'(upper - eta) - f'(lower - eta)
    second = pdf_derivative(upper) - pdf_derivative(lower)

    # dP/dc_m = f(c_m - eta) * ([m == category] - [m == category - 1])
    cut_derivative = np.zeros((n_cuts, eta.size))
    if 1 <= category <= n_cuts:
        cut_derivative[category - 1] += pdf(float(cuts[category - 1]))
    if 2 <= category <= n_cuts + 1:
        cut_derivative[category - 2] -= pdf(float(cuts[category - 2]))

    # d2P/(deta dc_m): differentiate dP/dc_m with respect to eta, i.e. flip the
    # sign of f' at each involved cut.
    mixed = np.zeros((n_cuts, eta.size))
    if 1 <= category <= n_cuts:
        mixed[category - 1] -= pdf_derivative(float(cuts[category - 1]))
    if 2 <= category <= n_cuts + 1:
        mixed[category - 2] += pdf_derivative(float(cuts[category - 2]))

    return probability, first, second, cut_derivative, mixed


def ordered_probability(
    result: Any, exog: np.ndarray, *, outcome: int | None = None
) -> np.ndarray:
    """``P(y = outcome)`` for an ordered model (Stata ``predict, pr``)."""

    pieces = _ordered_pieces(result, exog, outcome=outcome)
    if pieces is None:
        raise UnsupportedError(
            "predict: this result has no cut points, so it is not an ordered "
            "model and 'pr' has no ordered definition"
        )
    return pieces[0]


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
    vector.  Two kinds of parameters are *not* design columns and must be
    dropped before forming ``x' V x``:

    * **auxiliary** parameters, which Stata names with a leading slash:
      ``/sigma`` for tobit, ``/lnalpha`` for nbreg, ``/cut1``, ``/cut2``, ... for
      the ordered models.  Mixing a scale parameter into a coefficient would be
      a units error.  `predict, stdp` after `tobit` treats the coefficients as
      the estimand and ``sigma`` as a nuisance parameter.
    * the parameters of a **later equation** in a multi-equation model
      (``zip``/``zinb``: the inflation equation).  The design matrix describes
      the first (count) equation, and Stata's ``predict, stdp`` after ``zip`` is
      exactly ``sqrt(x_c' V_cc x_c)`` over the count block -- verified on
      StataNow 19.5 to 2.0e-16 against ``e(V)``, versus 9.6e-1 if the inflation
      block is folded in.

    Anything else -- an ``exog`` that neither matches the whole design nor is
    shorter than it -- is a caller error, and raises rather than guessing which
    columns were meant.
    """

    params = np.asarray(result.params, dtype=float).reshape(-1)
    n_params = params.size
    names = [str(name) for name in getattr(result, "param_names", []) or []]

    if len(names) == n_params:
        design_block = [
            position
            for position, name in enumerate(names)
            if not name.startswith("/")
        ]
    else:
        design_block = list(range(n_params))

    if n_columns == len(design_block):
        return np.asarray(design_block, dtype=int)

    # A *shorter* design is only meaningful for a multi-equation model, whose
    # parameters carry ``<equation>:<term>`` names or repeat a plain name once
    # per equation (``zip`` reports ``Intercept`` and ``inflate:Intercept``).
    # For a single-equation model a short design is a caller error.
    multi_equation = any(":" in name for name in names) or (
        len(set(names)) != len(names)
    )
    if multi_equation and 0 < n_columns < len(design_block):
        return np.asarray(design_block[:n_columns], dtype=int)

    raise ValueError(
        f"predict: exog has {n_columns} column(s) but the model has "
        f"{len(design_block)} coefficient(s)"
        + (
            f" (design columns: {[names[i] for i in design_block]})"
            if len(names) == n_params
            else ""
        )
        + "; pass one column per regressor, i.e. the design matrix the "
        "model was fitted with (auxiliary parameters such as /sigma are not "
        "design columns)."
    )


def _residual_variance(result: Any) -> float:
    """Innovation variance ``sigma^2`` used by ``stdf``.

    Stata's ``stdf^2 = stdp^2 + sigma^2`` where ``sigma`` is the model's own
    innovation scale, *not* the residual mean square of the observed response.
    That distinction matters for the limited-dependent-variable models: after
    ``tobit mpg weight foreign, ll(17)`` Stata reports ``stdf^2 - stdp^2 =
    e(var_e) = 14.297844`` for every observation, while ``rss/N`` of the
    censored response is 12.127405 (relative gap 1.5e-1).  ``rss/N`` is only the
    right answer for models whose response *is* the additive-error response
    (``regress``, ``ivregress``, ``reghdfe``, ``xtreg``), where ``e(rmse)^2`` and
    ``rss/N`` agree up to the ``N``-versus-``N-k`` convention Stata already
    records in ``e(rmse)``.

    Resolution order, most authoritative first:

    1. a model-supplied hook ``_innovation_variance()`` on the result;
    2. a recorded ``e(var_e)`` -- Stata's own name for ``sigma^2``
       (``tobit``/``truncreg``/``intreg`` record it, and Stata keeps only
       ``e(sigma)`` for the truncated/interval models);
    3. ``e(rmse)^2`` -- the linear families' residual scale;
    4. a ``/sigma`` coefficient in the parameter vector, squared;
    5. ``rss / N`` from the stored residuals;
    6. otherwise raise: falling back to ``rss / (N - k)`` would silently report
       a different quantity than Stata.

    ``e(sigma)`` is deliberately *not* consulted: westata records it as an alias
    of ``e(rmse)`` for the linear models, so reading it would make the
    documented ``rss/N`` fallback unreachable (the two differ by ``N/(N-k)``).

    Verified on StataNow 19.5: ``truncreg y x1 x2, ll(1)`` has
    ``stdf^2 - stdp^2 = 0.23222929 = e(sigma)^2`` and ``intreg`` likewise
    (``0.2270283``), so step 2 reproduces Stata exactly for all three limited
    families.
    """

    hook = getattr(result, "_innovation_variance", None)
    if callable(hook):
        value = hook()
        if value is not None:
            value = float(value)
            if np.isfinite(value) and value >= 0.0:
                return value

    e = getattr(result, "e", None)

    def _scalar(key: str) -> float | None:
        if e is None:
            return None
        getter = getattr(e, "get", None)
        raw = getter(key) if callable(getter) else getattr(e, key, None)
        if raw is None:
            return None
        try:
            array = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size != 1:
            return None
        value = float(array[0])
        return value if np.isfinite(value) else None

    recorded = _scalar("var_e")
    if recorded is not None:
        return recorded

    recorded = _scalar("rmse")
    if recorded is not None:
        return recorded**2

    # A scale parameter reported as ``/sigma`` (tobit, truncreg after the delta
    # transformation) is the innovation scale itself, so sigma^2 is its square.
    # Reading the *name* keeps this independent of any estimator family list.
    names = [str(name) for name in getattr(result, "param_names", []) or []]
    if names and names[-1] in ("/sigma", "sigma"):
        params = np.asarray(result.params, dtype=float).reshape(-1)
        if params.size == len(names):
            scale = float(params[-1])
            if np.isfinite(scale) and scale > 0.0:
                return scale**2

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
    outcome: int | None = None,
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
    outcome:
        Ordered models only: which category's probability ``pr`` returns
        (Stata's ``predict, pr outcome(#k)``).  ``None`` means Stata's default,
        category 1.
    """

    model_name = str(getattr(result, "model_name", ""))
    normalized = str(kind).strip()

    # A model that owns its prediction semantics takes precedence, and it also
    # defines which types exist: ``zip``/``zinb``/``ztp`` depend on the inflation
    # equation, so no generic table can enumerate their answers.
    dispatch = getattr(result, "_predict_dispatch", None)
    declared = model_predict_types(result)

    exog = np.asarray(exog, dtype=float)

    if dispatch is not None and callable(dispatch):
        if declared is not None and normalized not in declared:
            raise UnsupportedError(
                f"predict: {model_name!r} supports {declared}; got {kind!r}"
            )
        delegated = _dispatch_prediction(
            result,
            dispatch,
            normalized,
            exog,
            lower=lower,
            upper=upper,
        )
        if delegated is not None:
            return delegated

    if declared is not None:
        if normalized not in declared:
            raise UnsupportedError(
                f"predict: {model_name!r} supports {declared}; got {kind!r}"
            )
        allowed = declared
    else:
        allowed = supported_types(model_name)
        if normalized not in allowed:
            raise UnsupportedError(
                f"predict: {model_name!r} supports {allowed}; got {kind!r}"
            )

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
        # Ordered models: ``pr`` is the probability of one *category*, which is a
        # cut-point difference rather than a link function of the linear index.
        # Stata's default category is 1 (``predict, pr`` after ``ologit rep78``
        # equals P(y=1) to 0.0e+00, and ``margins, dydx(weight)`` reproduces
        # ``predict(pr outcome(1))``), so ``outcome`` follows that.
        if normalized == "pr" and ordered_structure(result) is not None:
            return ordered_probability(result, exog, outcome=outcome)

        # A response-scale prediction combines *all* structural parameters.  When
        # the model carries parameters beyond its design columns (an inflation
        # equation, a dispersion, ...) the generic formula would silently use the
        # wrong subset -- this used to raise a bare ``matmul`` dimension error
        # for ``zip``.  Refuse explicitly and name the types that do work; the
        # model itself can still answer through ``_predict_dispatch``.
        if params.size != exog.shape[1]:
            raise UnsupportedError(
                f"predict: {kind!r} needs the model's own response-scale "
                f"definition, but {model_name!r} has {params.size} parameters "
                f"against {exog.shape[1]} design column(s) (extra parameters such "
                "as an inflation equation or a dispersion are not design "
                "columns), so the generic link cannot combine them. Available "
                f"here: {allowed}. The model may publish a '_predict_dispatch' "
                "callback to answer this type."
            )
        # ``n`` is a count mean, whose link is the exponential for every count
        # family (poisson, nbreg, ztp/tpoisson, tnbreg).  It used to be selected
        # by model name, which silently pushed ``ztp`` down the binary-probit
        # branch and returned a probability instead of a mean --
        # ``ztp y x1 x2``: Stata's ``predict, n`` is exactly ``exp(xb)``
        # (verified on StataNow 19.5 to 4.3e-16).
        if normalized == "n":
            return np.exp(exog @ params[: exog.shape[1]])
        link = "logit" if model_name in ("logit", "ologit") else "probit"
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
