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
    # ``newey_p.ado:20``: ``local myopts XB Index STDP REsiduals``.  Note how much
    # narrower this is than ``regress``: ``newey`` is a NON-``vce(ols)`` estimator,
    # and ``regress_postestimation.sthlp:152-153`` states the influence
    # diagnostics (``hat``/``rstudent``/``cooksd``/...) are unavailable with any
    # other VCE.  ``fitted`` is absent too -- ``newey_p`` does not list it.
    # ``newey_p.ado:20``: XB Index STDP Residuals.  ``index`` is a SYNONYM for
    # ``xb`` (``newey_p.ado:54`` sends both to one branch), handled by
    # ``NeweyWestResult.predict`` which maps it before dispatch -- so listing it
    # here would make ``supported_types("newey")`` claim a kind the base
    # dispatcher cannot serve, and the audit caught exactly that.
    "newey": ("xb", "stdp", "residuals"),
    # ``qreg_postestimation.sthlp:86-87``: "predict dtype newvar ... [, xb | stdp |
    # residuals]" for qreg/iqreg/bsqreg.  None of the quantile estimators was
    # registered, so EVERY type raised "no prediction types registered for 'qreg'"
    # -- including ``xb``, which the fitted design matrix could serve directly.
    "qreg": ("xb", "stdp", "residuals"),
    "iqreg": ("xb", "stdp", "residuals"),
    "bsqreg": ("xb", "stdp", "residuals"),
    # ``qreg_postestimation.sthlp:89-95`` gives ``sqreg`` a WIDER set: it adds
    # ``stddp`` (standard error of the DIFFERENCE between two linear predictions)
    # and takes an ``equation()`` option, because it fits several quantiles at once.
    "sqreg": ("xb", "stdp", "stddp", "residuals"),
    # ``hat``/``rstandard``/``rstudent``/``cooksd``/``dfbeta``/``stdr``/``covratio``
    # are Stata's influence diagnostics.  ``regress_postestimation.sthlp:152-153``
    # states they are unavailable with any ``vce()`` other than ``vce(ols)``, so
    # they live with the plain linear estimator rather than with every model that
    # can produce a residual.
    "regress": (
        "xb",
        "stdp",
        "stdf",
        "residuals",
        # ``regress_postestimation.sthlp:193``: "score is equivalent to residuals in
        # linear regression."  It is the equation-level score that ``suest`` needs, and
        # in a linear model it coincides with the residual.
        "score",
        "fitted",
        "hat",
        "leverage",
        "rstandard",
        "rstudent",
        "cooksd",
        "dfbeta",
        "stdr",
        "covratio",
        "dfits",
        "welsch",
    ),
    "ivregress 2sls": ("xb", "stdp", "stdf", "residuals", "fitted"),
    # ``ivregress`` reaches predict now that the IV result retains its design
    # (round 73); before that the whole column refused every type.
    "ivregress liml": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "ivregress gmm": ("xb", "stdp", "stdf", "residuals", "fitted"),
    # ``xtreg``: ``xtreg_postestimation.sthlp:128-133`` documents xb (the default,
    # WITHOUT the panel effect), stdp, xbu (a + xb + u_i) and xbd.  Only the ones
    # that are actually computed are listed -- a registered-but-unimplemented type
    # is worse than an unlisted one, because it advertises a feature that then
    # raises.  ``ue`` used to be listed here and always raised UnsupportedError.
    "xtreg, fe": ("xb", "stdp", "stdf", "residuals", "fitted", "u", "xbu", "ue"),
    "xtreg, re": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "xtreg, be": ("xb", "stdp", "stdf", "residuals", "fitted"),
    "xtreg, fd": ("xb", "stdp", "stdf", "residuals", "fitted"),
    # ``logit_postestimation.sthlp:209`` / ``probit_postestimation.sthlp:149``:
    # "``score`` calculates the equation-level score, the first derivative of the log
    # likelihood ..." -- SINGULAR, one variable per observation.
    #
    # Round 215 misread ``logit_p.ado:4``'s ``[, SCores *]`` as the plural spelling and
    # registered ``scores`` instead.  ``SCores`` there is Stata's PARSER abbreviation,
    # which matches the documented option ``score``; and ``predict.sthlp:192-195``
    # reserves plural ``scores`` for MULTI-EQUATION models -- "an equation-level score
    # variable is created for each equation".  A binary fit has one equation.
    "logit": ("xb", "stdp", "pr", "residuals", "score"),
    "probit": ("xb", "stdp", "pr", "residuals", "score"),
    # ``poisson_postestimation.sthlp:105, 147-148``:
    #   pr(n)     probability Pr(y = n), n a nonnegative integer (number or variable)
    #   pr(a,b)   probability Pr(a < y < b)
    # ``pr`` therefore needs an argument; the plain form is not "unavailable", it is
    # incomplete, and the dispatcher reports that distinction.
    "poisson": ("xb", "stdp", "n", "pr", "ir", "score", "residuals"),
    # ``score``/``scores`` are NOT registered for nbreg.  ``nbreg_postestimation.sthlp:172-181``
    # spells it ``scores`` and it produces TWO variables -- the derivative with respect
    # to the regression equation AND with respect to the dispersion equation -- so it
    # cannot be one array from this API.  Listing it would make the registry claim a
    # type that raises, which is the defect the registry test exists to catch.
    "nbreg": ("xb", "stdp", "n", "pr", "ir", "residuals"),
    "ologit": ("xb", "stdp", "pr", "residuals"),
    "oprobit": ("xb", "stdp", "pr", "residuals"),
    # ``tobit_p.ado:5`` does accept ``SCores`` and ``:295-330`` gives a closed form, but
    # the two equation scores there are PER-OBSERVATION quantities with respect to
    # (beta-linear-index, sigma); converting them into a stacked (n, k+1) array requires
    # multiplying the regression score by the regressors AND accounting for the fact that
    # the sigma equation's score is not a design column.  An attempt at that produced
    # column sums that were NOT zero at the MLE -- the identity `sum_i d ll_i / d b_j = 0`
    # that every correctly built score satisfies -- so the type is left unregistered
    # rather than shipped with a formula I could not verify.
    # ``tobit_postestimation.sthlp:88-95`` lists five main statistics:
    #   xb  stdp  stdf  pr(a,b)  e(a,b)  ystar(a,b)
    # ``pr`` was the one missing.  It is ``Pr(a < y < b)`` on the CENSORED outcome, and
    # ``tobit.ado:145-148`` blesses it for ``margins`` via ``Pr(passthru)`` -- which is
    # why ``margins(predict_kind="pr")`` failed here while the macro said it was allowed.
    "tobit": ("xb", "stdp", "stdf", "e", "ystar", "pr", "residuals"),
    # --- P6 limited dependent variables ---
    # Stata: truncreg/intreg are continuous models, so they offer xb, stdp, stdf.
    "truncreg": ("xb", "stdp", "stdf", "residuals", "fitted"),
    # ``intreg.ado:421-425`` blesses the same interval statistics as tobit:
    #   ereturn local marginsok default XB Pr(passthru) E(passthru) YStar(passthru)
    # and ``intreg_postestimation.sthlp:133-145`` defines them identically --
    # pr(a,b) = Pr(a < y < b), e(a,b) = E(xb+u | a < xb+u < b), ystar(a,b) = E(y*).
    # Only xb/stdp/stdf/fitted were offered here, so the macro named three capabilities
    # that did not exist.
    #
    # The difference that matters: intreg stores ``/lnsigma`` while tobit stores
    # ``sigma``; ``_scale_sigma`` resolves that by parameter name.
    "intreg": ("xb", "stdp", "stdf", "e", "ystar", "pr", "fitted"),
    # heckman/etregress predict from the outcome equation.
    # ``heckman.ado:491``: ``est local marginsok default XB PSel YCond YExpected Mills
    # NShazard`` -- five selection statistics beyond the linear prediction, none of which
    # were registered here.
    "heckman": (
        "xb",
        "stdp",
        "stdf",
        "fitted",
        "psel",
        "ycond",
        "yexpected",
        "mills",
        "nshazard",
    ),
    # ``etregress.ado:423``: ``est local marginsok "XB CTE YCTrt YCNTrt PTrt XBTrt
    # default"``.  ``cte`` is the conditional treatment effect -- the difference of the
    # two potential-outcome means -- so it is derived, not independent.
    "etregress": (
        "xb",
        "stdp",
        "stdf",
        "fitted",
        "cte",
        "yctrt",
        "ycntrt",
        "ptrt",
        "xbtrt",
    ),
    # --- P7 count models ---
    # Verified on StataNow 19.5: zip offers xb/n/stdp but rejects stdf with
    # r(198), the same split as poisson/nbreg.
    # ``zip_postestimation.sthlp:86-93`` lists n / ir / pr / pr(n) / pr(a,b) /
    # xb / stdp -- and NO residuals.  An earlier table added ``residuals`` here,
    # which the model's own ``_predict_types`` hook then refused, so the type was
    # advertised and unusable.  The hook is authoritative: it is what actually
    # computes the answer.
    "zip": ("xb", "n", "stdp"),
    "zinb": ("xb", "n", "stdp"),
    # ``tnbreg_p.ado:20``: ``local myopts "N IR CM Pr(string) CPr(string)"`` plus
    # the ``_pred_se`` set (xb / stdp).  No residuals for a truncated count.
    "ztp": ("xb", "n", "stdp"),
    "tnbreg": ("xb", "n", "stdp"),
}


#: ``model_name -> the predict type Stata assumes when no option is given``.
#:
#: Stata prints "(option pr assumed; Pr(survive))" after a bare ``predict`` on a logit and
#: "(option n assumed; predicted number of events)" on a poisson; only linear models assume
#: ``xb``. This table is the authority for that default, and it is THE ONLY copy --
#: ``StataResults.predict``, :func:`predict` and ``margins`` all read it, so the three entry
#: points cannot drift apart again.
#:
#: They did drift: ``StataResults.predict`` hardcoded ``kind="xb"`` while ``margins`` used
#: this table, so ``res.predict()`` on a logit returned log-odds by default -- neither
#: Stata's meaning nor statsmodels' (``sm.Logit.predict`` returns probabilities). Feeding
#: log-odds into a classifier silently produced a Brier score ~15x too large on the titanic
#: fweight fit (gap report 0.0.6 §3.5).
DEFAULT_PREDICT: dict[str, str] = {
    "regress": "xb",
    "ivregress 2sls": "xb",
    "ivregress liml": "xb",
    "ivregress gmm": "xb",
    # NOTE: no "reghdfe" entry. It cannot support predict at all: with absorbed fixed
    # effects the stored design matrix still carries an ``Intercept`` column while the
    # parameter vector does not (the FE absorb it), so ``X @ params`` does not even
    # dimension-check -- measured on ``y ~ x | g``, where the design has 2 columns and
    # ``params`` has 1. Listing it here would advertise a default that cannot run.
    "xtreg, fe": "xb",
    "xtreg, re": "xb",
    "xtreg, be": "xb",
    "xtreg, fd": "xb",
    "logit": "pr",
    "probit": "pr",
    "poisson": "n",
    "nbreg": "n",
    "ologit": "pr",
    "oprobit": "pr",
    "tobit": "xb",
}


def default_predict_kind(model_name: str) -> str:
    """The predict type Stata assumes for ``model_name`` when no option is given.

    Falls back to ``"xb"``, which is also what ``supported_types`` reports first for every
    model, so an unlisted estimator keeps the old behaviour.
    """

    return DEFAULT_PREDICT.get(str(model_name), "xb")


#: Panel fixed-effects components.  ``xtreg_postestimation.sthlp:127-134``:
#:
#: ======  ==========================================
#: ``xb``  ``a + xb`` (default; WITHOUT ``u_i``)
#: ``stdp``standard error of the fitted value
#: ``ue``  ``u_i + e_it``, the combined residual
#: ``xbu`` ``a + xb + u_i``, including the panel effect
#: ``u``   ``u_i``, the fixed error component
#: ``e``   ``e_it``, the overall error component
#: ======  ==========================================
#:
#: ``xbd``/``d`` (absorbed-variable effects) apply to ``areg``-style fits where a
#: separate absorbed term exists; they are not offered here.
_PANEL_FE_KINDS = frozenset({"u", "xbu", "ue"})

#: Panel model names whose predict offers the component types above.
_PANEL_FE_MODELS = frozenset({"xtreg, fe"})


def _predict_panel_components(result: Any, exog: Any, kind: str) -> Any:
    """The ``u_i``/``xbu``/``ue`` family for a fixed-effects panel fit.

    ``xtreg_postestimation.sthlp:127-134`` (FE statistics):

    ======  ============================================
    ``xb``  ``a + xb`` -- the default, WITHOUT ``u_i``
    ``ue``  ``u_i + e_it``, the combined residual
    ``xbu`` ``a + xb + u_i``, including the panel effect
    ``u``   ``u_i``, the fixed error component
    ``e``   ``e_it``, the overall error component
    ======  ============================================

    The three pieces come straight out of what the fit already holds:
    ``result.resid`` is the WITHIN residual ``e_it`` (its per-entity mean is zero
    by construction), and ``result.fittedvalues`` is ``y - e_it`` -- i.e. it
    ALREADY includes ``u_i``.  So::

        u_i  = mean_by_entity(fittedvalues) - mean_by_entity(a + xb)
        e_it = resid
        xb   = fittedvalues - u_i
    """

    model_name = str(getattr(result, "model_name", "") or "")
    if model_name not in _PANEL_FE_MODELS:
        raise UnsupportedError(
            f"predict, {kind}: the panel error-component types (u/xbu/ue) are "
            f"defined for fixed-effects panel fits (xtreg, fe); this result is "
            f"{model_name or 'unlabelled'}"
        )

    entity = getattr(result, "panel_entity", None)
    if entity is None:
        raise UnsupportedError(
            f"predict, {kind}: this panel result does not carry its entity "
            "labels, so u_i cannot be separated from e_it"
        )
    entity = np.asarray(entity).reshape(-1)

    residuals = np.asarray(result.resid, dtype=float).reshape(-1)

    def _entity_means(values: np.ndarray) -> np.ndarray:
        means = np.empty_like(values)
        for level in np.unique(entity):
            mask = entity == level
            means[mask] = values[mask].mean()
        return means

    # ``fittedvalues`` is y - e_it, which still carries u_i; the linear predictor
    # is exog @ b.  Their per-entity means differ by exactly u_i.
    full_fit = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
    linear = np.asarray(exog, dtype=float) @ np.asarray(
        result.params, dtype=float
    )[: np.asarray(exog).shape[1]]
    effect = _entity_means(full_fit) - _entity_means(linear)

    if kind == "u":
        return effect
    if kind == "xbu":
        return linear + effect
    # ``ue``: u_i + e_it, the combined residual.
    return effect + residuals


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


def _heckman_selection_design(result: Any) -> np.ndarray | None:
    """Rebuild the selection equation's design matrix from ``e(select_varlist)``.

    ``heckman``'s ``design_matrix`` is the OUTCOME equation's.  The selection statistics
    need the probit design instead, and nothing else on the result records it, so the
    estimator writes the regressor names into ``e(select_varlist)`` and they are resolved
    against ``model_data`` here.
    """

    varlist = str(result.e.get("select_varlist", "") or "").split()
    frame = getattr(result, "model_data", None)
    if not varlist or frame is None:
        return None
    columns: list[np.ndarray] = []
    for name in varlist:
        if name in ("Intercept", "_cons", "const"):
            # ``_selection_names`` includes the constant, which has no column in
            # ``model_data`` -- it is the all-ones column of the probit design.
            columns.append(
                np.ones(
                    len(frame),
                    dtype=float,
                )
            )
            continue
        try:
            columns.append(np.asarray(frame[name], dtype=float).reshape(-1))
        except (KeyError, TypeError):
            return None
    design = np.column_stack(columns)
    # The selection equation is fitted on the FULL sample, so the design has one row per
    # observation in ``model_data`` -- NOT ``result.nobs``, which counts only the selected
    # rows (256 of 400 in the reference fit).  Requiring equality with ``nobs`` rejected
    # every valid reconstruction.
    if design.shape[0] != len(frame):
        return None
    return design


def _predict_etregress(result: Any, exog: np.ndarray, kind: str) -> np.ndarray:
    """``etregress``'s treatment-specific predictions.

    ``etregress.ado:423`` declares::

        est local marginsok "XB CTE YCTrt YCNTrt PTrt XBTrt default"

    and ``etregress_postestimation.sthlp:136-167`` defines each:

    ``ptrt``    ``Pr(treatment=1) = Pr(w gamma + u > 0)``
    ``xbtrt``   the treatment equation's linear prediction, ``w gamma``
    ``yctrt``   ``E(y | treatment=1)``
    ``ycntrt``  ``E(y | treatment=0)``
    ``cte``     the conditional treatment effect, i.e. ``yctrt - ycntrt``

    ``cte`` is therefore NOT an independent formula -- it is the difference of the two
    potential-outcome means, which is what makes it checkable: computing it separately
    and comparing is the test.

    The outcome equation carries the treatment as a regressor, so ``y:t`` is the shift a
    treated observation receives.  ``yctrt`` is the outcome evaluated with that regressor
    set to 1 and the selection correction added; ``ycntrt`` with it set to 0.
    """

    from scipy import stats as _normal

    names = [str(name) for name in result.param_names]
    treat_positions = [
        index
        for index, name in enumerate(names)
        if name.startswith("treat:")
    ]
    outcome_positions = [
        index
        for index, name in enumerate(names)
        if name.startswith("y:")
    ]
    lambda_position = next(
        (index for index, name in enumerate(names) if name.lower() == "lambda"), None
    )
    if not treat_positions or not outcome_positions or lambda_position is None:
        raise UnsupportedError(
            "predict: this etregress result does not carry the usual "
            "``y:``/``lambda``/``treat:`` parameter layout"
        )

    params = np.asarray(result.params, dtype=float)
    outcome_names = [names[index] for index in outcome_positions]
    # The treatment INDICATOR is appended as the LAST column of the outcome design by
    # ``Etregress.from_formula`` ("names = list(outcome_x[1]) + [treatment_lhs]"), so it is
    # identifiable by position rather than by guessing plausible spellings -- a list of
    # ("t", "treat", "treatment") failed on a column called ``yb``.
    treatment_columns = [
        name
        for name in outcome_names
        if name.split(":", 1)[-1] not in ("Intercept", "_cons", "const")
    ]
    treatment_name = treatment_columns[-1] if treatment_columns else None
    if treatment_name is None and kind not in ("ptrt", "xbtrt"):
        raise UnsupportedError(
            f"predict, {kind} after etregress needs the TREATMENT variable as a "
            "regressor in the outcome equation; this result has "
            f"{outcome_names} and no treatment term"
        )

    treatment_design = _etregress_treatment_design(result)
    if treatment_design is None:
        raise UnsupportedError(
            f"predict, {kind} after etregress needs the TREATMENT equation's design "
            "matrix, which this result did not retain"
        )

    gamma = params[treat_positions]
    linear_treatment = treatment_design @ gamma
    probability = _normal.norm.cdf(linear_treatment)
    density = _normal.norm.pdf(linear_treatment)
    lambda_coefficient = float(params[lambda_position])

    if kind == "ptrt":
        return probability
    if kind == "xbtrt":
        return linear_treatment

    design = np.asarray(exog, dtype=float)
    beta = params[outcome_positions]
    position_in_outcome = outcome_names.index(treatment_name)
    # The outcome equation as fitted includes the treatment term; the two potential
    # outcomes differ only in that coefficient's contribution.
    base = design @ beta
    treatment_coefficient = float(beta[position_in_outcome])
    # ``base`` already contains ``b_t * t`` for the OBSERVED t; subtract it so each
    # potential outcome can set its own level.
    observed_treatment = _etregress_observed_treatment(
        result, design.shape[0], treatment_name
    )
    if observed_treatment is None:
        raise UnsupportedError(
            f"predict, {kind} after etregress needs the observed treatment indicator"
        )
    without = base - treatment_coefficient * observed_treatment
    treated = without + treatment_coefficient

    # Selection correction: the inverse Mills ratio enters with the lambda coefficient.
    aligned_probability, aligned_density, aligned_hazard = _align_selection_terms(
        result, treatment_design.shape[0], design.shape[0], probability, density
    )

    yctrt = treated + lambda_coefficient * aligned_hazard
    ycntrt = without - lambda_coefficient * (
        aligned_density / np.maximum(1.0 - aligned_probability, 1e-300)
    )
    if kind == "yctrt":
        return yctrt
    if kind == "ycntrt":
        return ycntrt
    if kind == "cte":
        return yctrt - ycntrt
    raise UnsupportedError(f"predict: {kind!r} is not an etregress treatment statistic")


def _etregress_treatment_design(result: Any) -> np.ndarray | None:
    """Rebuild the treatment equation's design from ``e(treat_varlist)``."""

    varlist = str(result.e.get("treat_varlist", "") or "").split()
    frame = getattr(result, "model_data", None)
    if not varlist or frame is None:
        return None
    columns: list[np.ndarray] = []
    for name in varlist:
        if name in ("Intercept", "_cons", "const"):
            columns.append(np.ones(len(frame), dtype=float))
            continue
        try:
            columns.append(np.asarray(frame[name], dtype=float).reshape(-1))
        except (KeyError, TypeError):
            return None
    design = np.column_stack(columns)
    if design.shape[0] != len(frame):
        return None
    return design


def _etregress_observed_treatment(
    result: Any, count: int, column: str | None = None
) -> np.ndarray | None:
    """The observed 0/1 treatment indicator, on the outcome equation's rows."""

    frame = getattr(result, "model_data", None)
    if frame is None:
        return None
    candidates: list[str] = []
    if column:
        candidates.append(column.split(":", 1)[-1])
    candidates.extend(["t", "treat", "treatment"])
    for name in candidates:
        if name in getattr(frame, "columns", []):
            values = np.asarray(frame[name], dtype=float).reshape(-1)
            if values.size == count:
                return values
    return None


def _align_selection_terms(
    result: Any,
    total: int,
    count: int,
    probability: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Line up the probit quantities with the outcome equation's rows."""
    hazard = np.where(probability > 1e-300, density / probability, np.nan)
    if total == count:
        return probability, density, hazard
    keep = _heckman_selection_mask(result, total, count)
    return probability[keep], density[keep], hazard[keep]


def _heckman_selection_mask(
    result: Any, total: int, selected_count: int
) -> np.ndarray:
    """Which rows of the full sample the outcome equation was fitted on.

    ``result.nobs`` counts the selected rows but does not say WHICH ones.  The selection
    indicator is recovered from ``model_data``: the outcome column is missing exactly where
    the observation was not selected, which is how the estimator itself defines selection
    (the docstring says a missing ``y`` marks a non-selected row).
    """

    mask = np.zeros(total, dtype=bool)
    frame = getattr(result, "model_data", None)
    if frame is not None:
        for column in ("y", "depvar"):
            if column in getattr(frame, "columns", []):
                observed = np.asarray(frame[column], dtype=float).reshape(-1)
                if observed.size == total:
                    finite = np.isfinite(observed)
                    if int(finite.sum()) == selected_count:
                        return finite
                break
        # Last resort: the first non-missing numeric column with the right count.
        for name in getattr(frame, "columns", []):
            try:
                values = np.asarray(frame[name], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if values.size != total:
                continue
            finite = np.isfinite(values)
            if int(finite.sum()) == selected_count:
                return finite
    # No reliable information: fall back to the leading rows and say so through the
    # caller's result, rather than silently pretending the mapping is exact.
    mask[:selected_count] = True
    return mask


def _predict_heckman(result: Any, exog: np.ndarray, kind: str) -> np.ndarray:
    """``heckman``'s selection-specific predictions.

    ``heckman.ado:491`` declares the set::

        est local marginsok default XB PSel YCond YExpected Mills NShazard

    and ``heckman_postestimation.sthlp:201-221`` defines each:

    ``psel``      probability of selection, ``Phi(z gamma)``
    ``mills``     nonselection hazard ``phi(z gamma)/Phi(z gamma)`` -- Heckman's inverse
    ``nshazard``  Mills ratio; a SYNONYM of ``mills`` returning the same numbers
    ``ycond``     ``E(y | observed)`` = ``xb + lambda * mills``
    ``yexpected`` ``E(y*)`` with an unobserved case counted as 0

    ``ycond`` and ``yexpected`` are the pair most easily confused.  On a row with
    ``psel = 1`` they coincide, so a test on selected rows alone cannot tell them apart;
    the tests compare them where ``psel < 1``.
    """

    from scipy import stats as _normal

    names = [str(name) for name in result.param_names]
    selection_positions = [
        index for index, name in enumerate(names) if name.startswith("select:")
    ]
    outcome_positions = [
        index
        for index, name in enumerate(names)
        if not name.startswith("select:") and name.lower() != "lambda"
    ]
    lambda_position = next(
        (index for index, name in enumerate(names) if name.lower() == "lambda"), None
    )
    if not selection_positions or not outcome_positions or lambda_position is None:
        raise UnsupportedError(
            "predict: this heckman result does not carry the usual "
            "``y:``/``lambda``/``select:`` parameter layout"
        )

    params = np.asarray(result.params, dtype=float)
    design = np.asarray(exog, dtype=float)
    selection_design = _heckman_selection_design(result)
    if selection_design is None:
        raise UnsupportedError(
            f"predict, {kind} after heckman needs the SELECTION equation's design "
            "matrix, which this result did not retain"
        )

    gamma = params[selection_positions]
    beta = params[outcome_positions]
    lambda_coefficient = float(params[lambda_position])

    linear_selection = selection_design @ gamma
    probability = _normal.norm.cdf(linear_selection)
    density = _normal.norm.pdf(linear_selection)
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(probability > 1e-300, density / probability, np.nan)

    if kind == "psel":
        return probability
    if kind in ("mills", "nshazard"):
        return hazard
    # ``ycond``/``yexpected`` predict the OUTCOME equation, so they live on the estimation
    # sample (the selected rows), while the selection design spans the FULL sample -- the
    # probit was fitted on all of it.  Align the selection quantities before combining.
    #
    # The mask must be the REAL selection indicator.  Falling back to "the first N rows"
    # is wrong whenever selection is interleaved, which it usually is: that produced a
    # ``ycond`` that differed from ``xb + lambda * mills`` in the third decimal.
    if selection_design.shape[0] != design.shape[0]:
        total = selection_design.shape[0]
        keep = _heckman_selection_mask(result, total, design.shape[0])
        probability = probability[keep]
        density = density[keep]
        hazard = hazard[keep]
    if kind == "ycond":
        return design @ beta + lambda_coefficient * hazard
    if kind == "yexpected":
        # E(y*) with an unobserved case counting as 0:
        #     Phi(z gamma) * (xb + lambda * mills) == Phi * xb + lambda * phi
        return probability * (design @ beta) + lambda_coefficient * density
    raise UnsupportedError(f"predict: {kind!r} is not a heckman selection statistic")


def _scale_sigma(result: Any) -> float:
    """The scale parameter on the SIGMA scale, whichever way the model stores it.

    ``tobit`` stores ``sigma`` directly; ``intreg`` stores ``lnsigma`` (its parameter is
    named ``/lnsigma``).  Reading the last parameter as if it were sigma gives a value near
    zero for ``intreg`` -- which silently produces probabilities of 0.5 and an expectation
    equal to the linear index, i.e. plausible-looking nonsense.  The name decides.
    """

    params = np.asarray(result.params, dtype=float)
    names = [str(name).strip().lstrip("/").lower() for name in result.param_names]
    stored = float(params[-1])
    if names and names[-1] in ("lnsigma", "lnalpha", "ln_sigma"):
        return float(np.exp(stored))
    return stored


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
    sigma = _scale_sigma(result)
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


def _predict_tobit_pr(
    result: Any, exog: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """``Pr(a < y < b)`` -- the probability the CENSORED outcome falls in the interval.

    ``tobit_postestimation.sthlp:92``: ``pr(a,b)`` is ``Pr(a < y < b)``, where ``y`` is
    the observed (censored) outcome -- not the latent ``y*``.  The distinction is what
    makes the two arguments mean something: a censoring point that falls inside ``(a, b)``
    puts the whole point mass of that boundary INTO the interval, so the probability is
    the latent mass on the part of ``(a, b)`` that is observable, plus any censoring point
    the interval covers.

    Concretely, with the model's own limit ``L`` (``e(llopt)``) and ``U`` (``e(ulopt)``):

        Pr(a < y < b) = Pr(max(a, L) < y* < min(b, U))  +  [L in (a, b)]  +  [U in (a, b)]

    where the latent part is a normal probability around ``xb``.  With no limits the
    expression reduces to the plain normal interval, which is why an uncensored fit
    returns ``Phi(z_b) - Phi(z_a)``.
    """

    from scipy import stats as _stats

    params = np.asarray(result.params, dtype=float)
    beta = params[:-1]
    sigma = _scale_sigma(result)
    xb = exog @ beta

    e_macros = getattr(result, "e", {}) or {}
    model_lower = e_macros.get("llopt")
    model_upper = e_macros.get("ulopt")
    limit_lower = float(model_lower) if model_lower is not None else None
    limit_upper = float(model_upper) if model_upper is not None else None

    # The observable part of the interval: below the model's censoring limit the outcome
    # is pinned AT the limit, so only the limit itself can carry probability.
    effective_lower = lower
    if effective_lower is not None and limit_lower is not None:
        effective_lower = max(effective_lower, limit_lower)
    elif effective_lower is None and limit_lower is not None:
        effective_lower = limit_lower
    effective_upper = upper
    if effective_upper is not None and limit_upper is not None:
        effective_upper = min(effective_upper, limit_upper)
    elif effective_upper is None and limit_upper is not None:
        effective_upper = limit_upper

    def _interval_mass(first: float | None, second: float | None) -> np.ndarray:
        if first is None and second is None:
            return np.ones_like(xb, dtype=float)
        if first is None:
            return _stats.norm.cdf((second - xb) / sigma)
        if second is None:
            return _stats.norm.sf((first - xb) / sigma)
        if second <= first:
            return np.zeros_like(xb, dtype=float)
        return _stats.norm.cdf((second - xb) / sigma) - _stats.norm.cdf(
            (first - xb) / sigma
        )

    probability = _interval_mass(effective_lower, effective_upper)

    # A censoring limit that the requested interval covers holds the point mass of every
    # observation pinned at it.
    if limit_lower is not None:
        covers_lower = (lower is None or limit_lower > lower) and (
            upper is None or limit_lower < upper
        )
        if covers_lower:
            probability = probability + _stats.norm.cdf((limit_lower - xb) / sigma)
    if limit_upper is not None:
        covers_upper = (lower is None or limit_upper > lower) and (
            upper is None or limit_upper < upper
        )
        if covers_upper:
            probability = probability + _stats.norm.sf((limit_upper - xb) / sigma)

    return np.clip(probability, 0.0, 1.0)


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
    sigma = _scale_sigma(result)
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


def _is_auxiliary_name(name: str) -> bool:
    """Whether a reported parameter is an auxiliary statistic rather than a regressor.

    Stata's spellings, all of which appear in this package's results:

    ``/sigma``       truncreg's scale
    ``/lnsigma``     intreg's log-scale
    ``/lnalpha``     nbreg's dispersion
    ``cut1``...      ordinal thresholds (with or without the leading slash)
    ``var(e.<dep>)`` tobit's scale -- the one that is easy to miss, because it carries no
                     leading slash and reads like a variable name
    """

    text = str(name).strip()
    if text.startswith("/"):
        return True
    if text.startswith("cut") and text[len("cut") :].isdigit():
        return True
    return text.startswith("var(e.")


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
        # Auxiliary parameters are the ones Stata reports after the design block: they are
        # spelled "/sigma" (truncreg), "/lnalpha" (nbreg), "cut1"/"/cut1" (ordinal) or
        # "var(e.<depvar>)" (tobit). Only the leading slash test was here, which counted
        # tobit's "var(e.y)" as a design column and made every predict call raise.
        design_block = [
            position
            for position, name in enumerate(names)
            if not _is_auxiliary_name(name)
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

    # A scale parameter reported as ``/sigma`` (truncreg after the delta transformation)
    # or ``var(e.<depvar>)`` (tobit, which is Stata's spelling there) is the innovation
    # scale itself, so sigma^2 is its square. Reading the *name* keeps this independent of
    # any estimator family list. ``var(e.y)`` was added when tobit's parameter was renamed
    # to Stata's actual row name -- without it every tobit predict broke with KeyError.
    names = [str(name) for name in getattr(result, "param_names", []) or []]
    _last = names[-1] if names else ""
    if _last in ("/sigma", "sigma") or _last.startswith("var(e."):
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


def _predict_standard_error_of_difference(
    result: Any, exog: np.ndarray
) -> np.ndarray:
    """``stddp`` for ``sqreg``: the SE of ``xb[eq1] - xb[eq2]``.

    ``qreg_postestimation.sthlp:130-132`` defines it as the standard error of the
    difference in linear predictions between equations 1 and 2, so

        Var(d) = Var(xb1) + Var(xb2) - 2 Cov(xb1, xb2)

    The cross-equation covariance is the part that matters: adding the two variances
    alone (ignoring the covariance) systematically overstates the difference whenever
    the quantile equations are positively correlated, which is the normal case.

    ``sqreg`` stores its parameters as ``q<tau>:<name>`` blocks, so the two equations
    are the first two DISTINCT prefixes in ``param_names``.
    """

    exog = np.asarray(exog, dtype=float)
    if exog.ndim != 2:  # pragma: no cover - the dispatcher validates this
        raise ValueError(
            "predict: exog must be 2-dimensional (observations x regressors), "
            f"got {exog.ndim} dimension(s)"
        )

    names = [str(name) for name in getattr(result, "param_names", [])]
    prefixes: list[str] = []
    for name in names:
        prefix = name.split(":", 1)[0] if ":" in name else ""
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    if len(prefixes) < 2:
        raise UnsupportedError(
            "predict, stddp() needs an sqreg fit with at least TWO quantiles: it is "
            "the standard error of the difference in linear predictions between "
            "equations 1 and 2 (qreg_postestimation.sthlp:130-132); this result has "
            f"{len(prefixes)} equation block(s)"
        )

    first, second = prefixes[0], prefixes[1]
    first_block = [i for i, name in enumerate(names) if name.split(":", 1)[0] == first]
    second_block = [i for i, name in enumerate(names) if name.split(":", 1)[0] == second]
    if len(first_block) != len(second_block) or len(first_block) != exog.shape[1]:
        raise UnsupportedError(
            "predict, stddp() needs both equations to use the same regressors: "
            f"equation {first!r} has {len(first_block)} and {second!r} has "
            f"{len(second_block)}, while the design has {exog.shape[1]} column(s)"
        )

    cov = np.asarray(result.cov_params(), dtype=float)
    if not np.all(np.isfinite(cov)):
        raise ConsistencyError(
            "predict: the parameter covariance contains non-finite entries; the "
            "model may be unidentified or the VCE failed"
        )
    cov = (cov + cov.T) / 2.0

    # Variance of each equation's linear index.
    variance_first = np.einsum(
        "ij,jk,ik->i", exog, cov[np.ix_(first_block, first_block)], exog
    )
    variance_second = np.einsum(
        "ij,jk,ik->i", exog, cov[np.ix_(second_block, second_block)], exog
    )
    # And their covariance, which is what keeps the difference from being overstated.
    covariance = np.einsum(
        "ij,jk,ik->i", exog, cov[np.ix_(first_block, second_block)], exog
    )

    variance = variance_first + variance_second - 2.0 * covariance
    # A negative value can only be rounding noise on a near-perfect correlation; clip
    # rather than take the square root of a negative number.
    return np.sqrt(np.maximum(variance, 0.0))


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


#: Stata's influence diagnostics (``[R] regress postestimation``).  They are
#: only defined under ``vce(ols)``: the manual states at
#: ``regress_postestimation.sthlp:152-153`` that ``stdr``, ``rstudent``,
#: ``cooksd``, ``leverage``, ``dfbeta()``, ``stdf``, ``covratio``, ``dfits`` and
#: ``welsch`` "are not available if any vce() other than vce(ols) was specified".
_DIAGNOSTIC_KINDS = frozenset(
    {
        "hat",
        "leverage",
        "rstandard",
        "rstudent",
        "cooksd",
        "dfbeta",
        "stdr",
        "covratio",
        "dfits",
        "welsch",
    }
)


def _influence_pieces(
    result: Any, exog: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """The shared quantities behind every influence diagnostic.

    Returns ``(resid, hat, rss, s2, df_resid, k, s)`` where ``hat`` is the
    diagonal of the projection matrix at the estimation sample.

    All five closed forms were verified against an explicit leave-one-out refit
    (``h_j`` via Sherman-Morrison, ``rstudent`` via the leave-one-out residual
    variance, ``cooksd`` via ``(b-b_j)'(X'X)(b-b_j)/(k s^2)``, and
    ``dfbeta`` via ``(X'X)^-1 x_j e_j/(1-h_j)``) to 1e-15.
    """

    design = np.asarray(exog, dtype=float)
    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    if design.shape[0] != resid.size:
        raise ConsistencyError(
            "influence diagnostics need the estimation sample: the design has "
            f"{design.shape[0]} rows but there are {resid.size} residuals"
        )

    covariance = np.asarray(result.cov_params(), dtype=float)
    df_resid = float(result.df_resid)
    k = float(design.shape[1])
    if df_resid <= 0:
        raise ConsistencyError(
            "influence diagnostics need positive residual degrees of freedom"
        )

    rss = float(resid @ resid)
    s2 = rss / df_resid
    s = float(np.sqrt(s2))

    counts = np.asarray(result.e["N"], dtype=float) if "N" in result.e else float(resid.size)
    if not np.isfinite(counts) or counts <= 0:
        counts = float(resid.size)
    if counts != resid.size:
        # fweight expands N beyond the row count; the leverage identity holds on
        # rows, so fall back to the row count and say so through the value.
        counts = float(resid.size)

    # ``(X'X)^-1`` from the reported covariance: ``V = s2 (X'X)^-1`` for the
    # conventional VCE, which is exactly the case diagnostics are allowed in.
    xtx_inv = covariance / s2
    hat = np.einsum("ij,jk,ik->i", design, xtx_inv, design)

    return resid, hat, rss, s2, df_resid, k, s


def _predict_diagnostic(result: Any, exog: np.ndarray, kind: str) -> np.ndarray:
    """One influence diagnostic at the estimation sample."""

    cov_type = str(getattr(result, "cov_type", "nonrobust")).strip().lower()
    if cov_type not in ("nonrobust", "ols", ""):
        raise UnsupportedError(
            "predict: the influence diagnostics (hat, leverage, rstandard, "
            "rstudent, cooksd, dfbeta) are not available after a "
            f"{cov_type} VCE; Stata restricts them to vce(ols)"
        )

    resid, hat, rss, s2, df_resid, k, s = _influence_pieces(result, exog)

    if kind in ("hat", "leverage"):
        return hat

    if kind == "rstandard":
        denominator = s * np.sqrt(np.clip(1.0 - hat, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, resid / denominator, np.nan)

    if kind == "rstudent":
        # Externally studentised (jackknifed) residual.
        #
        # This is the ratio of the LEAVE-ONE-OUT prediction residual to its own
        # standard deviation:
        #     e_j^(j) = e_j / (1 - h_j)
        #     s_(j)^2 = (rss - e_j^2/(1-h_j)) / (df_resid - 1)
        #     r_j     = e_j^(j) / s_(j)
        #
        # A tempting alternative is ``e_j / (s_(j) sqrt(1-h_j))`` -- the form
        # used for *rstandard* -- but that is a different quantity: it omits the
        # 1/(1-h_j) inflation of the leave-out residual while keeping the
        # sqrt(1-h_j) scaling, and it is off by up to 9% on the auto data.
        # Verified against an explicit refit without observation j: this form
        # reproduces it to 2.3e-10, the other to 9.0e-2.
        numerator = resid**2 / np.clip(1.0 - hat, 1e-300, None)
        s2_minus = (rss - numerator) / (df_resid - 1.0)
        leave_out = resid / np.clip(1.0 - hat, 1e-300, None)
        denominator = np.sqrt(np.clip(s2_minus, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, leave_out / denominator, np.nan)

    if kind == "cooksd":
        # Cook's D = (e_j^2 / (k s^2)) * h_j / (1 - h_j)^2
        leverage = np.clip(hat, 0.0, 1.0 - np.finfo(float).eps)
        return (resid**2 / (k * s2)) * (leverage / (1.0 - leverage) ** 2)

    if kind == "dfbeta":
        # Scale-free coefficient change: (b - b_j) / se(b).
        covariance = np.asarray(result.cov_params(), dtype=float)
        standard_errors = np.sqrt(np.diag(covariance))
        xtx_inv = covariance / s2
        leverage = np.clip(hat, 0.0, 1.0 - np.finfo(float).eps)
        # rows: observations, columns: coefficients
        change = (xtx_inv @ np.asarray(exog, dtype=float).T).T * (
            resid / (1.0 - leverage)
        )[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return change / np.where(standard_errors > 0, standard_errors, np.nan)

    if kind == "stdr":
        # Standard error of the residual: s * sqrt(1 - h_j).
        # ``fpredict.ado:62`` labels it "Std. err. of the residual".
        return s * np.sqrt(np.clip(1.0 - hat, 0.0, None))

    # The next three are built from ``rstudent`` and ``hat``, exactly as
    # ``regres_p.ado:158-191`` does:
    #     dfits    = rstudent * sqrt(h/(1-h))
    #     welsch   = dfits * sqrt((N-1)/(1-h))
    #     covratio = (((N - df_m - t^2 - 1)/(N - df_m - 2))^(df_m+1)) / (1-h)
    #                where t = resid/(rmse*sqrt(1-h))
    student = _predict_diagnostic(result, exog, "rstudent")
    leverage = np.clip(hat, 0.0, 1.0 - np.finfo(float).eps)
    scaling = np.sqrt(leverage / (1.0 - leverage))

    if kind == "dfits":
        return student * scaling

    if kind == "welsch":
        nobs = float(resid.size)
        return student * scaling * np.sqrt((nobs - 1.0) / (1.0 - leverage))

    if kind == "covratio":
        # t = resid / (rmse * sqrt(1-h)), with rmse = sqrt(s2).
        rmse = float(np.sqrt(s2))
        denominator = rmse * np.sqrt(np.clip(1.0 - hat, 1e-300, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(denominator > 0, resid / denominator, np.nan)
        nobs = float(resid.size)
        df_model = float(result.df_model)
        numerator = nobs - df_model - t_stat**2 - 1.0
        base = nobs - df_model - 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / base
            return np.where(
                ratio > 0,
                np.power(ratio, df_model + 1.0) / (1.0 - leverage),
                np.nan,
            )

    raise UnsupportedError(f"predict: unhandled diagnostic {kind!r}")


def _equation_level_scores(
    result: Any, exog: np.ndarray, model_name: str
) -> np.ndarray | None:
    """Stack every equation's score into an ``(n, k)`` array, or ``None`` if unheard of.

    Stata's ``predict, scores`` writes one NEW VARIABLE per equation.  The array-API
    equivalent keeps all of them: rows are observations, columns are the design's
    parameters, and each entry is ``d ll_i / d b_j``.  That IS the stacking the single
    array cannot replace by "the first variable" -- but it is exactly representable as a
    2-D array, which is what this returns.

    The GLM score is ``(y_i - mu_i) / Var_i * (d mu_i / d eta_i) * x_ij``; for the
    families here that collapses to a small expression, and the MLE identity
    ``sum_i s_ij == 0`` makes the result checkable without a fixture.
    """

    params = np.asarray(result.params, dtype=float)
    if exog.shape[1] > params.size:
        return None
    coefficients = params[: exog.shape[1]]
    linear = exog @ coefficients

    if model_name in ("logit", "logistic"):
        probability = 1.0 / (1.0 + np.exp(-linear))
        # ``fitted`` is the probability, so the residual is y - p.
        residual = np.asarray(result.resid, dtype=float).reshape(-1)
    elif model_name in ("probit", "dprobit"):
        from scipy import stats as _normal

        probability = _normal.norm.cdf(linear)
        density = _normal.norm.pdf(linear)
        denominator = probability * (1.0 - probability)
        if np.any(denominator <= 1e-300):  # pragma: no cover - saturated fits only
            return None
        residual = (
            np.asarray(result.resid, dtype=float).reshape(-1)
            * density
            / denominator
        )
    elif model_name == "poisson":
        mean = np.exp(linear)
        # ``resid`` is y - mu, and the score is the same expression.
        residual = np.asarray(result.resid, dtype=float).reshape(-1)
    else:
        return None

    return residual.reshape(-1, 1) * exog


def _unsupported_reason(model_name: str, kind: str, allowed: Any) -> str:
    """The message for a type the model does not offer.

    ``scores`` needs more than the generic list on EVERY command that documents it:
    it is a real option that this API cannot express, because it produces MULTIPLE
    variables -- one per equation.  The help files are explicit:

    * ``nbreg_postestimation.sthlp:172-181`` -- regression score AND dispersion score;
    * ``ologit_postestimation`` -- "The number of score variables created will equal
      the number of outcomes";
    * ``tobit``/``truncreg``/``intreg`` -- regression equation plus the auxiliary
      parameter's equation.

    Saying only "not in the list" reads as "this package lacks the feature" rather than
    "the shape cannot be returned here".

    Note the spelling asymmetry, which is Stata's own: ``poisson`` documents ``score``
    (singular, one equation) while the others document ``scores``.
    """

    normalized = str(kind).strip()
    if normalized == "scores":
        return (
            f"predict, scores after {model_name} produces MULTIPLE variables -- one "
            "per equation (regression equation plus the dispersion/auxiliary "
            "equation; for ologit, one per outcome) -- while this method returns one "
            "array.  Reporting only the first would silently drop the others. "
            "Available here: " + ", ".join(allowed)
        )
    if normalized == "score" and model_name == "nbreg":
        return (
            "predict, scores after nbreg produces TWO variables -- the derivative of "
            "the log likelihood with respect to the regression equation AND with "
            "respect to the dispersion equation (nbreg_postestimation.sthlp:172-181) "
            "-- while this method returns one array.  Reporting only the first would "
            "silently drop the dispersion score, and the Poisson expression (y - mu) "
            "would ignore the dispersion entirely. Available here: "
            + ", ".join(allowed)
        )
    return f"predict: {model_name!r} supports {allowed}; got {kind!r}"


def predict(
    result: Any,
    exog: np.ndarray,
    kind: str | None = None,
    *,
    lower: float | None = None,
    upper: float | None = None,
    outcome: int | None = None,
    n: int | float | None = None,
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

        When ``None`` the type follows Stata's per-model default
        (:func:`default_predict_kind`): ``xb`` for linear models, ``pr`` for
        logit/probit/ordinal, ``n`` for count models.  It used to default to ``xb``
        everywhere, which made a bare ``res.predict()`` return log-odds on a logit --
        neither Stata's meaning nor statsmodels'.
    lower, upper:
        Censoring limits, required by the ``tobit`` types ``e`` and ``ystar``.
        When omitted, ``result.e["llopt"]``/``result.e["ulopt"]`` are used (the older
        ``lower``/``upper`` spelling is still honoured as a fallback).
    outcome:
        Ordered models only: which category's probability ``pr`` returns
        (Stata's ``predict, pr outcome(#k)``).  ``None`` means Stata's default,
        category 1.
    """

    model_name = str(getattr(result, "model_name", ""))
    # ``kind=None`` means "whatever Stata would assume". Resolving it HERE as well as in
    # ``StataResults.predict`` keeps both entry points identical; that is the invariant the
    # two used to violate.
    if kind is None:
        kind = default_predict_kind(model_name)
    normalized = str(kind).strip()

    # A model that owns its prediction semantics takes precedence, and it also
    # defines which types exist: ``zip``/``zinb``/``ztp`` depend on the inflation
    # equation, so no generic table can enumerate their answers.
    dispatch = getattr(result, "_predict_dispatch", None)
    declared = model_predict_types(result)

    exog = np.asarray(exog, dtype=float)

    # A type the model does not declare is refused BEFORE the dispatch hook, so the
    # reason is the same whichever path the model would have taken.  ``nbreg``'s
    # ``scores`` gets an explicit explanation because it is a real option that cannot
    # be expressed as one array (``nbreg_postestimation.sthlp:172-181``: it produces
    # the regression score AND the dispersion score).
    if declared is not None and normalized not in declared:
        raise UnsupportedError(_unsupported_reason(model_name, kind, declared))

    if dispatch is not None and callable(dispatch):
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
        allowed = declared
    else:
        allowed = supported_types(model_name)
        if normalized not in allowed:
            raise UnsupportedError(_unsupported_reason(model_name, kind, allowed))

    params = np.asarray(result.params, dtype=float)

    # ``heckman``'s selection statistics are built from the SELECTION equation, which is
    # not the design matrix passed in here (that one belongs to the outcome equation).
    # They therefore have to answer before any generic path reads ``params`` as if every
    # entry multiplied a column of ``exog``.
    if model_name == "heckman" and normalized in (
        "psel",
        "ycond",
        "yexpected",
        "mills",
        "nshazard",
    ):
        return _predict_heckman(result, exog, normalized)

    # ``etregress``'s treatment statistics come from the TREATMENT equation's probit plus
    # the outcome equation's treatment coefficient -- neither of which the generic paths
    # below can combine, because ``params`` spans both equations.
    if model_name == "etregress" and normalized in (
        "cte",
        "yctrt",
        "ycntrt",
        "ptrt",
        "xbtrt",
    ):
        return _predict_etregress(result, exog, normalized)

    # Standard errors of the prediction come first: they consume the raw design
    # matrix and covariance, so they must not run through the ``xb`` branch.
    if normalized in ("stdp", "stdf"):
        return _predict_standard_error(result, exog, forecast=normalized == "stdf")

    if normalized == "stddp":
        # ``qreg_postestimation.sthlp:130-132``: "stddp is allowed only after you have
        # fit a model using sqreg.  The standard error of the difference in linear
        # predictions between equations 1 and 2 is calculated."
        #
        # So it is a SINGLE number per observation:
        #     sqrt(Var(xb1) + Var(xb2) - 2*Cov(xb1, xb2))
        # where the two indices are the first TWO quantiles of the fit.  The
        # cross-equation covariance is what makes this different from adding two
        # standard errors -- ignoring it overstates the difference whenever the two
        # equations are positively correlated, which they usually are.
        return _predict_standard_error_of_difference(result, exog)

    # Influence diagnostics.  ``regress_postestimation.sthlp:152-153`` restricts
    # them to ``vce(ols)``, so they are refused under any robust/clustered VCE
    # rather than silently computed from the wrong covariance.
    if normalized in _DIAGNOSTIC_KINDS:
        return _predict_diagnostic(result, exog, normalized)

    if normalized in ("xb", "fitted"):
        return exog @ params[: exog.shape[1]]

    if normalized == "residuals":
        return np.asarray(result.resid, dtype=float).reshape(-1)

    # ``score`` -- ``poisson_postestimation.sthlp:163-164``: "the equation-level score,
    # the derivative of the log likelihood with respect to the linear prediction."
    # For a Poisson, ``ll = y*eta - exp(eta) - log(y!)``, so ``d ll / d eta = y - mu``.
    #
    # It is NOT the same as the residual ``y - mu`` for every family: the negative
    # binomial score carries the dispersion factor, so reusing the Poisson expression
    # there would report a number for a different likelihood.
    if normalized == "scores":
        # ``scores`` produces ONE variable per equation.  Returning them stacked as an
        # (n x k) array is the array-API equivalent of Stata writing k new variables:
        # every equation's score is present, in the design's column order, so nothing is
        # dropped.  The single-variable refusal stays for models whose equations are not
        # aligned with the design (see ``_unsupported_reason``).
        stacked = _equation_level_scores(result, exog, model_name)
        if stacked is not None:
            return stacked

    if normalized == "score":
        if model_name == "regress":
            # ``regress_postestimation.sthlp:193``: "score is equivalent to residuals in
            # linear regression."  The equation-level score of a Gaussian likelihood IS
            # the residual divided by sigma^2 up to a constant that cancels in the
            # sandwich, which is why Stata documents the two as the same variable.
            return np.asarray(result.resid, dtype=float).reshape(-1)
        if model_name in ("logit", "logistic"):
            # ``d ll / d eta`` for a LOGIT is ``y - p``, which is exactly the stored
            # residual (``fitted`` is the probability).  No link factor belongs here:
            # the derivative of log p with respect to eta is (1 - p) and of log(1 - p)
            # is -p, so they combine to y - p.
            return np.asarray(result.resid, dtype=float).reshape(-1)
        if model_name in ("probit", "dprobit"):
            # A PROBIT's score is NOT ``y - p``.  Its log-likelihood derivatives carry
            # the normal density over the probability:
            #
            #     d ll / d eta = (y - p) * phi(eta) / (p (1 - p))
            #
            # Reusing the logit expression here produced a vector whose sum was 0.25
            # instead of 0 on a fit with a constant -- the intercept's first-order
            # condition is ``sum_i d ll_i / d eta_i == 0``, so that non-zero sum is what
            # exposed the wrong link.  The corrected expression sums to zero to machine
            # precision.
            from scipy import stats as _normal

            linear = np.asarray(result.predict(kind="xb"), dtype=float).reshape(-1)
            probability = _normal.norm.cdf(linear)
            density = _normal.norm.pdf(linear)
            denominator = probability * (1.0 - probability)
            if np.any(denominator <= 0.0):  # pragma: no cover - saturated fits only
                raise UnsupportedError(
                    "predict, score after probit: a fitted probability is 0 or 1, so "
                    "the score's density ratio is undefined"
                )
            residuals = np.asarray(result.resid, dtype=float).reshape(-1)
            return residuals * density / denominator
        if model_name != "poisson":
            raise UnsupportedError(
                f"predict, score after {model_name} is not implemented: the score "
                "carries the family's own log-likelihood derivative, so the Poisson "
                "expression (y - mu) would report a number for a different "
                "likelihood. Available here: "
                + ", ".join(SUPPORTED_TYPES.get(model_name, ()))
            )
        mean = np.exp(exog @ params[: exog.shape[1]])
        observed = np.asarray(result.resid, dtype=float).reshape(-1) + np.asarray(
            result.fittedvalues, dtype=float
        ).reshape(-1)
        return observed - mean

    # ``nbreg`` spells it ``scores`` (with the s) and it produces TWO variables:
    # ``nbreg_postestimation.sthlp:172-181`` -- "The first new variable will contain
    # the derivative of the log likelihood with respect to the regression equation.
    # The second new variable will contain the derivative of the log likelihood with
    # respect to the dispersion equation."
    #
    # This API returns ONE array per call, so the shape cannot be honoured.  Refusing
    # with that reason beats returning one of the two and letting the caller believe it
    # is the whole statistic -- or reusing the Poisson ``y - mu``, which ignores the
    # dispersion equation entirely.
    if normalized in ("scores", "score") and model_name == "nbreg":
        raise UnsupportedError(
            "predict, scores after nbreg produces TWO variables -- the derivative "
            "with respect to the regression equation AND with respect to the "
            "dispersion equation (nbreg_postestimation.sthlp:172-181) -- while this "
            "method returns one array.  Reporting only the first would silently drop "
            "the dispersion score, and the Poisson expression (y - mu) would ignore "
            "the dispersion entirely. Available here: "
            + ", ".join(SUPPORTED_TYPES.get(model_name, ()))
        )

    if normalized in _PANEL_FE_KINDS:
        return _predict_panel_components(result, exog, normalized)

    if normalized in ("pr", "n", "ir"):
        # Ordered models: ``pr`` is the probability of one *category*, which is a
        # cut-point difference rather than a link function of the linear index.
        # Stata's default category is 1 (``predict, pr`` after ``ologit rep78``
        # equals P(y=1) to 0.0e+00, and ``margins, dydx(weight)`` reproduces
        # ``predict(pr outcome(1))``), so ``outcome`` follows that.
        if normalized == "pr" and ordered_structure(result) is not None:
            return ordered_probability(result, exog, outcome=outcome)

        # ``n`` is a count mean whose link is the exponential for EVERY count
        # family (poisson, nbreg, ztp, tnbreg).  It uses only the DESIGN columns,
        # so extra parameters (nbreg's ``/lnalpha``, an inflation equation) are
        # sliced away rather than treated as an obstacle.  This used to be
        # unreachable for ``nbreg`` because the parameter/column mismatch check
        # below fired first -- even though ``predict, n`` is nbreg's DEFAULT type
        # (``nbreg_postestimation.sthlp:91``: "number of events; the default").
        # ``poisson_postestimation.sthlp:141-142``: "ir calculates the incidence rate
        # exp(xb), which is the predicted number of events when exposure is 1.
        # Specifying ir is equivalent to specifying nooffset."  With no offset or
        # exposure in the model -- which is the only case this package fits -- that is
        # exactly ``n``, so ``ir`` is an alias rather than a second formula.
        if normalized in ("n", "ir"):
            return np.exp(exog @ params[: exog.shape[1]])

        # ``pr(n)`` -- the probability mass at a count, NOT the count mean.
        # ONLY the count families reach here: for ``logit``/``probit``/``ologit`` the
        # SAME spelling ``pr`` means the plain predicted probability and takes no
        # argument, so this branch must be gated on the family rather than on the
        # spelling alone.  ``poisson_postestimation.sthlp:147-148``: "pr(n) calculates
        # the probability Pr(y = n), where n is a nonnegative integer".  Without ``n``
        # the type is INCOMPLETE rather than unavailable, and saying so is more useful
        # than "unsupported".
        if normalized == "pr" and model_name in ("poisson", "nbreg"):
            target = n
            # ``pr(a, b)`` -- the interval form, ``Pr(a < y < b)``.  The help text at
            # ``poisson_postestimation.sthlp:106`` prints STRICT inequalities
            # (``{ul:<}``), so both endpoints are excluded; ``pr(a,b)`` is therefore
            # ``cdf(b-1) - cdf(a)`` for integer counts, not ``cdf(b) - cdf(a)``.
            if target is None and (lower is not None or upper is not None):
                target = None
            elif target is None:
                raise ValueError(
                    f"predict, pr() needs the count: {model_name} defines "
                    "pr(n) = Pr(y = n) and pr(a, b) = Pr(a < y < b) "
                    "(poisson_postestimation.sthlp:105-106, 147-148).  "
                    "Pass n=<integer> for a point mass, or lower=/upper= for an "
                    "interval."
                )
            mean = np.exp(exog @ params[: exog.shape[1]])
            from scipy import stats as _dist

            if model_name == "poisson":
                distribution = _dist.poisson(mean)
            else:
                # ``nbreg`` is a NEGATIVE BINOMIAL, not a Poisson.  Using the Poisson
                # pmf here would return a plausible number that ignores the dispersion
                # parameter entirely.
                #
                # Stata: ``Var(y) = mu + alpha*mu^2`` with ``alpha = exp(lnalpha)``.
                # scipy's ``nbinom`` wants ``(size, p)`` with ``size = 1/alpha`` and
                # ``p = 1/(1 + alpha*mu)``, giving mean ``size*(1-p)/p = mu``.
                lnalpha_position = next(
                    (
                        index
                        for index, name in enumerate(
                            getattr(result, "param_names", [])
                        )
                        if str(name).strip().lstrip("/").lower()
                        in ("lnalpha", "alpha")
                    ),
                    None,
                )
                if lnalpha_position is None:  # pragma: no cover - defensive
                    raise UnsupportedError(
                        f"predict, pr() needs the dispersion parameter of "
                        f"{model_name!r}, which this result does not expose"
                    )
                coefficients = np.asarray(result.params, dtype=float)
                alpha = float(np.exp(coefficients[lnalpha_position]))
                distribution = _dist.nbinom(1.0 / alpha, 1.0 / (1.0 + alpha * mean))

            if target is not None:
                return distribution.pmf(target)

            # Interval: strictly between the bounds.  A missing bound is open-ended.
            upper_mass = (
                1.0 if upper is None else distribution.cdf(np.asarray(upper) - 1)
            )
            lower_mass = 0.0 if lower is None else distribution.cdf(lower)
            return upper_mass - lower_mass

        # A response-scale prediction combines *all* structural parameters.  When
        # the model carries parameters beyond its design columns (an inflation
        # equation, a dispersion, ...) the generic formula would silently use the
        # wrong subset -- this used to raise a bare ``matmul`` dimension error
        # for ``zip``.  Refuse explicitly and name the types that do work; the
        # model itself can still answer through ``_predict_dispatch``.
        #
        # ``tobit`` is handled BEFORE this check: it has a sigma parameter that is not a
        # design column, which is exactly the situation being guarded against, and its
        # own branches below know how to split ``params`` correctly.
        if model_name in ("tobit", "intreg") and normalized in ("e", "ystar", "pr"):
            return _predict_tobit_family(
                result, exog, normalized, lower=lower, upper=upper
            )
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
        link = "logit" if model_name in ("logit", "ologit") else "probit"
        return _predict_binary(result, link, exog)

    if model_name in ("tobit", "intreg") and normalized in ("e", "ystar", "pr"):
        return _predict_tobit_family(
            result, exog, normalized, lower=lower, upper=upper
        )

    raise UnsupportedError(
        f"predict: type {kind!r} is registered for {model_name!r} but has no "
        "implementation; this is a westata bug"
    )


def _predict_tobit_family(
    result: Any,
    exog: np.ndarray,
    normalized: str,
    *,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    """Dispatch ``tobit``'s three interval statistics: ``e``, ``ystar`` and ``pr``.

    Split out of ``predict`` because the family has a sigma parameter that is not a design
    column, so it must be answered BEFORE the generic "params != design columns" guard
    refuses it -- the guard exists to catch exactly that shape, and tobit is the one model
    that legitimately has it and knows how to handle it.

    ``tobit.ado:131-132`` stores the limits as ``e(llopt)``/``e(ulopt)``; the older
    ``lower``/``upper`` spelling is still honoured so results built before the rename
    keep working.
    """

    stored_lower = result.e.get("llopt", result.e.get("lower"))
    stored_upper = result.e.get("ulopt", result.e.get("upper"))
    lower_limit = (
        float(stored_lower) if lower is None and stored_lower is not None else lower
    )
    upper_limit = (
        float(stored_upper) if upper is None and stored_upper is not None else upper
    )
    if normalized == "pr":
        # ``Pr(a < y < b)`` on the censored outcome; takes the same lower/upper arguments
        # as ``e(a,b)``/``ystar(a,b)`` (``tobit_postestimation.sthlp:92``).
        return _predict_tobit_pr(result, exog, lower, upper)
    if normalized == "e":
        return _predict_tobit_e(result, exog, lower_limit, upper_limit)
    return _predict_tobit_ystar(result, exog, lower_limit, upper_limit)
