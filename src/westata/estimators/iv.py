"""Instrumental variables: Stata ``ivregress 2sls`` / ``liml`` / ``gmm``.

The point estimates come from ``linearmodels.iv`` (``IV2SLS`` / ``IVLIML`` /
``IVGMM``), which solves the same estimating equations Stata does, so this
module only owns the Stata-facing layer: the ``cov_type`` mapping, the degrees
of freedom, and the ``e()`` scalars.

Estimator / backend mapping
---------------------------
======================  ==========================  =====================
westata class           linearmodels backend        Stata command
======================  ==========================  =====================
:class:`IV2SLS`         ``linearmodels.iv.IV2SLS``  ``ivregress 2sls``
:class:`IVLIML`         ``linearmodels.iv.IVLIML``  ``ivregress liml``
:class:`IVGMM`          ``linearmodels.iv.IVGMM``   ``ivregress gmm``
======================  ==========================  =====================

Formula syntax
--------------
``"y ~ exog1 + exog2 + [endog ~ instr1 + instr2]"`` -- linearmodels' native
``from_formula`` convention (the one statsmodels users also expect for IV).
Parsing is delegated to ``linearmodels.iv.IV2SLS.from_formula``; westata does
not reimplement that grammar, and it inherits linearmodels' listwise deletion
of rows with missing values (linearmodels emits its own
``MissingValueWarning``).

Degrees of freedom (``e(df_r)``)
--------------------------------
Stata's ``ivregress`` leaves ``e(df_r)`` **missing** by default and reports
``N - K`` (or ``G - 1`` under ``vce(cluster)``) only when the ``small`` option
is given, where ``K`` is the number of *estimated parameters* (exogenous plus
endogenous regressors, i.e. the columns of the regressor matrix), and
``e(rank) = K``.  **Excluded instruments are not estimated parameters and
consume no degree of freedom**: they only serve to identify the endogenous
regressors, so ``N - K`` is unchanged by over-identification.  This is the
convention in ``[R] ivregress`` (Method and formulas): the residual variance is
``s^2 = u'u / (N - K)`` with ``K`` the columns of ``X``, not ``N`` minus the
number of instruments.  Consequently ``e(df_m) = K - 1`` when the model
contains a constant (``K`` otherwise), and adding a second instrument to an
exactly identified model leaves both ``e(df_r)`` and ``e(df_m)`` untouched.

westata cannot store a missing there (see ``e(df_r)`` under "Known deviations
from Stata" below), so it stores ``N``; the *inference* degrees of freedom it
uses are the same ``N`` for nonrobust/HC and ``G - 1`` under clustering.

Covariance estimators
---------------------
``cov_type`` is validated against :data:`westata.vce.dispatch.
SUPPORTED_COV_TYPES` and mapped as follows:

======================  ==========================================================
westata ``cov_type``    Stata option / linearmodels call
======================  ==========================================================
``None``/``nonrobust``  (default) ``"unadjusted"``, ``debiased=False`` = ESS/N
``HC0``                 ``vce(robust)``: ``"robust"``, ``debiased=False``
``HC1``                 ``vce(robust) small``: ``"robust"``, ``debiased=True``
``cluster``             ``vce(cluster g)``: ``"clustered"``, ``debiased=False``
                        (CR0); ``ssc=SSC(adj=True)`` adds ``small`` -> CR1
``HC2``, ``HC3``        rejected (:class:`westata.errors.UnsupportedError`)
======================  ==========================================================

**``HC0`` is Stata's plain ``vce(robust)``; ``HC1`` is ``vce(robust) small``.**
This is the opposite of OLS/``regress``, where ``vce(robust)`` *is* HC1, so the
mapping is spelled out here and pinned by ``tests/test_iv.py``.  Verified live
on StataNow 19.5 (2026-10-24, 140x14 panel, ``ivregress 2sls y x1 (e1 = z1
z2)``): westata ``HC0`` matches ``vce(robust)`` to 6.1e-15 whereas ``HC1`` is
larger by exactly ``sqrt(N/(N-K))`` = 1.0109, and ``HC1`` matches
``vce(robust) small`` to 6.0e-15.

Likewise ``vce(cluster g)`` is CR0 for **all three** estimators: Stata adds
``G/(G-1)`` and ``(N-1)/(N-K)`` only under ``small``.  westata's default
``ssc`` for IV is therefore ``SSC(adj=False)`` and 2SLS/LIML/GMM share it.

Why the 2SLS/GMM split
~~~~~~~~~~~~~~~~~~~~~~
For 2SLS (``kappa == 1``) the Stata/linearmodels VCE is the familiar sandwich
built from the *projected* regressors ``Xhat = P_Z X``:

    V = (Xhat'Xhat)^-1 [meat] (Xhat'Xhat)^-1

which is exactly what :func:`westata.vce.dispatch.compute_cov` computes when
handed ``Xhat`` and the structural residuals.  ``IV2SLS`` therefore routes its
whole covariance through ``compute_cov``, so the ``N/(N-K)``, ``G/(G-1)`` and
``(N-1)/(N-K)`` corrections live in one place and ``ssc`` is honoured for
every ``cov_type``.

LIML and GMM cannot go through ``compute_cov``: their bread is the k-class
matrix ``(1-kappa) X'X + kappa X'P_Z X`` (LIML) or the GMM weight-matrix
sandwich ``(X'ZWZ'X)^-1`` (GMM), which the OLS-shaped ``compute_cov`` API
cannot express (for LIML ``kappa > 1`` makes the k-class bread indefinite, so
no real ``A`` with ``A'A = bread`` exists).  Those two estimators therefore use
the backend covariance explicitly -- which is *not* a divergence from Stata:
linearmodels' ``kappa``-corrected ``HomoskedasticCovariance`` /
``HeteroskedasticCovariance`` / ``ClusteredCovariance`` are the same formulas
as ``[R] ivregress`` (Baum, Schaffer & Stillman 2003).  ``ssc`` is still
honoured: ``ssc.adj`` maps to ``debiased`` and ``ssc.cluster_adj=False``
divides the clustered matrix by ``G/(G-1)``.

Known deviations from Stata
---------------------------
* ``HC2``/``HC3`` are rejected for all three estimators.  Stata's
  ``ivregress`` offers neither and linearmodels has no leverage-corrected IV
  sandwich, so the only alternative would be to return an HC0 number under an
  HC2/HC3 label.
* Stata's ``ivregress`` prints ``z`` statistics and a Wald ``chi2``: the
  reported table is ``Coefficient / Std. err. / z / P>|z| / [95% conf.
  interval]`` with a ``Wald chi2(k)`` header (see ``[R] ivregress``, Example 1).
  westata therefore uses ``dist="z"`` for IV, unlike OLS/panel which use t.
  ``e(chi2)`` records the undivided Wald statistic and ``e(df_m)`` its degrees
  of freedom.
* ``e(df_r)``: Stata stores missing (``.``) for ``ivregress`` unless ``small``
  is given, and ``N - K`` (or ``G - 1`` under clustering) with ``small``.
  **westata stores ``N`` instead, because that slot is also the inference
  degrees of freedom (``results.df_resid``).  A missing value there silently
  turns every ``pvalues`` and ``conf_int`` entry into NaN** -- measured on this
  very result object, ``df_resid=nan`` yields ``pvalues=[nan, nan]`` and
  ``conf_int()`` with all-NaN bounds.  Reproducing Stata's missingness
  bit-for-bit would therefore require the results layer to separate the two
  concepts (``e(df_r)`` as reported data vs ``df_resid`` as the reference
  distribution), which is a ``results/base.py`` change outside this round's
  scope.  The current value is the pre-existing convention, now documented
  rather than silently wrong.  ``e(rmse)`` follows the same ``small`` rule as
  Stata.
* ``e(ll)`` is the Gaussian log-likelihood evaluated at the reported estimates,
  which is Stata's ``e(ll)`` definition for ``ivregress 2sls``.  Stata's
  ``ivregress liml`` instead stores the concentrated *LIML* likelihood, which
  is not reproduced here (documented, not silently approximated).
* ``e(F)`` is the overall Wald test of all non-constant coefficients divided by
  its degrees of freedom (the same statistic linearmodels reports as
  ``f_statistic``); ``e(F_first)`` is the smallest first-stage F over the
  endogenous regressors (``estat firststage`` analogue).  A first-stage F below
  10 emits a ``UserWarning`` rather than an error.  Stata itself stores only
  ``e(chi2)`` (or ``e(F)`` under ``small``), so ``e(F)`` is a westata
  convenience that is *derived* from ``e(chi2)`` and ``e(df_m)``.
* Weights follow ``ivregress``: ``aweight``/``fweight``/``pweight``/``iweight`` are
  accepted and normalized by :func:`westata.data.weights.normalize_weights`, the same
  routine ``regress`` uses, so a weight means the same thing in both. A weighted IV is an
  IV on the ``sqrt(w)``-transformed data throughout -- point estimates, the sandwich, and
  the fit statistics -- because transforming only part of the pipeline yields numbers that
  look right to three decimals and are not.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS as _LMIV2SLS
from linearmodels.iv import IVGMM as _LMIVGMM
from linearmodels.iv import IVLIML as _LMIVLIML

from westata.data.weights import normalize_weights
from westata.errors import ConsistencyError, UnsupportedError, WeightError
from westata.protocols import SSC
from westata.results.base import StataResults
from westata.vce.dispatch import SUPPORTED_COV_TYPES, compute_cov
from westata.vce.ssc import resolve_ssc

__all__ = ["IV2SLS", "IVLIML", "IVGMM"]

#: ``cov_type`` values accepted by the IV estimators.  ``HC2``/``HC3`` are in
#: :data:`westata.vce.dispatch.SUPPORTED_COV_TYPES` for OLS but have no IV
#: analogue, so they are rejected here (see the module docstring).
_IV_COV_TYPES = tuple(
    t for t in SUPPORTED_COV_TYPES if t not in ("HC2", "HC3")
)

#: linearmodels covariance names, verified against linearmodels 7.1
#: (``COVARIANCE_ESTIMATORS`` in ``linearmodels/iv/model.py``).
_LM_COV_TYPES = {
    "nonrobust": "unadjusted",
    "HC0": "robust",
    "HC1": "robust",
    "cluster": "clustered",
    # ``kernel`` is linearmodels' Newey-West covariance; it accepts ``kappa`` so
    # it works for the k-class (LIML) and GMM paths, which ``compute_cov`` cannot
    # express.  Used only for LIML/GMM; 2SLS routes HAC through ``compute_cov``.
    "HAC": "kernel",
}

_WEAK_INSTRUMENT_F = 10.0

#: Stata spells the weight option as a PREFIX (``[aweight=w]``), and some callers copy that
#: spelling as a keyword. ``weights``/``weight_type`` are now real parameters, so only these
#: plural aliases are rejected -- loudly, because silently ignoring a weight would change
#: every number in the table.
_WEIGHT_KEYWORDS = frozenset({"fweights", "aweights", "pweights", "iweights"})


def _ensure_intercept(formula: str) -> str:
    """Add a constant to an IV formula unless the user removed it.

    Stata's ``ivregress`` includes a constant by default; linearmodels does not.
    Matching Stata means adding ``1 +`` to the right-hand side, unless the
    formula already has an intercept or explicitly suppresses it with ``0``/``-1``.
    """

    if "~" not in formula:
        return formula
    lhs, rhs = formula.split("~", 1)
    # The exogenous part is everything before the "[" block.
    if "[" in rhs:
        exogenous, rest = rhs.split("[", 1)
        instrument_block = "[" + rest
    else:
        exogenous, instrument_block = rhs, ""

    stripped = exogenous.strip()
    has_intercept = "intercept" in stripped.lower()
    suppresses = any(
        token in stripped.split() for token in ("0", "-1", "+0", "+ -1")
    )
    if has_intercept or suppresses or (stripped.startswith("0") and len(stripped) >= 1):
        return formula

    return f"{lhs}~ 1 + {stripped} {instrument_block}".rstrip()


def _reject_unsupported_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Reject weight keywords loudly and unknown keywords with ``TypeError``."""

    for key in kwargs:
        if key in _WEIGHT_KEYWORDS:
            raise UnsupportedError(
                f"{key!r} is Stata's weight PREFIX, not a keyword. Use "
                "weights=<array> with weight_type='aweight'/'fweight'/'pweight'/"
                "'iweight' instead; the weight is never silently ignored."
            )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")


def _as_2d(value: Any, name: str, nobs: int) -> np.ndarray:
    """Coerce ``value`` to an ``nobs x k`` float array (``None`` -> ``nobs x 0``)."""

    if value is None:
        return np.empty((nobs, 0), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got {array.ndim} dimensions")
    if array.shape[0] != nobs:
        raise ValueError(f"{name} has {array.shape[0]} rows but endog has {nobs} rows")
    return array


def _canonical_cov_type(cov_type: str | None) -> str:
    """Map a user ``cov_type`` to the canonical IV spelling.

    For IV the Stata correspondence is ``HC0`` == ``vce(robust)`` and ``HC1`` ==
    ``vce(robust) small`` (see the module docstring); the names themselves are
    kept because they are the package-wide vocabulary shared with OLS.
    """

    if cov_type is None:
        return "nonrobust"
    key = str(cov_type).strip().upper()
    # ``robust`` IS STATA'S OPTION NAME, and for ivregress it means HC0 -- that is what
    # `ivregress 2sls ..., vce(robust)` computes (HC0, not HC1). Accepting the word keeps a
    # caller from having to remember which numeric spelling each command family wants, which is
    # the deviation the v3 report records at section 4, item (5).
    if key == "ROBUST":
        return "HC0"
    canonical = {t.upper(): t for t in _IV_COV_TYPES}.get(key)
    if canonical is None:
        if key in ("HC2", "HC3"):
            raise UnsupportedError(
                f"cov_type={cov_type!r} is not available for IV estimators: "
                "Stata's 'ivregress' has no leverage-corrected robust VCE and "
                "linearmodels has no IV analogue, so westata will not report "
                "an HC0 number under an HC2/HC3 label. Use 'HC0', 'HC1', "
                "'cluster' or 'nonrobust'."
            )
        raise ValueError(
            f"Unknown cov_type {cov_type!r}; supported cov_type values are: "
            + ", ".join(_IV_COV_TYPES)
        )
    return canonical


def _cluster_labels(
    clusters: Any, nobs: int, kept_index: Sequence[Any] | None
) -> Any:
    """Return cluster labels aligned with the estimation sample.

    One-dimensional for ordinary clustering, or ``nobs x dimensions`` for MULTIWAY
    clustering, which Stata spells ``vce(cluster a b)`` and computes by
    Cameron-Gelbach-Miller inclusion-exclusion. Multiway used to be refused here, so the
    ADHM (2020) replication could not produce the paper's own standard error; the
    combination step now lives in :func:`westata.vce.cluster.cov_multiway_cluster`.
    """

    if isinstance(clusters, pd.DataFrame) and clusters.shape[0] != nobs and kept_index is not None:
        if clusters.index.is_unique and set(kept_index).issubset(set(clusters.index)):
            clusters = clusters.loc[list(kept_index)]
    elif isinstance(clusters, pd.Series) and clusters.shape[0] != nobs and kept_index is not None:
        # The usual case: the user passes a column of the *original* frame while
        # the estimator dropped rows with missing values.  Align by index
        # instead of guessing positions.
        if clusters.index.is_unique and set(kept_index).issubset(set(clusters.index)):
            clusters = clusters.loc[list(kept_index)]

    values = np.asarray(clusters)
    if values.ndim == 2 and values.shape[1] > 1:
        # Multiway: each column is one clustering dimension. Passed through as a 2-D array
        # so the dispatcher can split it; the length check is on the rows.
        if values.shape[0] != nobs:
            raise ValueError(
                "cov_kwds['clusters'] must have one entry per estimation "
                f"observation; got {values.shape[0]} for nobs={nobs}"
            )
        return values
    values = np.asarray(clusters.squeeze() if hasattr(clusters, "squeeze") else clusters)
    values = values.reshape(-1)
    if values.shape[0] != nobs:
        raise ValueError(
            "cov_kwds['clusters'] must have one entry per estimation "
            f"observation; got {values.shape[0]} labels for nobs={nobs}"
        )
    return values


def _quadratic_form(params: np.ndarray, cov: np.ndarray) -> float:
    """Return ``params' cov^-1 params``, or NaN when ``cov`` is singular."""

    if params.size == 0:
        return float("nan")
    try:
        return float(params @ np.linalg.solve(cov, params))
    except np.linalg.LinAlgError:
        return float("nan")


#: ``e(title)`` per ``ivregress`` estimator -- the banner each prints. Measured by running
#: all three against StataNow 19.5:
#:
#:   Instrumental-variables 2SLS regression
#:   Instrumental-variables LIML regression
#:   Instrumental-variables GMM regression
_IV_TITLES = {
    "ivregress 2sls": "Instrumental-variables 2SLS regression",
    "ivregress liml": "Instrumental-variables LIML regression",
    "ivregress gmm": "Instrumental-variables GMM regression",
}


class IV2SLS:
    """Two-stage least squares (Stata ``ivregress 2sls``).

    Parameters
    ----------
    endog:
        Dependent variable, length ``n`` (linearmodels calls this
        ``dependent``; it is named ``endog`` here for the Stata ``ivregress``
        reading "the left-hand-side variable").
    exog:
        Exogenous regressors, ``n x k1``, including the constant when wanted.
        ``None`` (or a zero-column array) fits a model without exogenous
        regressors.
    endog_instr:
        Endogenous regressors, ``n x k2``.  May be ``None`` for a model without
        endogenous regressors (the estimator then reduces to OLS).
    instruments:
        Excluded instruments, ``n x p`` with ``p >= k2``.  Required whenever
        ``endog_instr`` has columns.
    param_names:
        Coefficient names in ``[exog..., endog_instr...]`` order.  Defaults to
        ``x1, x2, ...`` for ``exog`` and ``endog1, ...`` for ``endog_instr``.

    Notes
    -----
    Weights are supported and follow ``ivregress``: ``weight_type`` is required whenever
    ``weights`` is given, because Stata never guesses it. ``e(N)`` is the effective sample
    size (the weight sum for ``fweight``, the row count otherwise) and ``e(rss)`` is on the
    normalised weight scale, matching ``regress``.
    """

    _backend_cls = _LMIV2SLS
    _stata_command = "ivregress 2sls"
    _uses_shared_vce = True
    #: ``ivregress 2sls`` stores ``e(rank_mb)`` when ``vce(robust)``/``cluster``
    #: is used; LIML and GMM do not (see :meth:`_stata_scalars`).
    _stores_rank_mb = True

    def __init__(
        self,
        endog: Any,
        exog: Any,
        endog_instr: Any,
        instruments: Any,
        *,
        param_names: Sequence[str] | None = None,
        weights: Any = None,
        weight_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        _reject_unsupported_kwargs(kwargs)

        y = np.asarray(endog, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError(f"endog must be one-dimensional; got shape {y.shape}")
        nobs = y.shape[0]

        X_exog = _as_2d(exog, "exog", nobs)
        X_endog = _as_2d(endog_instr, "endog_instr", nobs)
        Z_instr = _as_2d(instruments, "instruments", nobs)

        k_exog, k_endog = X_exog.shape[1], X_endog.shape[1]
        k_params = k_exog + k_endog
        if k_params == 0:
            raise ValueError(
                "the model has no regressors: pass at least one exogenous or "
                "endogenous regressor"
            )
        if k_endog > 0 and Z_instr.shape[1] < k_endog:
            raise ValueError(
                f"{k_endog} endogenous regressor(s) need at least as many "
                f"instruments; got {Z_instr.shape[1]}"
            )
        if k_endog == 0 and Z_instr.shape[1] > 0:
            raise ValueError(
                "instruments were supplied without endogenous regressors; "
                "drop the instruments or add the endogenous regressors"
            )
        if nobs <= k_params:
            raise ConsistencyError(
                f"{self._stata_command}: {nobs} observations but {k_params} "
                "parameters; the model is not identified"
            )

        if param_names is None:
            names = [f"x{i + 1}" for i in range(k_exog)] + [
                f"endog{i + 1}" for i in range(k_endog)
            ]
        else:
            names = [str(name) for name in param_names]
            if len(names) != k_params:
                raise ValueError(
                    f"param_names has {len(names)} entries but the model has "
                    f"{k_params} parameters (exog + endog_instr)"
                )
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique")

        self._y = y
        # ---- weights -------------------------------------------------------
        #
        # ``ivregress`` accepts all four Stata weight types (``ivregress.sthlp``
        # "Weights ... aweights, fweights, iweights, and pweights are allowed"). This
        # used to raise UnsupportedError; the ADH replication needed it for 153 of 166
        # specifications, so the caller had to rescale by sqrt(w) by hand -- including
        # the constant, and forgetting that gives a silent 4.4% bias.
        #
        # The normalization (and therefore the Stata semantics of each weight type) comes
        # from the same ``normalize_weights`` OLS uses, so a weighted IV and a weighted
        # OLS agree on what ``aweight`` MEANS rather than re-deriving it here.
        self._weight_spec = None
        self._weights = None
        if weights is not None:
            if weight_type is None:
                raise WeightError(
                    "weights were supplied without weight_type; specify one of "
                    "'fweight', 'aweight', 'pweight', 'iweight' (Stata never "
                    "guesses the weight type)"
                )
            spec = normalize_weights(weights, weight_type)
            if spec.weights.shape[0] != nobs:
                raise WeightError(
                    f"weights have {spec.weights.shape[0]} entries but the data "
                    f"have {nobs} rows"
                )
            self._weight_spec = spec
            self._weights = np.asarray(spec.weights, dtype=float)
        elif weight_type is not None:
            raise WeightError("weight_type was supplied without weights")

        self._X_exog = X_exog
        self._X_endog = X_endog
        self._Z_instr = Z_instr
        self._param_names = names
        self._kept_index: list[Any] | None = None
        #: Names of the three regressor blocks; populated by ``from_formula``.
        #: The array constructor leaves them positional placeholders.
        self._exog_names: list[str] = [
            f"x{i}" for i in range(0 if X_exog is None else X_exog.shape[1])
        ]
        self._endog_names: list[str] = [
            f"endog{i}" for i in range(0 if X_endog is None else X_endog.shape[1])
        ]
        self._instr_names: list[str] = [
            f"instr{i}" for i in range(0 if Z_instr is None else Z_instr.shape[1])
        ]
        #: Stata's ``e(b)`` ordering (endogenous first).  ``from_formula`` fills
        #: it; the array constructor leaves it empty, so ``stata_params`` falls
        #: back to the design order.
        self._formula_rhs_order: list[str] = []

    # -------------------------------------------------------------- builders
    @classmethod
    def from_formula(cls, formula: str, data: pd.DataFrame, **kwargs: Any) -> "IV2SLS":
        """Build the model from ``"y ~ exog + [endog ~ instruments]"``.

        The grammar is linearmodels' (and statsmodels') IV convention; parsing
        is delegated to ``linearmodels.iv.IV2SLS.from_formula`` so that the
        supported syntax is exactly the backend's.  Rows with missing values in
        any of the used columns are dropped by linearmodels, and the same
        sample is used for estimation and for ``cov_kwds['clusters']``
        alignment.
        """

        # ``weights``/``weight_type`` are legal here, so they come out BEFORE the
        # unexpected-keyword check; leaving them in made them look like typos.
        weights = kwargs.pop("weights", None)
        weight_type = kwargs.pop("weight_type", None)
        subset = kwargs.pop("subset", None)
        _reject_unsupported_kwargs(kwargs)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # ``subset=`` restricts the estimation sample, the same way ``OLS`` does and the
        # way Stata's ``if``/``in`` do. It was missing here while present on OLS/Logit/
        # Poisson, so "estimate on a subset" needed a different spelling per estimator.
        #
        # Applied BEFORE parsing so that every downstream step -- the kept index, the
        # weight alignment, the cluster labels -- sees the same rows. Pre-filtering the
        # frame by hand gives identical numbers (verified), so this is a consistency fix,
        # not a correctness one.
        if subset is not None:
            data = data.loc[subset]

        # Stata's ivregress always includes a constant unless noconstant is
        # given, while linearmodels only includes one if the formula says so.
        # Add it here so the same formula produces the same parameters in both,
        # instead of silently returning one coefficient fewer.
        formula = _ensure_intercept(formula)

        try:
            parsed = _LMIV2SLS.from_formula(formula, data)
        except ValueError as exc:
            # linearmodels validates rank and instrument counts while parsing.
            # Re-raise a rank failure as westata's own diagnosis; an instrument
            # count failure is a user API mistake and keeps its ValueError type,
            # matching what the array constructor raises.
            if "full column rank" in str(exc):
                raise ConsistencyError(
                    f"ivregress: {exc}. A collinear regressor/instrument cannot "
                    "be estimated; drop the duplicate before building the model."
                ) from exc
            raise
        dependent = parsed.dependent.pandas
        exog = parsed.exog.pandas
        endog = parsed.endog.pandas
        instruments = parsed.instruments.pandas

        names = list(exog.columns) + list(endog.columns)

        # Weights are aligned to the ESTIMATION SAMPLE, not to ``data``: linearmodels
        # drops rows with missing values while parsing, so a weight vector indexed by the
        # original frame would be misaligned -- and misaligned weights do not raise, they
        # just weight the wrong observations. ``dependent.index`` is the kept sample.
        # (``weights``/``weight_type`` were popped above, before the keyword check.)
        if weights is not None and not isinstance(weights, str):
            if hasattr(weights, "loc") and hasattr(weights, "index"):
                weights = np.asarray(
                    weights.loc[dependent.index], dtype=float
                )
            else:
                values = np.asarray(weights, dtype=float).reshape(-1)
                if values.shape[0] != len(data) and values.shape[0] != len(dependent):
                    raise WeightError(
                        f"weights have {values.shape[0]} entries but the data have "
                        f"{len(data)} rows ({len(dependent)} after dropping missing "
                        "values)"
                    )
                weights = values

        model = cls(
            dependent.to_numpy(dtype=float).reshape(-1),
            exog.to_numpy(dtype=float) if exog.shape[1] else None,
            endog.to_numpy(dtype=float) if endog.shape[1] else None,
            instruments.to_numpy(dtype=float) if instruments.shape[1] else None,
            param_names=names,
            weights=weights,
            weight_type=weight_type,
        )
        #: ``ivregress`` reports ``e(b)`` in the order **endogenous then included
        #: exogenous**, with ``_cons`` last.  The fixtures prove it:
        #: ``iv_liml.json``'s cmdline is ``ivregress liml y x2 (x = z1 z2)`` and
        #: its names are ``['x', 'x2', '_cons']`` -- the endogenous ``x`` precedes
        #: the exogenous ``x2``.  This package builds the design as
        #: ``exog + endog``, so the two orders differ whenever both are present.
        model._formula_rhs_order = [
            str(name)
            for name in list(endog.columns) + list(exog.columns)
            if str(name).strip().lower() not in ("intercept", "intercept_", "_cons", "const")
        ]
        model._kept_index = list(dependent.index)
        #: ``e(depvar)`` (``ivregress.sthlp`` documents it like every estimation
        #: command); ``StataResults.e`` falls back to ``response_name``.
        try:
            model._response_name = str(list(dependent.columns)[0])
        except (IndexError, AttributeError):  # pragma: no cover - unnamed response
            pass
        # The three regressor blocks, named separately: ``estat firststage``
        # needs to know which columns are included exogenous regressors, which
        # are endogenous, and which are excluded instruments.
        model._exog_names = [str(c) for c in exog.columns]
        model._endog_names = [str(c) for c in endog.columns]
        model._instr_names = [str(c) for c in instruments.columns]
        return model

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
        level: float | None = None,
    ) -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        Parameters
        ----------
        cov_type:
            ``None``/``"nonrobust"``, ``"HC0"``, ``"HC1"`` or ``"cluster"``.
            ``None`` selects Stata's default for ``ivregress`` (unadjusted).
            ``"HC0"`` corresponds to Stata's ``vce(robust)`` and ``"HC1"`` to
            ``vce(robust) small``; ``"cluster"`` is CR0 unless ``ssc.adj`` is on
            (Stata's ``small``).  See the module docstring.
        cov_kwds:
            ``{"clusters": labels}`` for ``cov_type="cluster"``; must be empty
            otherwise.
        ssc:
            Stata small-sample settings; ``None`` uses
            ``SSC(adj=False, cluster_adj=True)``, i.e. ``ivregress`` without
            ``small``.
        """

        canonical = _canonical_cov_type(cov_type)
        # Stata's ivregress defaults differ from regress: the nonrobust VCE uses
        # s^2 = ESS/N (no ``small`` adjustment), so the default here is
        # SSC(adj=False) rather than the package-wide SSC(adj=True).
        settings = SSC(adj=False, cluster_adj=True) if ssc is None else resolve_ssc(ssc)
        clusters = self._prepare_cov_kwds(canonical, cov_kwds)

        y, X_exog, X_endog, Z_instr = self._y, self._X_exog, self._X_endog, self._Z_instr
        nobs = y.shape[0]
        X = np.column_stack([X_exog, X_endog])
        k_params = X.shape[1]

        # Both rank conditions are checked here so that a rank-deficient model
        # fails with westata's own diagnosis instead of a numpy LinAlgError from
        # deep inside the backend.
        self._check_rank(X, Z_instr)

        if self._uses_shared_vce:
            # Point estimates only; the covariance is rebuilt by compute_cov so
            # that the Stata small-sample factors live in exactly one place.
            ref = self._backend_fit("nonrobust", None, settings)
            params = np.asarray(ref.params, dtype=float)
            resid = y - X @ params
            cov, df_t = self._shared_vce(
                resid, canonical, clusters, settings, cov_kwds
            )
        else:
            # k-class/GMM: one fit yields both the estimates and the covariance
            # that compute_cov cannot express.
            ref = self._backend_fit(canonical, clusters, settings, cov_kwds)
            params = np.asarray(ref.params, dtype=float)
            resid = y - X @ params
            cov, df_t = self._backend_vce(ref, canonical, clusters, settings)

        scalars = self._stata_scalars(
            Y=y,
            X=X,
            params=params,
            resid=resid,
            cov=cov,
            ref=ref,
            clusters=clusters,
            canonical=canonical,
        )

        results = StataResults(
            params=params,
            param_names=self._param_names,
            cov=cov,
            resid=resid,
            fittedvalues=X @ params,
            nobs=int(nobs),
            df_resid=df_t,
            df_model=scalars["df_m"],
            dist="z",
            model_name=self._stata_command,
            cov_type=canonical,
            e=scalars,
            level=level,
        )
        #: The IV design pieces, retained so ``estat firststage`` can rebuild the
        #: auxiliary regressions.  Stata's ``e(insts)``/``e(instd)`` divide the
        #: regressor space into included exogenous regressors, endogenous
        #: regressors and excluded instruments; the estimator already holds all
        #: three, but only the combined ``X`` reached the result.
        results.iv_structure = {
            "exog": self._X_exog,
            "endog": self._X_endog,
            "instruments": self._Z_instr,
            "exog_names": [str(n) for n in self._exog_names],
            "endog_names": [str(n) for n in self._endog_names],
            "instrument_names": [str(n) for n in self._instr_names],
            "response": self._y,
        }
        #: ``e(b)`` order for ``stata_params``: endogenous regressors first, then
        #: the included exogenous ones (``iv_liml.json``: ``x`` before ``x2``).
        results._formula_rhs_order = list(self._formula_rhs_order or [])
        #: The estimation design, retained so ``predict`` can run.  Without it
        #: EVERY predict type raised ConsistencyError ("the design matrix was not
        #: retained") -- an audit of the full predict matrix showed the whole
        #: IV2SLS column failing, which no per-type test would have revealed
        #: because each one was "correctly" refusing.
        results.design_matrix = (X, list(self._param_names))
        #: ``e(depvar)``: the IV result builds its own StataResults, so the
        #: base-class fallback needs this attribute set here.
        _response = getattr(self, "_response_name", None)
        if _response:
            results.response_name = str(_response)
            results.e["depvar"] = str(_response)
        #: Cluster labels, when the VCE was clustered.  Retained so
        #: ``estat endogenous`` can rebuild the augmented ``regF`` regression with
        #: the same VCE -- after a clustered fit Stata reports ONLY ``regF``.
        results.cluster_labels = (
            None if clusters is None else np.asarray(clusters)
        )
        return results

    # -------------------------------------------------------------- internals
    def _backend_kwargs(
        self, canonical: str, clusters: np.ndarray | None
    ) -> dict[str, Any]:
        """Extra constructor keyword arguments for the linearmodels backend.

        ``ivregress`` has no estimator-selecting option here: 2SLS/LIML are
        fully determined by the data, so they pass nothing.  GMM additionally
        chooses a weight matrix, and *that* choice follows ``vce()`` -- see
        :class:`IVGMM`.
        """

        return {}

    def _backend_model(
        self, canonical: str = "nonrobust", clusters: np.ndarray | None = None
    ) -> Any:
        """Instantiate the linearmodels backend for these data."""

        exog = self._X_exog if self._X_exog.shape[1] else None
        endog = self._X_endog if self._X_endog.shape[1] else None
        instruments = self._Z_instr if self._Z_instr.shape[1] else None
        y = self._y.reshape(-1, 1)
        kwargs = self._backend_kwargs(canonical, clusters)
        if self._weights is not None:
            # linearmodels applies ``weights`` internally to the moment conditions, which
            # is the aweight/fweight semantics: minimise sum(w_i * e_i^2). The Stata
            # normalization (including pweight rescaling) already happened in
            # ``normalize_weights``, so pass the normalized vector rather than the raw one.
            kwargs["weights"] = self._weights.reshape(-1, 1)
        return self._backend_cls(y, exog, endog, instruments, **kwargs)

    def _check_rank(self, X: np.ndarray, Z_instr: np.ndarray) -> None:
        k_params = X.shape[1]
        instrument_matrix = np.column_stack([self._X_exog, Z_instr])
        if np.linalg.matrix_rank(X) < k_params:
            raise ConsistencyError(
                f"{self._stata_command}: the regressor matrix is rank deficient "
                f"(rank {np.linalg.matrix_rank(X)} < {k_params} columns); drop "
                "collinear regressors"
            )
        if np.linalg.matrix_rank(instrument_matrix) < instrument_matrix.shape[1]:
            raise ConsistencyError(
                f"{self._stata_command}: the instrument matrix is rank deficient "
                f"(rank {np.linalg.matrix_rank(instrument_matrix)} < "
                f"{instrument_matrix.shape[1]} columns); collinear instruments "
                "cannot identify the model"
            )
        if np.linalg.matrix_rank(instrument_matrix) < k_params:
            raise ConsistencyError(
                f"{self._stata_command}: the model is under-identified "
                f"({k_params} parameters but only "
                f"{np.linalg.matrix_rank(instrument_matrix)} identifying "
                "instruments incl. exogenous regressors)"
            )

    def _prepare_cov_kwds(
        self, canonical: str, cov_kwds: Mapping[str, Any] | None
    ) -> np.ndarray | None:
        """Validate ``cov_kwds`` and return aligned cluster labels (or ``None``)."""

        if cov_kwds is not None and not isinstance(cov_kwds, Mapping):
            raise TypeError(
                "cov_kwds must be a mapping (dict-like) of estimator keyword "
                f"arguments; got {type(cov_kwds).__name__}"
            )
        kwds = {} if cov_kwds is None else dict(cov_kwds)
        if canonical == "HAC":
            # ``newey`` needs an explicit bandwidth (``newey.sthlp:55``); the
            # validated value is stashed for ``_shared_vce`` to forward.
            if not kwds:
                raise ValueError(
                    "cov_type='HAC' requires cov_kwds['maxlags']; Stata's "
                    "newey takes lag() as a required option with no default"
                )
            unknown = set(kwds) - {"maxlags", "lags"}
            if unknown:
                raise ValueError(
                    "unexpected cov_kwds entries for cov_type='HAC': "
                    + ", ".join(sorted(unknown))
                )
            self._hac_kwds = kwds
            return None
        if canonical != "cluster":
            if kwds:
                raise ValueError(
                    f"cov_kwds was supplied but cov_type={canonical!r} takes no "
                    "estimator options; pass cov_kwds only with "
                    "cov_type='cluster'"
                )
            return None

        # All THREE spellings are accepted, matching every other estimator: ``groups``
        # (the OLS/ML documentation), ``clusters`` (natural for Stata's ``vce(cluster
        # clustvar)``) and ``cluster`` (what ``hdfe`` used). Each name is POPPED so the
        # "unexpected cov_kwds" check below still catches genuine typos.
        clusters = None
        for key in ("clusters", "groups", "cluster"):
            candidate = kwds.pop(key, None)
            if candidate is not None and clusters is None:
                clusters = candidate
        if clusters is None:
            raise ValueError(
                "cov_type='cluster' requires the cluster labels; pass them as "
                "cov_kwds={'clusters': labels} (aliases 'groups' and 'cluster' are also "
                "accepted)"
            )
        if kwds:
            raise ValueError(
                "unexpected cov_kwds entries for cov_type='cluster': "
                + ", ".join(sorted(kwds))
            )
        return _cluster_labels(clusters, self._y.shape[0], self._kept_index)

    def _shared_vce(
        self,
        resid: np.ndarray,
        canonical: str,
        clusters: np.ndarray | None,
        settings: SSC,
        cov_kwds: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Stata VCE through :func:`westata.vce.dispatch.compute_cov`.

        2SLS is a k-class estimator with ``kappa == 1``, so its bread is the
        projected cross product ``Xhat'Xhat`` and the whole covariance is the
        OLS-shaped sandwich that ``compute_cov`` already implements.
        """

        X = np.column_stack([self._X_exog, self._X_endog])
        Z = np.column_stack([self._X_exog, self._Z_instr])
        # THE PROJECTION MUST HAPPEN ON THE WEIGHTED DATA. sqrt(w)*Xhat = P_{sqrt(w)Z}
        # applied to sqrt(w)X, and that is not sqrt(w) times the unweighted Xhat unless
        # the projection is scale-invariant -- which it is not. Building Xhat first and
        # scaling afterwards left the slope standard errors 0.5% off while the intercept
        # matched exactly, a pattern that pointed straight at this line.
        if self._weights is not None:
            root = np.sqrt(self._weights).reshape(-1, 1)
            X = X * root
            Z = Z * root
        pinvz = np.linalg.pinv(Z)
        Xhat = Z @ (pinvz @ X)
        nobs, k_params = X.shape
        if np.linalg.matrix_rank(Xhat) < k_params:
            raise ConsistencyError(
                f"{self._stata_command}: the projected regressor matrix Xhat is "
                "rank deficient, so the model is not identified"
            )
        # ``cluster`` needs the labels, ``HAC`` the bandwidth.  Forward both from
        # the caller rather than rebuilding them here, so a HAC fit cannot
        # silently lose ``maxlags`` and fall back to another estimator.
        if canonical == "cluster":
            dispatch_kwds: dict[str, Any] | None = (
                None if clusters is None else {"groups": clusters}
            )
        elif canonical == "HAC":
            dispatch_kwds = dict(getattr(self, "_hac_kwds", None) or cov_kwds or {})
        else:
            dispatch_kwds = None
        # Two different Stata conventions are in play, and ``ssc.adj`` controls
        # only the first:
        #
        # * nonrobust: ``[R] ivregress`` uses s^2 = ESS/N by default; N-k only
        #   with the ``small`` option.
        # * HC1: the N/(N-K) factor is always part of HC1 (as for regress).
        #
        # compute_cov ties both to ``ssc.adj``, so pass a per-call SSC here with
        # adj=True for the robust variants and let ``df_resid`` carry only the
        # nonrobust denominator decision.
        robust_variant = canonical in ("HC0", "HC1", "HC2", "HC3", "cluster", "HAC")
        cov_ssc = settings
        if canonical == "HAC":
            # ``newey`` always applies N/(N-K): ``newey.sthlp:113-114`` says
            # lag(0) reproduces ``vce(robust)``, and that VCE carries the factor
            # regardless of ``ssc.adj``.  Without this branch the covariance came
            # out exactly (N-K)/N too small -- a clean 398/400 = 0.995 ratio that
            # reads like a kernel error rather than a missing scale factor.
            cov_ssc = SSC(
                adj=True,
                cluster_adj=settings.cluster_adj,
                df=settings.df,
            )
        elif robust_variant and not settings.adj:
            # An explicit HC1 with adj=False must still apply N/(N-K); HC0 must
            # not. Only re-enable the factor for HC1.
            cov_ssc = SSC(
                adj=(canonical == "HC1"),
                cluster_adj=settings.cluster_adj,
                df=settings.df,
            )
        denominator = float(nobs - k_params) if settings.adj else float(nobs)
        if self._weights is not None:
            # X/Z/Xhat are already sqrt(w)-transformed above (the projection needs the
            # weighted data), so only the residual still has to follow. sqrt(w)*e, not
            # w*e: e~'e~ must equal e'We for the nonrobust branch to give s^2 (X'WX)^-1.
            #
            # The weights do NOT rescale N: Stata keeps the sample size and the effective
            # sample size separately, which is why ``nobs``/``denominator`` stay as they are.
            resid = resid * np.sqrt(self._weights)
            return compute_cov(
                Xhat,
                resid,
                cov_type=canonical,
                cov_kwds=dispatch_kwds,
                ssc=cov_ssc,
                nobs=nobs,
                df_resid=denominator,
                estimate_model="ivregress",
            )
        return compute_cov(
            Xhat,
            resid,
            cov_type=canonical,
            cov_kwds=dispatch_kwds,
            ssc=cov_ssc,
            nobs=nobs,
            df_resid=denominator,
            estimate_model="ivregress",
        )

    def _backend_fit(
        self,
        canonical: str,
        clusters: np.ndarray | None,
        settings: SSC,
        cov_kwds: Mapping[str, Any] | None = None,
    ) -> Any:
        """Fit the backend with the covariance settings ``ssc`` asks for."""

        # ``[R] ivregress`` "Methods and formulas": s^2 = ESS/(N-k) only when the
        # ``small`` option is given; the default is s^2 = ESS/N.  linearmodels'
        # ``debiased`` flag means "divide by N-k" (and, for the clustered and
        # heteroskedastic VCEs, "apply the Stata small-sample factors"), so the
        # default must be debiased=False for every ivregress VCE -- including
        # ``vce(cluster)``, which is CR0 by default and only becomes CR1 with
        # ``small``.  Verified against StataNow 19.5 for 2SLS/LIML/GMM.
        if canonical == "nonrobust":
            debiased = bool(settings.adj)
        elif canonical == "HC0":
            debiased = False
        elif canonical == "HC1":
            debiased = True
        elif canonical == "HAC":
            # ``newey.ado:187`` multiplies by ``nobs/tdf`` UNCONDITIONALLY:
            #     mat `v' = (`xtxi'*`xtx'*`xtxi')*(`nobs'/`tdf')
            # There is no ``small`` option on ``newey``.  So HAC always applies
            # the factor, regardless of the IV default ``SSC(adj=False)`` that
            # ``ivregress`` uses for its other VCEs.
            #
            # Without this branch ``liml``/``gmm`` HAC agreed exactly with HC0 at
            # lag 0, while 2SLS HAC (through ``compute_cov``) carried the factor
            # -- an inconsistency between the two paths.
            debiased = True
        else:
            # ``cluster``: CR0 by default.  ``ssc.adj=True`` is the caller asking
            # for the ``small`` variant, which is exactly ``debiased=True``.
            debiased = bool(settings.adj)
        config: dict[str, Any] = {}
        if canonical == "cluster":
            n_groups = int(np.unique(np.asarray(clusters)).size)
            if n_groups < 2:
                # ConsistencyError for the same reason as ``vce/cluster.py``: the other
                # estimators already refused this with that class, so one user error had
                # two exception types. It is also a ValueError, so ``except ValueError``
                # callers are unaffected.
                raise ConsistencyError(
                    "cluster-robust covariance requires at least 2 groups; "
                    f"got {n_groups}"
                )
            config["clusters"] = clusters
        if canonical == "HAC":
            # LIML/GMM cannot use ``compute_cov`` (their bread is the k-class
            # ``(1-kappa)X'X + kappa X'P_Z X`` or GMM's weight matrix), but
            # linearmodels ships ``KernelCovariance`` -- a Bartlett-kernel
            # Newey-West sandwich that accepts ``kappa``, so it covers both.
            # Passing the kernel and bandwidth through is therefore enough; no
            # separate implementation is needed on this path.
            lags = (cov_kwds or {}).get("maxlags", (cov_kwds or {}).get("lags"))
            if lags is None:
                raise ValueError(
                    "cov_type='HAC' requires cov_kwds['maxlags']; Stata's "
                    "newey takes lag() as a required option with no default"
                )
            bandwidth = int(lags)
            if bandwidth < 0:
                raise ValueError(
                    f"lag() must be nonnegative (newey.ado:24-27); got {bandwidth}"
                )
            config["kernel"] = "bartlett"
            config["bandwidth"] = bandwidth
        return self._backend_model(canonical, clusters).fit(
            cov_type=_LM_COV_TYPES[canonical], debiased=debiased, **config
        )

    def _backend_vce(
        self,
        ref: Any,
        canonical: str,
        clusters: np.ndarray | None,
        settings: SSC,
    ) -> tuple[np.ndarray, float]:
        """k-class / GMM covariance taken from an already fitted backend.

        ``compute_cov`` cannot express the k-class bread
        ``(1-kappa)X'X + kappa X'P_Z X`` nor GMM's weight matrix, so this path
        declares the backend covariance explicitly (see the module docstring).
        ``ssc`` is still mapped in :meth:`_backend_fit` (``ssc.adj`` ->
        ``debiased``); ``ssc.cluster_adj=False`` additionally drops the
        ``G/(G-1)`` factor, leaving the ``(N-1)/(N-K)`` part that Stata's
        ``vce(cluster)`` also keeps.
        """

        cov = np.array(ref.cov, dtype=float, copy=True)
        if canonical == "cluster":
            n_groups = int(np.unique(np.asarray(clusters)).size)
            if settings.adj and not settings.cluster_adj:
                cov = cov / (n_groups / (n_groups - 1.0))
            df_t = float(n_groups - 1)
        else:
            df_t = float(self._y.shape[0] - self._X_endog.shape[1] - self._X_exog.shape[1])
        return cov, df_t

    def _stata_scalars(
        self,
        *,
        Y: np.ndarray,
        X: np.ndarray,
        params: np.ndarray,
        resid: np.ndarray,
        cov: np.ndarray,
        ref: Any,
        clusters: np.ndarray | None = None,
        canonical: str = "nonrobust",
    ) -> dict[str, Any]:
        """Build the ``e()`` scalars for the fitted model."""
        #: True for every robust VCE the package accepts; drives e(vcetype)/e(vce).
        _robust_vce = str(canonical).strip().lower() in {
            "robust", "hc0", "hc1", "hc2", "hc3", "hc"
        }

        nobs, k_params = X.shape
        # ``[R] ivregress`` leaves e(df_r) missing: the reported t/z inference is
        # normal-based, so there is no residual degrees of freedom to record.
        # The observable quantities use N directly:
        #   e(rmse) = sqrt(rss/N), e(r2_a) = 1 - (1-r2)(N-c)/N.
        df_r = float(nobs)
        rss = float(resid @ resid)
        # A WEIGHTED fit reports weighted sums, the way ``regress [aw=w]`` does: e(rss) is
        # sum(w*e^2), TSS is the weighted deviation sum, and e(N) is the EFFECTIVE sample
        # size (the weight sum for fweight, the row count otherwise). Using the unweighted
        # moments would leave the point estimates right and R^2/rmse wrong -- the same class
        # of error as building the VCE from untransformed data.
        effective_n = df_r
        if self._weights is not None and self._weight_spec is not None:
            effective_n = float(self._weight_spec.effective_n)
            rss = float(np.sum(self._weights * np.asarray(resid, dtype=float) ** 2))

        # Whether the regressor space contains a constant vector, determined
        # from the data (column space of X containing the ones vector) rather
        # than from a variable name.  Stata's e(df_m) and the centred R^2 both
        # hinge on this: with a constant, df_m = K - 1 and TSS is demeaned.
        is_constant = np.array([bool(np.ptp(X[:, j]) == 0.0) for j in range(k_params)])
        ones = np.ones((nobs, 1))
        has_constant = bool(
            np.linalg.matrix_rank(np.column_stack([X, ones])) == np.linalg.matrix_rank(X)
        )

        if self._weights is not None:
            # Weighted TSS: the weight sum of squared deviations from the WEIGHTED mean
            # (which is what Stata's `[aw=w]` summarise would report).
            weights = np.asarray(self._weights, dtype=float)
            if has_constant:
                weighted_mean = float(np.sum(weights * Y) / np.sum(weights))
                tss = float(np.sum(weights * (Y - weighted_mean) ** 2))
            else:
                tss = float(np.sum(weights * Y**2))
        elif has_constant:
            centred = Y - Y.mean()
            tss = float(centred @ centred)
        else:
            tss = float(Y @ Y)
        r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
        # ``[R] ivregress``: R2_a = 1 - (1 - R2)(N - c)/(N - k).  The N-k
        # denominator applies to R2_a even though it does not to the VCE: these
        # are separate conventions in the same manual page.
        r2_adj_denominator = float(effective_n - k_params)
        r2_adj = (
            1.0
            - ((effective_n - int(has_constant)) / r2_adj_denominator) * (1.0 - r2)
            if np.isfinite(r2) and r2_adj_denominator > 0.0
            else float("nan")
        )

        rmse = float(np.sqrt(rss / effective_n))
        # Gaussian likelihood at the reported estimates (Stata's e(ll) for
        # ivregress 2sls); sigma is the MLE sqrt(rss/n), as in westata.OLS.
        ll = float(-0.5 * nobs * (np.log(2.0 * np.pi) + np.log(rss / nobs) + 1.0))

        tested = np.array([not is_constant[j] for j in range(k_params)])
        df_m = int(tested.sum())
        wald = _quadratic_form(params[tested], cov[np.ix_(tested, tested)])
        fvalue = wald / df_m if df_m > 0 and np.isfinite(wald) else float("nan")

        f_first = self._first_stage_f(ref, nobs)

        e: dict[str, Any] = {
            "N": effective_n,
            # The row count, kept alongside because e(N) and the number of observations
            # differ once fweight expands the data -- ``regress [fw=w]`` reports
            # e(N)=sum(w) while the frame still has n rows.
            "N_obs": float(nobs),
            "df_m": df_m,
            "df_r": df_r,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "F": fvalue,
            "F_first": f_first,
            "chi2": wald,
            "ll": ll,
            "rank": k_params,
            # Sums of squares: Stata stores e(rss)/e(mss) for every ivregress
            # estimator (and e(tss) for 2sls) with rss + mss = tss, verified on
            # the 140x14 panel to 8.9e-16.
            "rss": rss,
            "mss": tss - rss,
            "tss": tss,
            "k_endog": int(self._X_endog.shape[1]),
            # Stata's banner and the two footer lines naming the variable roles:
            #
            #   Instrumental-variables 2SLS regression       Number of obs   =   100
            #   ...
            #   Endogenous: x2
            #   Exogenous:  x1 z
            #
            # Measured from ``ivregress 2sls y x1 (x2 = z)``.
            "title": _IV_TITLES.get(str(self._stata_command), ""),
            # ``ivregress gmm`` is the ONLY IV method that sets e(vcetype) -- measured:
            # 2sls and liml store "", while gmm stores "Robust" because its weight matrix
            # is estimated robustly. That macro is what draws the banner row above the
            # coefficient header, so without it the gmm table was missing that row.
            # ``e(vcetype)`` FOLLOWS THE VCE, NOT THE ESTIMATOR. The earlier note claimed
            # gmm was the only method to set it; live `display e(vcetype) e(vce)` refutes that:
            #
            #     ivregress 2sls ..., r     "Robust"  "robust"
            #     ivregress 2sls ...        ""        ""
            #     ivregress gmm ...         "Robust"  "robust"
            #
            # That macro draws the "Robust" banner over the coefficient header and lowercases
            # the "std. err." label, so an empty string suppressed both on a robust 2sls.
            "vcetype": "Robust" if _robust_vce else "",
            "vce": "robust" if _robust_vce else "",
            "endog": list(getattr(self, "_endog_names", [])),
            "exog": [
                name
                for name in [
                    *getattr(self, "_exog_names", []),
                    *getattr(self, "_instr_names", []),
                ]
                # The intercept is an exog column in the design but Stata does not name it
                # in the footer: "Exogenous:  x1 z", not "Exogenous:  _cons x1 z".
                if str(name).strip().lower() not in ("intercept", "_cons", "const")
            ],
        }
        e.update(self._extra_estimator_scalars(ref, canonical))
        if clusters is not None:
            # Stata stores e(N_clust) for every vce(cluster var) fit, including
            # the IV estimators (verified: G = 14 -> e(N_clust) = 14).
            #
            # A MULTIWAY spec arrives as an (n, dimensions) array. ``np.unique`` on that
            # raises ``TypeError: '<' not supported between 'str' and 'float'`` as soon as
            # the two label columns have different dtypes, and even when it does not, the
            # row count is not the number of clusters. Stata reports the SMALLEST
            # per-dimension group count, which is also the one that limits inference.
            array = np.asarray(clusters)
            if array.ndim == 2 and array.shape[1] > 1:
                e["N_clust"] = float(
                    min(
                        np.unique(array[:, index]).size
                        for index in range(array.shape[1])
                    )
                )
            else:
                e["N_clust"] = float(np.unique(array).size)
        if self._stores_rank_mb and canonical != "nonrobust":
            # ``ivregress 2sls, vce(robust|cluster)`` additionally stores
            # e(rank_mb), which equals e(rank) = K.  LIML and GMM never store it,
            # and neither does any estimator with ``vce(unadjusted)`` (verified
            # across vce() options on StataNow 19.5).
            e["rank_mb"] = float(k_params)
        return e

    def _extra_estimator_scalars(self, ref: Any, canonical: str) -> dict[str, Any]:
        """``e()`` scalars Stata stores for this estimator only.

        ``ivregress 2sls``/``liml`` store ``e(kappa)`` (1 for 2SLS, the LIML
        kappa otherwise) and ``e(iterations) = 0``; ``ivregress gmm`` stores
        ``e(J)`` and ``e(iterations)`` instead and stores no ``kappa`` at all.
        """

        out: dict[str, Any] = {"iterations": 0.0}
        kappa = getattr(ref, "kappa", None)
        if kappa is not None:
            out["kappa"] = float(kappa)
        return out

    def _partial_first_stage_f(self) -> float:
        """Stata's partial first-stage F for each endogenous regressor; smallest returned.

        For endogenous regressor ``j``: regress it on the INCLUDED exogenous variables plus a
        constant (the restricted model), then again with the excluded instruments added (the
        full model). ``F = ((rss_r - rss_f) / n_excluded) / (rss_f / (n - k_full))``.

        The denominator uses the FULL design's rank, which is what puts Stata at 21 rather
        than 20 here.
        """

        y_endog = np.asarray(self._X_endog, dtype=float)
        included = np.asarray(self._X_exog, dtype=float)
        excluded = np.asarray(self._Z_instr, dtype=float)
        if excluded.shape[1] == 0:
            return float("nan")
        ones = np.ones((y_endog.shape[0], 1))
        design_r = np.column_stack([ones, included])
        design_f = np.column_stack([ones, included, excluded])
        n = design_r.shape[0]
        k_full = design_f.shape[1]
        n_excluded = excluded.shape[1]
        df_denom = n - k_full
        if df_denom <= 0:
            return float("nan")
        statistics: list[float] = []
        for column in range(y_endog.shape[1]):
            target = y_endog[:, column]
            coefficients_r, *_ = np.linalg.lstsq(design_r, target, rcond=None)
            rss_r = float(((target - design_r @ coefficients_r) ** 2).sum())
            coefficients_f, *_ = np.linalg.lstsq(design_f, target, rcond=None)
            rss_f = float(((target - design_f @ coefficients_f) ** 2).sum())
            if rss_f <= 0.0:
                continue
            statistics.append((rss_r - rss_f) / n_excluded / (rss_f / df_denom))
        return float(min(statistics)) if statistics else float("nan")

    def _first_stage_f(self, ref: Any, nobs: int) -> float:
        """Smallest partial first-stage F over the endogenous regressors.

        Computed here rather than read from ``linearmodels``' ``f.stat``: that reports
        ``F(1,20) = 17967.770`` where Stata prints ``F(1,21) = 18866.2`` on the same data --
        one degree of freedom fewer, and a statistic 5% apart.

        Stata's ``estat firststage`` reports the PARTIAL F: the excluded instruments are
        tested jointly in a first-stage regression that already contains the included
        exogenous regressors, with ``df_denom = n - k_full`` over the FULL first-stage
        design. Measured on ``ivregress 2sls y l (k = z)`` (n = 24, full design
        [const, z, l] so df = 21):

            (rss_restricted - rss_full) / 1  /  (rss_full / 21) = 18866.1586

        matching Stata's 18866.2.
        """

        if self._X_endog.shape[1] == 0:
            return float("nan")
        try:
            f_first = float(self._partial_first_stage_f())
        except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            # Fall back to the backend's number, which at least signals the right order of
            # magnitude, rather than failing the whole fit.
            try:
                diagnostics = ref.first_stage.diagnostics
                f_stat = np.asarray(diagnostics["f.stat"], dtype=float)
                f_first = float(np.nanmin(f_stat)) if f_stat.size else float("nan")
            except (AttributeError, KeyError, TypeError, ValueError):
                return float("nan")
        if np.isfinite(f_first) and f_first < _WEAK_INSTRUMENT_F:
            warnings.warn(
                "weak instruments: the smallest first-stage F statistic is "
                f"{f_first:.3g} (< {_WEAK_INSTRUMENT_F:g}); 2SLS/LIML/GMM point "
                "estimates are biased towards OLS in this case",
                UserWarning,
                stacklevel=2,
            )
        return f_first


class IVLIML(IV2SLS):
    """Limited-information maximum likelihood (Stata ``ivregress liml``).

    Uses ``linearmodels.iv.IVLIML`` (k-class with the LIML ``kappa``).  The
    point estimates are identical to 2SLS when the model is exactly identified
    and differ otherwise, which is the expected LIML behaviour.

    ``cov_type="HC2"``/``"HC3"`` are rejected; see :class:`IV2SLS`.
    """

    _backend_cls = _LMIVLIML
    _stata_command = "ivregress liml"
    _uses_shared_vce = False
    _stores_rank_mb = False


class IVGMM(IV2SLS):
    """Generalized method of moments (Stata ``ivregress gmm``).

    ``ivregress gmm`` has *two* choices tied to ``vce()``, not one:

    * the covariance of ``e(b)`` (as for every IV estimator), and
    * the GMM weight matrix ``W``, which ``ivregress`` sets from the same
      moment covariance the ``vce()`` implies.

    So ``ivregress gmm, vce(cluster gid)`` clusters ``W`` too, and the *point
    estimates* move (3.58e-2 on the 140x14 panel of the retest).  westata
    therefore picks the backend weight matrix from ``cov_type``: the default
    heteroskedastic ("robust") one, or ``weight_type="clustered"`` when
    ``cov_type="cluster"``.  Verified against StataNow 19.5: coefficients and
    ``e(V)`` both agree to <= 5e-14 (140x14 and 240x24 panels, 2 and 3
    instruments).

    The covariance is the GMM weight-matrix sandwich from the backend; it
    cannot be expressed through :func:`westata.vce.dispatch.compute_cov`, which
    assumes an OLS-shaped bread.
    """

    _backend_cls = _LMIVGMM
    _stata_command = "ivregress gmm"
    _uses_shared_vce = False
    _stores_rank_mb = False

    def _backend_kwargs(
        self, canonical: str, clusters: np.ndarray | None
    ) -> dict[str, Any]:
        """Pick the GMM weight matrix that matches ``vce()``.

        Stata's ``ivregress gmm`` documents ``W`` as ``S^-1`` where ``S`` is the
        moment covariance implied by the requested ``vce()``; with
        ``vce(cluster g)`` that is the one-way clustered covariance.  Passing
        nothing (linearmodels' default ``weight_type="robust"``) reproduces the
        heteroskedastic default and ``vce(robust)``.
        """

        if canonical == "cluster":
            if clusters is None:  # pragma: no cover - guarded by _prepare_cov_kwds
                raise ValueError(
                    "cov_type='cluster' requires cluster labels for the GMM "
                    "weight matrix as well as the covariance"
                )
            return {"weight_type": "clustered", "clusters": clusters}
        return {}

    def _stata_two_step_covariance(self) -> np.ndarray:
        """Two-step GMM covariance exactly as ``[R] ivregress`` defines it.

        The defining feature is that the two weight matrices use DIFFERENT residual sets:

            W  from the initial 2SLS residuals
            S  from the GMM residuals

        A single sandwich construction -- one residual set for both -- produces standard errors
        0.2% to 21% away from Stata's while the point estimates agree to 1e-15, which is why
        this went unnoticed for several batches.
        """

        design = np.column_stack(
            [self._X_exog, self._X_endog]
            if self._X_exog.shape[1]
            else [self._X_endog]
        )
        instruments = np.column_stack(
            [self._X_exog, self._Z_instr]
            if self._X_exog.shape[1]
            else [self._Z_instr]
        )
        response = self._y.reshape(-1)
        n = response.size

        # Step 1: 2SLS, whose residuals build W.
        projection = (
            instruments @ np.linalg.pinv(instruments.T @ instruments) @ instruments.T
        )
        beta_2sls = np.linalg.pinv(design.T @ projection @ design) @ (
            design.T @ projection @ response
        )
        residual_2sls = response - design @ beta_2sls
        moments_2sls = instruments * residual_2sls[:, None]
        weight = np.linalg.pinv(moments_2sls.T @ moments_2sls / n)

        # Step 2: the GMM estimate, whose residuals build S.
        cross = design.T @ instruments
        bread = np.linalg.pinv(cross @ weight @ cross.T)
        beta_gmm = bread @ (cross @ weight @ (instruments.T @ response))
        residual_gmm = response - design @ beta_gmm
        moments_gmm = instruments * residual_gmm[:, None]
        score = moments_gmm.T @ moments_gmm / n

        return n * bread @ (cross @ weight @ score @ weight @ cross.T) @ bread

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Any = None,
        ssc: Any = None,
        **kwargs: Any,
    ) -> StataResults:
        """Fit, using Stata's two-step GMM variance unless the caller asked for another.

        The distinction has to be made HERE, on what the caller passed. ``_canonical_cov_type``
        maps ``None`` to a default NAME, so testing for None downstream never fires -- the
        earlier attempt did exactly that, left the variance unchanged, and still passed every
        test because the coefficients were already exact.
        """

        self._explicit_cov_type = cov_type is not None
        return super().fit(cov_type, cov_kwds, ssc, **kwargs)

    def _backend_vce(self, ref: Any, *args: Any, **kwargs: Any) -> Any:
        """Covariance for a GMM fit: Stata's two-step form by default, the requested one if asked.

        An explicit ``cov_type`` still wins. A caller asking for HC0, HC1 or a cluster variance
        has asked for that estimator, and overriding it silently would be the wrong kind of
        conformance.
        """

        result = super()._backend_vce(ref, *args, **kwargs)
        if getattr(self, "_explicit_cov_type", False):
            return result
        _, df_t = result
        return self._stata_two_step_covariance(), df_t

    def _extra_estimator_scalars(self, ref: Any, canonical: str) -> dict[str, Any]:
        """``e(J)`` and ``e(iterations)``; ``ivregress gmm`` stores no ``e(kappa)``.

        Stata reports ``e(iterations) = 1`` for its two-step GMM and the Hansen
        J statistic in ``e(J)``.  Verified on the 140x14 panel: default and
        ``vce(cluster gid)`` give J = 0.8351609988436951 and 1.1097180436518659,
        both matched by the backend's own J at <= 3e-14.
        """

        out: dict[str, Any] = {"iterations": 1.0}
        try:
            out["J"] = float(np.asarray(ref.j_stat.stat))
        except (AttributeError, TypeError, ValueError):
            pass
        return out
