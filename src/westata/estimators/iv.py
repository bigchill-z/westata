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
* Weighted IV is out of scope for P1: passing ``weights``/``weight_type``
  raises :class:`westata.errors.UnsupportedError`.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS as _LMIV2SLS
from linearmodels.iv import IVGMM as _LMIVGMM
from linearmodels.iv import IVLIML as _LMIVLIML

from westata.errors import ConsistencyError, UnsupportedError
from westata.protocols import SSC
from westata.results.base import StataResults
from westata.vce.dispatch import SUPPORTED_COV_TYPES, compute_cov
from westata.vce.ssc import resolve_ssc

__all__ = ["IV2SLS", "IVLIML", "IVGMM"]

#: ``cov_type`` values accepted by the IV estimators.  ``HC2``/``HC3`` are in
#: :data:`westata.vce.dispatch.SUPPORTED_COV_TYPES` for OLS but have no IV
#: analogue, so they are rejected here (see the module docstring).
_IV_COV_TYPES = tuple(t for t in SUPPORTED_COV_TYPES if t not in ("HC2", "HC3"))

#: linearmodels covariance names, verified against linearmodels 7.1
#: (``COVARIANCE_ESTIMATORS`` in ``linearmodels/iv/model.py``).
_LM_COV_TYPES = {
    "nonrobust": "unadjusted",
    "HC0": "robust",
    "HC1": "robust",
    "cluster": "clustered",
}

_WEAK_INSTRUMENT_F = 10.0

_WEIGHT_KEYWORDS = frozenset(
    {"weights", "weight_type", "fweights", "aweights", "pweights", "iweights"}
)


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
                f"IV estimation does not support weights in P1: {key!r} was "
                "passed, but 'ivregress' with weights is out of scope. Remove "
                "the weight and refit; weights are never silently ignored."
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
) -> np.ndarray:
    """Return one-dimensional cluster labels aligned with the estimation sample."""

    if isinstance(clusters, pd.Series) and clusters.shape[0] != nobs and kept_index is not None:
        # The usual case: the user passes a column of the *original* frame while
        # the estimator dropped rows with missing values.  Align by index
        # instead of guessing positions.
        if clusters.index.is_unique and set(kept_index).issubset(set(clusters.index)):
            clusters = clusters.loc[list(kept_index)]
    values = np.asarray(clusters.squeeze() if hasattr(clusters, "squeeze") else clusters)
    if values.ndim == 2 and values.shape[1] > 1:
        raise UnsupportedError(
            "two-way clustering is not implemented for IV in P1; pass a single "
            "cluster variable as cov_kwds={'clusters': ...}"
        )
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
    Weights are out of scope for P1.  ``weights``/``weight_type`` are accepted
    only so that they can be rejected with
    :class:`westata.errors.UnsupportedError`; they are never silently ignored.
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
        self._X_exog = X_exog
        self._X_endog = X_endog
        self._Z_instr = Z_instr
        self._param_names = names
        self._kept_index: list[Any] | None = None

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

        _reject_unsupported_kwargs(kwargs)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

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
        model = cls(
            dependent.to_numpy(dtype=float).reshape(-1),
            exog.to_numpy(dtype=float) if exog.shape[1] else None,
            endog.to_numpy(dtype=float) if endog.shape[1] else None,
            instruments.to_numpy(dtype=float) if instruments.shape[1] else None,
            param_names=names,
        )
        model._kept_index = list(dependent.index)
        return model

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
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
            cov, df_t = self._shared_vce(resid, canonical, clusters, settings)
        else:
            # k-class/GMM: one fit yields both the estimates and the covariance
            # that compute_cov cannot express.
            ref = self._backend_fit(canonical, clusters, settings)
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

        return StataResults(
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
        )

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
        return self._backend_cls(
            y, exog, endog, instruments, **self._backend_kwargs(canonical, clusters)
        )

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
        if canonical != "cluster":
            if kwds:
                raise ValueError(
                    f"cov_kwds was supplied but cov_type={canonical!r} takes no "
                    "estimator options; pass cov_kwds only with "
                    "cov_type='cluster'"
                )
            return None

        clusters = kwds.pop("clusters", None)
        if clusters is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['clusters']; pass the "
                "cluster labels as cov_kwds={'clusters': labels}"
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
    ) -> tuple[np.ndarray, float]:
        """Stata VCE through :func:`westata.vce.dispatch.compute_cov`.

        2SLS is a k-class estimator with ``kappa == 1``, so its bread is the
        projected cross product ``Xhat'Xhat`` and the whole covariance is the
        OLS-shaped sandwich that ``compute_cov`` already implements.
        """

        X = np.column_stack([self._X_exog, self._X_endog])
        Z = np.column_stack([self._X_exog, self._Z_instr])
        pinvz = np.linalg.pinv(Z)
        Xhat = Z @ (pinvz @ X)
        nobs, k_params = X.shape
        if np.linalg.matrix_rank(Xhat) < k_params:
            raise ConsistencyError(
                f"{self._stata_command}: the projected regressor matrix Xhat is "
                "rank deficient, so the model is not identified"
            )
        cov_kwds = None if clusters is None else {"groups": clusters}
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
        robust_variant = canonical in ("HC0", "HC1", "HC2", "HC3", "cluster")
        cov_ssc = settings
        if robust_variant and not settings.adj:
            # An explicit HC1 with adj=False must still apply N/(N-K); HC0 must
            # not. Only re-enable the factor for HC1.
            cov_ssc = SSC(
                adj=(canonical == "HC1"),
                cluster_adj=settings.cluster_adj,
                df=settings.df,
            )
        denominator = float(nobs - k_params) if settings.adj else float(nobs)
        return compute_cov(
            Xhat,
            resid,
            cov_type=canonical,
            cov_kwds=cov_kwds,
            ssc=cov_ssc,
            nobs=nobs,
            df_resid=denominator,
        )

    def _backend_fit(
        self,
        canonical: str,
        clusters: np.ndarray | None,
        settings: SSC,
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
        else:
            # ``cluster``: CR0 by default.  ``ssc.adj=True`` is the caller asking
            # for the ``small`` variant, which is exactly ``debiased=True``.
            debiased = bool(settings.adj)
        config: dict[str, Any] = {}
        if canonical == "cluster":
            n_groups = int(np.unique(np.asarray(clusters)).size)
            if n_groups < 2:
                raise ValueError(
                    "cluster-robust covariance requires at least 2 groups; "
                    f"got {n_groups}"
                )
            config["clusters"] = clusters
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

        nobs, k_params = X.shape
        # ``[R] ivregress`` leaves e(df_r) missing: the reported t/z inference is
        # normal-based, so there is no residual degrees of freedom to record.
        # The observable quantities use N directly:
        #   e(rmse) = sqrt(rss/N), e(r2_a) = 1 - (1-r2)(N-c)/N.
        df_r = float(nobs)
        rss = float(resid @ resid)

        # Whether the regressor space contains a constant vector, determined
        # from the data (column space of X containing the ones vector) rather
        # than from a variable name.  Stata's e(df_m) and the centred R^2 both
        # hinge on this: with a constant, df_m = K - 1 and TSS is demeaned.
        is_constant = np.array([bool(np.ptp(X[:, j]) == 0.0) for j in range(k_params)])
        ones = np.ones((nobs, 1))
        has_constant = bool(
            np.linalg.matrix_rank(np.column_stack([X, ones])) == np.linalg.matrix_rank(X)
        )

        if has_constant:
            centred = Y - Y.mean()
            tss = float(centred @ centred)
        else:
            tss = float(Y @ Y)
        r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
        # ``[R] ivregress``: R2_a = 1 - (1 - R2)(N - c)/(N - k).  The N-k
        # denominator applies to R2_a even though it does not to the VCE: these
        # are separate conventions in the same manual page.
        r2_adj_denominator = float(nobs - k_params)
        r2_adj = (
            1.0 - ((nobs - int(has_constant)) / r2_adj_denominator) * (1.0 - r2)
            if np.isfinite(r2) and r2_adj_denominator > 0.0
            else float("nan")
        )

        rmse = float(np.sqrt(rss / df_r))
        # Gaussian likelihood at the reported estimates (Stata's e(ll) for
        # ivregress 2sls); sigma is the MLE sqrt(rss/n), as in westata.OLS.
        ll = float(-0.5 * nobs * (np.log(2.0 * np.pi) + np.log(rss / nobs) + 1.0))

        tested = np.array([not is_constant[j] for j in range(k_params)])
        df_m = int(tested.sum())
        wald = _quadratic_form(params[tested], cov[np.ix_(tested, tested)])
        fvalue = wald / df_m if df_m > 0 and np.isfinite(wald) else float("nan")

        f_first = self._first_stage_f(ref, nobs)

        e: dict[str, Any] = {
            "N": nobs,
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
        }
        e.update(self._extra_estimator_scalars(ref, canonical))
        if clusters is not None:
            # Stata stores e(N_clust) for every vce(cluster var) fit, including
            # the IV estimators (verified: G = 14 -> e(N_clust) = 14).
            e["N_clust"] = float(np.unique(np.asarray(clusters)).size)
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

    def _first_stage_f(self, ref: Any, nobs: int) -> float:
        """Smallest first-stage F over the endogenous regressors (NaN if none)."""

        if self._X_endog.shape[1] == 0:
            return float("nan")
        try:
            diagnostics = ref.first_stage.diagnostics
            f_stat = np.asarray(diagnostics["f.stat"], dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            return float("nan")
        if f_stat.size == 0:
            return float("nan")
        f_first = float(np.nanmin(f_stat))
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
