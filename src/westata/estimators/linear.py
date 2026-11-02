"""Ordinary least squares, targeting Stata's ``regress``.

The estimation code only computes the point estimates and passes the residual
vector to :mod:`westata.vce`, which owns every small-sample correction.  This
keeps Stata conformance in one place instead of duplicating it per model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from westata.data.weights import normalize_weights
from westata.errors import ConsistencyError, UnsupportedError, WeightError
from westata.formula.factor import build_design
from westata.protocols import SSC
from westata.results.base import StataResults
from westata.vce.cluster import encode_groups
from westata.vce.dispatch import compute_cov


def _rhs_variable_order(formula: str) -> list[str]:
    """Right-hand-side variable names, in the order the formula writes them.

    ``"mpg ~ weight + foreign"`` returns ``["weight", "foreign"]``.  Only plain
    names are extracted: ``C(g)``, ``I(x**2)``, ``np.log(x)`` and interaction
    terms are skipped, because there is no Stata-side name to match them to and
    guessing one would be worse than leaving them out.

    Stata keeps the written order in ``e(b)`` and appends ``_cons``; patsy
    alphabetises, which is why the order has to be captured here rather than read
    off the design matrix.
    """

    if "~" not in formula:
        return []
    rhs = formula.split("~", 1)[1]
    # Drop the "0"/"1" intercept markers and split on +.
    order: list[str] = []
    for raw in rhs.replace("-", "+").split("+"):
        token = raw.strip()
        if not token or token in ("0", "1"):
            continue
        # Only bare identifiers are usable; anything with parentheses, brackets
        # or operators has a different (or no) Stata name.
        if not token.replace("_", "").isalnum():
            continue
        order.append(token)
    # De-duplicate while keeping order: ``x + x`` is one term.
    seen: set[str] = set()
    unique: list[str] = []
    for token in order:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _stata_vce_label(cov_type: Any, *, ml: bool = False) -> str:
    """Map this package's ``cov_type`` onto Stata's ``e(vce)`` spelling (LOWERCASE).

    ``regress.ado:48-78`` sets the macro from the ``vce()`` request, defaulting to ``ols``::

        local vce = cond("`r(vce)'" != "", "`r(vce)'", "ols")
        local vce cluster

    The internal names differ -- this package uses statsmodels' ``HC1``/``nonrobust`` --
    so the translation is explicit rather than a passthrough, which would put a token
    like ``nonrobust`` where Stata prints ``ols``.

    ``ml=True`` selects the MAXIMUM-LIKELIHOOD default instead: the ML estimators report
    ``oim`` (observed information matrix), not ``ols``.  ``arch.ado:1470-1474`` shows the
    pair together::

        if "`e(vcetype)'" == "Robust" { ... }
        else { ... est local vcetype "OIM" }
    """

    normalized = str(cov_type or "").strip().lower()
    if normalized == "nonrobust":
        return "oim" if ml else "ols"
    mapping = {
        "ols": "ols",
        "oim": "oim",
        "hc0": "hc0",
        # Stata has NO ``vce(hc1)``: the option is ``vce(robust)``, and that is the
        # estimator this package calls HC1. Verified: ``regress, vce(hc1)`` is rejected
        # outright, ``vce(robust)`` sets e(vce)="robust", while ``vce(hc2)`` sets
        # e(vce)="hc2" and e(vcetype)="Robust HC2". Reporting "hc1" put a token in
        # e(vce) that Stata can never produce.
        "hc1": "robust",
        "hc2": "hc2",
        "hc3": "hc3",
        "robust": "robust",
        "cluster": "cluster",
        "hac": "hac",
        "newey": "hac",
    }
    return mapping.get(normalized, normalized)


def _stata_vcetype_label(cov_type: Any, *, ml: bool = False) -> str:
    """Stata's ``e(vcetype)`` -- the DISPLAY title, capitalised and sometimes two words.

    This is a different string from ``e(vce)``, and the difference is load-bearing in
    Stata's own code: ``arch.ado:1470`` and ``archlm.ado:16`` both branch on

        if "`e(vcetype)'" == "Robust" { ... }

    while ``e(vce)`` holds the lowercase option token.  The non-robust case differs by
    family: ``ols`` for linear least squares, ``OIM`` for maximum likelihood
    (``arch.ado:1474``/``arima.ado:741``: ``est local vcetype "OIM"``).

    The HC forms are TWO WORDS, taken verbatim from ``didregress.ado:477`` and
    ``nl.ado:1115-1119``::

        ereturn local vcetype "Robust HC2"

    so a naive ``token.capitalize()`` produces ``"Hc2"`` and would not match.
    """

    token = _stata_vce_label(cov_type, ml=ml)
    if token == "ols":
        # Stata leaves e(vcetype) EMPTY for an ordinary least-squares fit: measured with
        # ``regress y s`` -> e(vce)="ols", e(vcetype)="". Returning "Ols" invented a value
        # Stata never stores, and it is the string Stata's own code compares against
        # ("Robust") when choosing a display path.
        return ""
    if token == "oim":
        return "OIM"
    if token in ("hc0", "hc1", "hc2", "hc3"):
        return f"Robust {token.upper()}"
    if token == "cluster":
        # ``vce(cluster g)`` stores e(vcetype)="Robust", NOT "Cluster". Measured:
        #
        #   regress y s, vce(cluster g)
        #     e(vce)="cluster"  e(vcetype)="Robust"
        #
        # The macro names the ESTIMATOR FAMILY (which table header to draw), not the
        # option: the clustered table is labelled "Robust" above the column header, and
        # the cluster count appears on its own line instead.
        return "Robust"
    return token[:1].upper() + token[1:]


def _cluster_dimension_columns(groups: Any) -> list[Any]:
    """Split cluster labels into per-dimension vectors (one entry for a one-way spec).

    Mirrors ``westata.vce.dispatch._cluster_dimensions`` for the reporting side: a
    two-column DataFrame or an ``(n, 2)`` array is multiway, anything else is a single
    dimension.
    """

    if isinstance(groups, pd.DataFrame):
        if groups.shape[1] >= 2:
            return [groups.iloc[:, index].to_numpy() for index in range(groups.shape[1])]
        return [groups.iloc[:, 0].to_numpy()]
    array = np.asarray(groups)
    if array.ndim == 2 and array.shape[1] >= 2:
        return [array[:, index] for index in range(array.shape[1])]
    return [array]


def _cluster_dimension_labels(groups: Any, n_dimensions: int) -> list[str]:
    """Names for the clustering dimensions, as ``clustertable`` prints them.

    A DataFrame carries its column names and those are what Stata shows (``a``, ``b``).
    A bare array carries none, so positional placeholders are used rather than inventing
    names that would look authoritative but be wrong.
    """

    if isinstance(groups, pd.DataFrame) and groups.shape[1] >= n_dimensions:
        return [str(name) for name in groups.columns[:n_dimensions]]
    # A Series knows its own name -- ``df["g"]`` carries "g". Using it lets the clustered
    # output say "in g", as Stata does; falling back to a placeholder printed "in group",
    # which is a name the caller never wrote.
    if isinstance(groups, pd.Series) and groups.name and n_dimensions == 1:
        return [str(groups.name)]
    if n_dimensions == 1:
        return ["group"]
    return [f"dim{index + 1}" for index in range(n_dimensions)]


def _resolve_weight_column(weights: Any, data: pd.DataFrame) -> Any:
    """Turn ``weights="colname"`` into the column's values; pass anything else through.

    ``statsmodels.formula.api`` accepts a column NAME here, and that is not a convenience
    detail: it is what makes its missing-value handling transparent, because the weights
    become one more column of the frame and are filtered with everything else. An array
    forces the caller to replicate the row-dropping themselves.

    A string that is NOT a column is an error rather than a silent array of one element --
    guessing would produce a confusing length failure later.
    """

    if isinstance(weights, str):
        if weights not in data.columns:
            raise WeightError(
                f"weights={weights!r} looks like a column name but the data have no such "
                f"column; available: {list(data.columns)[:12]}"
            )
        return data[weights].to_numpy(dtype=float)
    return weights


def _slice_weights_to_sample(weights: Any, kept_index: Any, n_rows: int) -> Any:
    """Restrict a weight vector to the rows the formula kept.

    Accepts either a full-length vector aligned with ``data`` (the API's documented form)
    or one already restricted to the estimation sample, so a caller who pre-sliced by hand
    keeps working.

    A length that is neither is refused with BOTH numbers and the reason, because the
    original error ("weights have 722 entries but the data have 675 rows") described a
    symptom while pointing at the caller's data.
    """

    values = np.asarray(weights, dtype=float).reshape(-1)
    kept = np.asarray(kept_index, dtype=int)
    if values.shape[0] == kept.shape[0]:
        return values
    if values.shape[0] == n_rows:
        return values[kept]
    raise WeightError(
        f"weights have {values.shape[0]} entries; expected {n_rows} (one per row of the "
        f"data) or {kept.shape[0]} (one per estimation-sample row). The formula drops rows "
        "with missing values, which is why the estimation sample can be shorter than the "
        "frame."
    )


def _wald_f(params: np.ndarray, cov: np.ndarray, tested: Sequence[int]) -> float:
    """Return ``b' V^-1 b / q`` over the ``tested`` coefficients.

    This is Stata's ``e(F)`` for ``regress``: the Wald statistic built from the
    covariance matrix that the model actually reports, divided by the number of
    tested coefficients (``[R] regress``, "Methods and formulas").  It reduces
    to the textbook ``(R2/q)/((1-R2)/(N-K))`` only when the covariance is the
    nonrobust one; under ``vce(robust)``/``vce(cluster)`` Stata reports the
    robust Wald version instead, so the classic formula would be the wrong
    number there.
    """

    idx = np.asarray(list(tested), dtype=int)
    b = np.asarray(params, dtype=float)[idx]
    V = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
    df_m = int(idx.size)
    if df_m == 0 or b.size == 0:
        return float("nan")
    try:
        return float(b @ np.linalg.solve(V, b) / df_m)
    except np.linalg.LinAlgError:
        return float("nan")


class OLS:
    """Ordinary least squares (Stata ``regress``).

    Parameters
    ----------
    endog:
        Dependent variable, length ``n``.
    exog:
        Design matrix, ``n x k``.  An intercept must be included explicitly,
        matching statsmodels semantics.
    weights:
        Optional Stata-weighted observation weights.
    weight_type:
        One of ``"fweight"``, ``"aweight"``, ``"pweight"``, ``"iweight"``.
        Required whenever ``weights`` is given; Stata never guesses the type.
        See :mod:`westata.data.weights` and Stata ``[U] 20.24``.
    """

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        weights: Any = None,
        weight_type: str | None = None,
        param_names: Sequence[str] | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).ravel()
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        self._param_names = (
            list(param_names) if param_names is not None else [f"x{i}" for i in range(X.shape[1])]
        )
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"endog has {self._y.shape[0]} rows but exog has {self._X.shape[0]} rows"
            )
        self._weight_spec = None
        #: Raw (un-normalised) ``sum(w)``: Stata's ``e(sum_w)``.  ``aweight`` and
        #: ``pweight`` are normalised to sum to ``N`` for estimation, so the
        #: normalized vector cannot be used to recover this scalar.
        self._raw_weight_sum = float(self._X.shape[0])
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None
        #: Response column name (``e(depvar)``), filled by ``from_formula``.
        self._response_name: str | None = None
        if weights is not None:
            if weight_type is None:
                raise WeightError(
                    "weights were supplied without weight_type; specify one of "
                    "'fweight', 'aweight', 'pweight', 'iweight' (Stata never "
                    "guesses the weight type)"
                )
            spec = normalize_weights(weights, weight_type)
            if spec.weights.shape[0] != self._X.shape[0]:
                raise WeightError(
                    f"weights have {spec.weights.shape[0]} entries but the data "
                    f"have {self._X.shape[0]} rows"
                )
            self._weight_spec = spec
            self._raw_weight_sum = float(np.asarray(weights, dtype=float).sum())
        elif weight_type is not None:
            raise WeightError("weight_type was supplied without weights")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        weights: Any = None,
        weight_type: str | None = None,
        subset: Any = None,
    ) -> "OLS":
        """Build a model from a patsy-style formula, e.g. ``"y ~ x + C(g)"``.

        **Rows with missing values are dropped.** A formula can only be evaluated on
        complete cases, so observations with a missing value in ANY variable the formula
        touches are removed before estimation -- the same thing Stata's estimation commands
        do, and the same thing ``statsmodels.formula.api`` does through its ``dropna``
        handling.

        ``weights`` follows that sample. Two forms are accepted:

        * **a column name** -- ``weights="timepwt48"`` -- the natural statsmodels spelling.
          The weights are read from the same frame, so they cannot disagree with it about
          which rows are in the sample.
        * **a vector aligned with ``data``** -- it is sliced by the rows the formula kept,
          so a 722-element vector stays correct when the formula drops 47 rows.

        Before this, a vector had to be pre-sliced by hand or the call failed with
        "weights have 722 entries but the data have 675 rows" -- a message that reads like
        malformed data when the data are fine. See
        ``replication/papers/ad2013/WESTATA_GAPS.md`` §1.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        # Positional alignment: patsy's kept_index is only safe to use with
        # .loc when the index is unique, so normalise it to a RangeIndex.
        data = data.reset_index(drop=True)

        # Resolve a weight COLUMN NAME before the design is built, so the name is read from
        # the same frame the formula is evaluated on. Doing it afterwards would read from a
        # frame whose index the caller no longer recognises.
        weights = _resolve_weight_column(weights, data)

        X, names, kept_index = build_design(formula, data)

        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise UnsupportedError(
                "P0 only supports a plain column name on the left-hand side; "
                f"got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)

        # THE WEIGHTS FOLLOW THE ESTIMATION SAMPLE. ``kept_index`` are positions in
        # ``data`` (a RangeIndex after the reset above), so the same slice selects the
        # matching weights. Passing the vector through untouched made the constructor's
        # length check fail for every caller whose formula dropped a row -- the single most
        # common situation with real data, and the error pointed at the caller's data
        # rather than at this line.
        if weights is not None:
            weights = _slice_weights_to_sample(weights, kept_index, len(data))

        model = cls(y, X, weights=weights, weight_type=weight_type, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        #: The response column's name, which Stata reports as ``e(depvar)`` and
        #: which ``estat summarize`` lists first.  Recorded here because the
        #: result object otherwise has no way to recover it.
        model._response_name = str(lhs)
        #: The right-hand-side variable names in the order the FORMULA listed
        #: them.  patsy alphabetises its design columns, so ``weight + foreign``
        #: becomes ``[Intercept, foreign, weight]``; Stata keeps the written order
        #: and puts ``_cons`` last.  Without this the original order is
        #: unrecoverable, and ``stata_params`` would report the wrong sequence.
        model._formula_rhs_order = _rhs_variable_order(formula)
        return model

    # ------------------------------------------------------------------ fit
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
            ``None`` selects the Stata default for this model: ``"nonrobust"``
            for unweighted/aweighted/fweighted/iweighted OLS, ``"HC1"`` when the
            model carries ``pweight`` (Stata: ``[pw=w]`` implies ``vce(robust)``).
            Passing ``"nonrobust"`` explicitly for a pweighted model raises
            :class:`ConsistencyError` because Stata cannot express that
            combination either.
        """

        X, y = self._X, self._y
        n, k = X.shape
        if n == 0:
            raise ConsistencyError(
                "regress: no observations remain after listwise deletion"
            )
        if n <= k:
            raise ConsistencyError(
                f"regress: {n} observations but {k} parameters; "
                "the model is not identified"
            )

        rank = int(np.linalg.matrix_rank(X))
        if rank < k:
            raise ConsistencyError(
                "regress: design matrix is rank deficient "
                f"(rank {rank} < {k} columns); drop collinear regressors"
            )

        spec = self._weight_spec
        if spec is None:
            w = np.ones(n, dtype=float)
            effective_n = float(n)
        else:
            w = spec.weights
            effective_n = float(spec.effective_n)

        if cov_type is None:
            effective_cov_type = "HC1" if (spec is not None and spec.implies_robust) else "nonrobust"
        else:
            effective_cov_type = cov_type
            if (
                spec is not None
                and spec.implies_robust
                and str(cov_type).strip().lower() == "nonrobust"
            ):
                raise ConsistencyError(
                    "pweight implies vce(robust): Stata's 'regress y x [pw=w]' is "
                    "identical to adding vce(robust), so a nonrobust covariance "
                    "cannot be requested for a pweighted model"
                )

        # Weighted least squares via the square-root transform keeps a single
        # well-conditioned least-squares solve for both weighted and unweighted
        # models and matches Stata's (X'WX)^-1 X'Wy exactly.
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y - X @ beta

        df_resid = effective_n - k
        if df_resid <= 0:
            raise ConsistencyError(
                f"regress: {effective_n:g} effective observations but {k} "
                "parameters; the model is not identified"
            )

        cov, df_t = compute_cov(
            X,
            resid,
            cov_type=effective_cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            nobs=n,
            df_resid=df_resid,
            nobs_adj=effective_n,
            weights=None if spec is None else w,
            weight_type=None if spec is None else spec.weight_type,
        )

        rss = float((w * resid**2).sum())
        ybar = float((w * y).sum() / w.sum())
        has_intercept = any(name == "Intercept" for name in self._param_names)
        if has_intercept:
            # Centred TSS: Stata's e(mss) is the explained sum of squares about
            # the mean, so e(rss) + e(mss) = e(tss) with tss = sum w (y - ybar)^2.
            tss = float((w * (y - ybar) ** 2).sum())
        else:
            # With noconstant Stata does not centre: r2 = 1 - rss/sum(w y^2) and
            # mss = sum(w y^2) - rss.  Verified on auto.dta,
            # ``regress mpg weight foreign, noconstant``: r2 = 0.8576921885397891,
            # mss = 30883.780324940726 (= 36008 - 5124.219675059).
            tss = float((w * y**2).sum())
        mss = tss - rss
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        # Stata's e(r2_a) = 1 - (1 - r2) * (N - c) / (N - K), where c = 1 when the
        # model has a constant and c = 0 with ``noconstant`` (verified on
        # auto.dta: 0.6532015851691599 and, for ``regress mpg weight foreign,
        # noconstant``, 0.8537391937770055 = 1 - (1-0.8576921885397891)*74/72).
        r2_adj = (
            1.0 - (1.0 - r2) * (effective_n - (1.0 if has_intercept else 0.0)) / df_resid
            if np.isfinite(r2)
            else float("nan")
        )
        sigma2 = rss / df_resid
        rmse = float(np.sqrt(sigma2))
        llf = float(
            -0.5 * effective_n * (np.log(2 * np.pi) + np.log(rss / effective_n) + 1.0)
        )
        # Stata's e(ll_0) is the same Gaussian likelihood for the intercept-only
        # model, i.e. with sigma^2_0 = TSS/N.  Verified on auto.dta for
        # nonrobust, vce(robust), vce(cluster), pweight and fweight (all 0.0
        # relative).  ``noconstant`` models have no e(ll_0) in Stata.
        # NOTE: under aweight Stata's e(ll)/e(ll_0) carry a weight-convention
        # offset that westata does not reproduce (both are ~2.2e-2 off, by the
        # same absolute 4.4044), so ``e(ll) - e(ll_0)`` -- the LR statistic -- is
        # still exact.  See ``retest/fix/linear-iv/FIXES.md``.
        ll_0 = (
            float(-0.5 * effective_n * (np.log(2 * np.pi) + np.log(tss / effective_n) + 1.0))
            if has_intercept and tss > 0
            else None
        )
        df_model = k - 1 if has_intercept else k
        # Stata's e(F) is a Wald test of ALL non-constant coefficients using the
        # covariance matrix actually reported, divided by its degrees of freedom
        # ([R] regress, "Methods and formulas"; verified against StataNow 19.5
        # for nonrobust, vce(robust), vce(cluster) and pweight at <= 1e-15).
        # The classic R^2-form F is what this reduces to for the nonrobust VCE,
        # so one definition covers every cov_type instead of only the default.
        tested = [i for i, name in enumerate(self._param_names) if name != "Intercept"]
        if df_model == 0:
            # Stata reports e(F) = 0 for the intercept-only model (``regress y``).
            fvalue = 0.0
        elif tested:
            fvalue = _wald_f(np.asarray(beta, dtype=float), cov, tested)
        else:
            fvalue = float("nan")
        # ``e(aic)``/``e(bic)`` are westata additions, not Stata values: Stata's
        # ``regress`` stores neither (the three ``regress_auto*.json`` fixtures
        # have no ``aic``/``bic`` key).  They are recorded because
        # ``StataResults.aic``/``.bic`` read them, and they use the Gaussian
        # deviance form ``N log(rss/N) + 2k``.
        #
        # NOTE this is NOT the same number as ``estat ic``'s AIC, which
        # ``estimates.ado:953-1002`` builds as ``-2*e(ll) + 2*e(df)``.  ``e(ll)``
        # here carries the ``N(1 + log 2pi)`` constant that the form above
        # omits, so the two differ by exactly that constant (210.0 on the auto
        # data).  Both are internally consistent; ``result.estat_ic()`` is the
        # Stata-conformant one.
        aic = float(effective_n * np.log(rss / effective_n) + 2 * k)
        bic = float(effective_n * np.log(rss / effective_n) + k * np.log(effective_n))

        # Stata stores the *reference-distribution* degrees of freedom in
        # e(df_r): N - K normally, G - 1 under vce(cluster id) ([R] regress,
        # "Degrees of freedom"; verified on auto.dta: 12 clusters -> e(df_r) =
        # 11 while e(rmse) keeps sqrt(rss/(N-K))).  ``results.df_resid`` already
        # carries that same value, so the two stay in step.
        e: dict[str, Any] = {
            "N": effective_n,
            "sum_w": self._raw_weight_sum,
            # ``regress.ado:174``: ``ereturn local marginsok "XB default"``.  This is the
            # macro ``margins`` consults for which prediction types it may use -- an
            # XB-scale derivative and the command's own default.  ``regress`` sets no
            # ``marginsnotok`` at all, so nothing is forbidden beyond that list.
            "marginsok": "XB default",
            # ``regress.sthlp:437-438``:
            #   e(vce)      vcetype specified in vce()
            #   e(vcetype)  title used to label Std. err.
            # ``regress.ado:48-78`` builds the first from ``vce()``, defaulting to ``ols``:
            #   local vce = cond("`r(vce)'" != "", "`r(vce)'", "ols")
            #   local vce cluster      (for vce(cluster clustvar))
            # The second is the human-readable banner, which is what Stata prints beside
            # the standard errors; ``e(vce)`` supplies it verbatim there.
            "vce": _stata_vce_label(effective_cov_type),
            "vcetype": _stata_vcetype_label(effective_cov_type),
            # ``regress.sthlp:440-441`` lists both macros in the stored-results block:
            #   e(estat_cmd)  program used to implement estat
            #   e(predict)    program used to implement predict
            # and the program ``regress_estat`` exists (``regress_estat.ado:2``), while
            # ``estat.ado:10-11`` dispatches on the macro:
            #   if `"`e(estat_cmd)'"' != "" { `ver' `e(estat_cmd)' `0' }
            # so ``regress_estat`` is the value Stata stores.  ``e(predict)`` is left out
            # on purpose: no ``regress_p.ado`` exists in ado/base and its value could not
            # be confirmed from local evidence, so recording a guess here would be worse
            # than the absent key.
            "estat_cmd": "regress_estat",
            # ``regress.ado:176``: ``ereturn local title "Linear regression"``.
            "title": "Linear regression",
            "df_m": df_model,
            "df_r": df_t,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "sigma": rmse,
            "F": fvalue,
            "ll": llf,
            "rss": rss,
            "mss": mss,
            "rank": k,
            # NOT Stata's ``e(aic)`` -- ``regress`` stores no such macro at all (its
            # help file's stored-results block has no aic/bic, verified by grep over
            # ``regress.sthlp``).  These two are this package's convenience values, on
            # the Gaussian-deviance scale ``N log(rss/N) + 2k``; they deliberately omit
            # the ``N (1 + log 2pi)`` constant that ``e(ll)`` carries, which is why they
            # differ from ``estat ic``'s AIC by exactly that amount.
            #
            # They are kept (removing them would break callers and the tests that pin
            # them) but must not be presented as Stata returns.  ``result.estat_ic()`` is
            # the Stata-conformant surface; use ``e(ll)``/``e(rank)`` if a value tied to
            # Stata's likelihood is needed.
            "aic": aic,
            "bic": bic,
            # This command does NOT store information criteria in Stata, so the two values
            # above are this package's convenience numbers.  The flag lets the summary
            # renderer leave them out of the table -- Stata's own ``regress`` output has no
            # AIC/BIC line, and printing a differently-scaled number there would invite a
            # comparison that cannot succeed.  ``result.estat_ic()`` is the Stata route.
            "_aic_is_stata": False,
        }
        if ll_0 is not None:
            e["ll_0"] = ll_0
        if str(effective_cov_type).strip().lower() == "cluster":
            # One accessor for the cluster labels, so ``groups``/``clusters``/``cluster``
            # all work here as they do elsewhere. Reading ``groups`` directly meant this
            # line KeyError'd (or silently mis-reported ``N_clust``) when a caller used one
            # of the other two spellings -- the exact inconsistency
            # ``replication/papers/adh2013/WESTATA_GAPS.md`` records.
            from westata.vce.dispatch import cluster_labels_from_kwds

            groups = cluster_labels_from_kwds(cov_kwds)
            if groups is not None:
                # A multiway spec supplies one label column per dimension; e(N_clust) then
                # follows Stata and reports the SMALLEST group count, which is the one that
                # limits inference (and matches the df_t the covariance returns). Encoding
                # the two-column object directly raised "must be one-dimensional", so the
                # covariance worked but the fit failed afterwards -- an ordering surprise
                # worth avoiding by handling both shapes here.
                dimensions = _cluster_dimension_columns(groups)
                counts = [
                    np.unique(encode_groups(column)).size for column in dimensions
                ]
                e["N_clust"] = float(min(counts))
                # ``clustertable`` prints the count for EVERY combination of dimensions,
                # so the per-dimension numbers are kept separately: e(N_clust) holds only
                # the minimum, which is what that particular macro means.
                e["_cluster_counts"] = [int(count) for count in counts]
                e["_cluster_names"] = _cluster_dimension_labels(groups, len(counts))
                # Joint counts for every multi-dimension combination, computed from the
                # actual label tuples -- NOT from the per-dimension numbers. The product
                # would be 30*7=210 here by coincidence, but two crossed dimensions with
                # gaps give far fewer real combinations, and a table of products would be
                # wrong in exactly the cases the table exists to reveal.
                from itertools import combinations as _combinations

                joint: dict[str, int] = {}
                labels = _cluster_dimension_labels(groups, len(counts))
                for size in range(2, len(counts) + 1):
                    for indices in _combinations(range(len(counts)), size):
                        tuples = list(
                            zip(*(dimensions[index] for index in indices))
                        )
                        flat = np.empty(len(tuples), dtype=object)
                        flat[:] = tuples
                        key = "#".join(labels[index] for index in indices)
                        joint[key] = int(np.unique(encode_groups(flat)).size)
                e["_cluster_combination_counts"] = joint

        results = StataResults(
            params=beta,
            param_names=self._param_names,
            cov=cov,
            resid=resid,
            fittedvalues=X @ beta,
            nobs=int(n),
            df_resid=df_t,
            df_model=df_model,
            dist="t",
            model_name="regress",
            cov_type=effective_cov_type,
            e=e,
            design_matrix=(X, self._param_names),
            model_data=self._model_data,
            level=level,
        )
        #: ``e(depvar)``: ``estat summarize`` lists this variable first.
        results.response_name = self._response_name
        #: Kept on the RESULT so ``stata_params`` can restore Stata's coefficient
        #: order; patsy's column order does not preserve it.
        results._formula_rhs_order = getattr(self, "_formula_rhs_order", None)
        #: The weight specification, retained because ``estat summarize`` applies
        #: the ESTIMATION weights automatically when the caller passes none --
        #: ``estat_summ.ado:208-231`` rebuilds the weight expression and summarises
        #: with it.  Without this the descriptive table silently returns UNWEIGHTED
        #: statistics for a weighted fit.
        results.weight_spec = getattr(self, "_weight_spec", None)
        return results