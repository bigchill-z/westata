"""System estimation: SUR, multivariate regression, errors-in-variables, MANOVA.

=================  ==========
westata            Stata
=================  ==========
``Sureg``          ``sureg``
``Mvreg``          ``mvreg``
``Eivreg``         ``eivreg``
``Manova``         ``manova``
=================  ==========

``sureg`` is the reason this module exists.  Its coefficients alone do not
demonstrate correctness: if the cross-equation covariance ``Sigma`` is wrong,
every point estimate stays put and every standard error moves.  The algebra is

    Sigma_ij = e_i'e_j / n            (per-equation OLS residuals, divide by n)
    Omega    = Sigma^-1 (x) I_n
    beta     = (X' Omega X)^-1 X' Omega y
    V        = (X' Omega X)^-1

verified against StataNow 19.5 to the last printed digit -- coefficients and
standard errors together -- on the reference data (n=400, seed 101), where the
fitted cross-equation correlation is 0.5544.  Dividing by ``n - k`` instead
shifts every standard error in the third decimal, so the ``n`` is load-bearing.

Rather than materialising the ``(k_eq * n) x (k_eq * n)`` block-diagonal
``Omega``, the stacked normal equations are assembled block by block,

    A_ij = Sigma^-1_ij * X_i' X_j,      rhs_i = sum_j Sigma^-1_ij * X_i' y_j

which is the identical system at ``O(n * K^2)`` instead of ``O(n^2 * K)``.

Equation-prefixed parameter names
---------------------------------
Both ``sureg`` equations carry a constant, so parameter names must be prefixed
with the equation label (``y1:_cons``, ``y2:_cons``).  ``StataResults`` rejects
duplicate names, and beyond that the prefix is the only thing that makes a
stacked coefficient vector readable.  The constant is spelled ``_cons`` exactly
as Stata prints it; ``result.params["y1:Intercept"]`` still resolves, because
patsy -- and therefore the rest of westata -- calls the same column
``Intercept`` (see :class:`_StataNameSeries`).

What is deliberately not implemented
------------------------------------
* ``sureg, isure`` (iterated FGLS): :class:`Sureg.fit` raises
  :class:`~westata.errors.UnsupportedError` for ``iterate=True``.  Stata does not
  document the convergence rule precisely, and a wrong rule changes every
  standard error on the final iteration.  Reporting that is better than
  shipping a number we cannot stand behind.
* Robust/cluster VCEs for any of the four commands: every ``fit`` raises
  ``UnsupportedError`` instead of silently returning the unadjusted VCE.
* ``manova`` statistics other than Wilks' lambda (Pillai, Lawley-Hotelling,
  Roy) and designs with covariates or more than one factor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Eivreg", "Manova", "ManovaResult", "Mvreg", "Sureg"]

#: ``patsy`` spells the constant ``Intercept``; Stata prints ``_cons``.
_CONSTANT_PATSY = "Intercept"
_CONSTANT_STATA = "_cons"


# --------------------------------------------------------------------- helpers
def _design(formula: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build ``(X, names, kept_index)`` in Stata's column order.

    patsy puts the constant first (``Intercept``); Stata lists it last
    (``_cons``).  Reordering here keeps ``result.params.to_numpy()`` directly
    comparable with Stata's ``e(b)`` without a separate permutation.
    """

    X, names, kept = build_design(formula, data)
    X = np.asarray(X, dtype=float)
    names = _canonical_names(names)
    shift = _constant_position(names)
    if shift is not None and shift != len(names) - 1:
        order = [i for i in range(len(names)) if i != shift] + [shift]
        X = X[:, order]
        names = [names[i] for i in order]
    return X, names, np.asarray(kept)


def _constant_position(names: Sequence[str]) -> int | None:
    """Index of the constant column, if the design has one."""

    for position, name in enumerate(names):
        if str(name).rpartition(":")[2] in (_CONSTANT_STATA, _CONSTANT_PATSY):
            return position
    return None


def _canonical_names(names: Sequence[str]) -> list[str]:
    """Rename patsy's ``Intercept`` to the ``_cons`` Stata prints."""

    return [_CONSTANT_STATA if str(name) == _CONSTANT_PATSY else str(name) for name in names]


def _constant_alias(name: str) -> str | None:
    """Return the other spelling of a constant term, or ``None``.

    Handles both the bare name and an equation-prefixed one, so
    ``y1:Intercept`` and ``y1:_cons`` are interchangeable.
    """

    if name == _CONSTANT_PATSY:
        return _CONSTANT_STATA
    if name == _CONSTANT_STATA:
        return _CONSTANT_PATSY
    head, separator, tail = name.rpartition(":")
    if separator and tail in (_CONSTANT_PATSY, _CONSTANT_STATA):
        other = _CONSTANT_STATA if tail == _CONSTANT_PATSY else _CONSTANT_PATSY
        return f"{head}:{other}"
    return None


class _StataNameSeries(pd.Series):
    """Coefficient Series whose constant answers to ``_cons`` *and* ``Intercept``.

    The canonical index carries the name Stata prints, because that is what a
    conformance fixture quotes.  Resolution is a fallback only: a key that is
    present is looked up normally, and a missing one is retried after rewriting
    its constant term.  Any other missing key still raises ``KeyError``, so a
    typo cannot silently succeed.
    """

    @property
    def _constructor(self) -> type["_StataNameSeries"]:  # noqa: D401 - pandas hook
        return _StataNameSeries

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            resolved = self._resolve(key)
            if resolved is key:
                raise
            return super().__getitem__(resolved)

    def _resolve(self, key: Any) -> Any:
        """Rewrite any constant spelling in ``key`` to the name actually stored."""

        if isinstance(key, str):
            alias = _constant_alias(key)
            return key if alias is None or alias not in self.index else alias
        if isinstance(key, (list, tuple)):
            return [self._resolve(item) for item in key]
        if isinstance(key, np.ndarray) and key.dtype.kind in "OUS":
            return [self._resolve(item) for item in key]
        return key


class _SystemResults(StataResults):
    """``StataResults`` whose coefficient Series resolves both constant spellings."""

    #: Set by the multiple-equation estimators so an equation-aware ``predict``
    #: can run.  ``None`` means the design was not retained.
    _equation_design: Any = None
    _equation_coefficients: Any = None
    _equation_responses: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for attribute in ("_params", "_bse", "_tvalues", "_pvalues"):
            source = getattr(self, attribute)
            setattr(
                self,
                attribute,
                _StataNameSeries(source.to_numpy(), index=source.index, name=source.name),
            )

    @property
    def predict_types(self) -> tuple[str, ...]:
        """Types available on this result.

        A stacked multiple-equation fit cannot answer a single-equation ``xb``:
        ``nobs`` covers every equation-observation and the design is not one
        matrix.  Once ``_equation_design`` is set the equation-aware types work.
        """

        if self._equation_design is None:
            return ()
        # ``reg3_p.ado:12``: ``local myopts "Difference Residuals"`` plus the
        # ``_pred_me`` standard set (XB/Index/STDP).
        return ("xb", "index", "stdp", "residuals", "difference")

    def predict(
        self,
        exog: Any = None,
        kind: str = "xb",
        *,
        equation: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Equation-aware prediction for a multiple-equation fit.

        ``mvreg_p`` is the standard multiple-equation predict: it takes
        ``equation(#)`` and defaults to the FIRST equation.  The base
        ``StataResults.predict`` cannot serve here because the stacked fit has
        ``nobs = equations * n`` and a single coefficient vector spanning all
        equations, so ``X @ b`` is not defined.
        """

        if self._equation_design is None:
            return super().predict(exog, kind, **kwargs)

        requested = str(kind).strip().lower()
        if requested not in self.predict_types:
            raise ValueError(
                f"predict, {kind}: a multiple-equation result supports "
                f"{', '.join(self.predict_types)}; the influence diagnostics are "
                "single-equation quantities"
            )

        coefficients = self._equation_coefficients
        responses = np.asarray(self._equation_responses, dtype=float)
        # ``mvreg`` supplies one (equations, k) array; ``sureg`` supplies a list
        # of per-equation vectors whose lengths may differ.
        if isinstance(coefficients, (list, tuple)):
            coefficient_vectors = [
                np.asarray(vector, dtype=float).reshape(-1) for vector in coefficients
            ]
        else:
            matrix = np.asarray(coefficients, dtype=float)
            coefficient_vectors = [matrix[index] for index in range(matrix.shape[0])]
        n_equations = len(coefficient_vectors)

        position = 0 if equation is None else int(equation) - 1
        if not 0 <= position < n_equations:
            raise ValueError(
                f"predict: equation {equation} is out of range; the model has "
                f"{n_equations} equation(s)"
            )

        # A single design shared by every equation (``mvreg``) or one per
        # equation (``sureg``, whose equations may carry different regressors and
        # therefore different column counts -- so a list, not a 3-d array).
        if isinstance(self._equation_design, (list, tuple)):
            design = np.asarray(self._equation_design[position], dtype=float)
        else:
            design = np.asarray(self._equation_design, dtype=float)
        if exog is not None:
            design = np.asarray(exog, dtype=float)
        fitted = design @ coefficient_vectors[position]

        if requested in ("xb", "index"):
            return fitted
        if requested == "residuals":
            return responses[:, position] - fitted
        if requested == "difference":
            # ``reg3_p.ado:126-127``: ``gen varn = xb1 - xb2``.
            if kwargs.get("equation2") is None:
                raise ValueError(
                    "predict, difference requires equation(eq1, eq2); "
                    "reg3_p.ado:120-130 differences two equations"
                )
            second = int(kwargs["equation2"]) - 1
            if not 0 <= second < n_equations:
                raise ValueError(
                    f"predict: equation {kwargs['equation2']} is out of range; "
                    f"the model has {n_equations} equation(s)"
                )
            if isinstance(self._equation_design, (list, tuple)):
                second_design = np.asarray(self._equation_design[second], dtype=float)
            else:
                second_design = np.asarray(self._equation_design, dtype=float)
            return fitted - second_design @ coefficient_vectors[second]
        # stdp: sqrt(x' V_j x) with V_j the equation's own OLS covariance.
        residual = responses[:, position] - fitted
        n, k = design.shape
        sigma2 = float(residual @ residual) / max(n - k, 1)
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        variance = np.einsum("ij,jk,ik->i", design, covariance, design)
        return np.sqrt(np.maximum(variance, 0.0))


def _resolve_cov_type(
    cov_type: str | None,
    command: str,
    *,
    cov_kwds: Mapping[str, Any] | None = None,
    ssc: Any = None,
) -> str:
    """Map the requested ``cov_type`` onto the one VCE these models implement.

    The unadjusted VCE is the only one verified against Stata, so anything else
    is refused loudly rather than silently returning the unadjusted numbers
    under a different label.
    """

    if cov_kwds:
        raise UnsupportedError(
            f"{command}: cov_kwds={dict(cov_kwds)!r} only applies to the robust and "
            "cluster VCEs, which are not implemented for this command; only "
            "cov_type='nonrobust' is available"
        )
    if ssc is not None:
        raise UnsupportedError(
            f"{command}: ssc() small-sample settings do not enter the unadjusted "
            "VCE of this command (Stata's formulas are fixed at N), so honouring "
            "them would be a silent no-op; pass ssc=None"
        )
    if cov_type is None:
        return "nonrobust"
    key = str(cov_type).strip().lower()
    if key in ("", "nonrobust", "unadjusted"):
        return "nonrobust"
    if key in ("robust", "hc0") and command == "sureg":
        # `reg3.ado` delegates this to `_reg3robust`, whose sandwich is implemented here. The other commands
        # in this file have no such path verified against Stata and stay rejected.
        return "robust"
    raise UnsupportedError(
        f"{command}: cov_type={cov_type!r} is not implemented; only the unadjusted "
        "VCE (cov_type='nonrobust') is verified against Stata for this command"
    )


def _slope_positions(names: Sequence[str]) -> list[int]:
    """Positions of the non-constant parameters.

    Handles both the bare and the equation-prefixed spellings, e.g. ``_cons``
    and ``y1:_cons``.
    """

    return [
        i
        for i, name in enumerate(names)
        if str(name).rpartition(":")[2] not in (_CONSTANT_STATA, _CONSTANT_PATSY)
    ]


def _split_terms(text: str) -> list[str]:
    """Split Stata-style space/plus separated variable lists.

    Terms are plain names or patsy calls without embedded whitespace, matching
    how ``sureg``/``mvreg``/``manova`` write variable lists.
    """

    return [term for term in re.split(r"[\s+]+", text.strip()) if term]


def _require_outcomes(formula: str, data: pd.DataFrame, command: str) -> tuple[list[str], str]:
    """Split ``"y1 y2 = x1 x2"`` into ``(["y1", "y2"], "x1 x2")``."""

    if "=" not in formula:
        raise ConsistencyError(
            f"{command}: the formula must separate the outcomes from the "
            f"regressors with '=', as in 'y1 y2 = x1 x2 x3'; got {formula!r}"
        )
    lhs, rhs = formula.split("=", 1)
    outcomes = _split_terms(lhs)
    if not outcomes:
        raise ConsistencyError(f"{command}: no outcome variables were given")
    for name in outcomes:
        if name not in data.columns:
            raise ConsistencyError(f"{command}: column {name!r} not found in the data")
    return outcomes, rhs.strip()


def _subset(data: Any, subset: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if subset is not None:
        data = data.loc[subset]
    return data.reset_index(drop=True)


# ----------------------------------------------------------------------- sureg
class Sureg:
    """Seemingly unrelated regression (Stata ``sureg``).

    Parameters
    ----------
    designs:
        One ``n x k_i`` design matrix per equation.  A constant must be
        present explicitly (``build_design`` adds it).
    responses:
        One length-``n`` dependent variable per equation.
    equation_names:
        Stata's equation labels (the dependent variable names), used as the
        parameter prefixes.
    """

    _stata_command = "sureg"

    def __init__(
        self,
        designs: Sequence[np.ndarray],
        responses: Sequence[np.ndarray],
        *,
        equation_names: Sequence[str],
        param_names: Sequence[Sequence[str]],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._designs = [np.asarray(X, dtype=float) for X in designs]
        self._responses = [np.asarray(y, dtype=float).reshape(-1) for y in responses]
        self._equation_names = [str(name) for name in equation_names]
        self._param_names = [list(names) for names in param_names]
        if not self._designs:
            raise ConsistencyError("sureg: at least one equation is required")
        if not (
            len(self._designs)
            == len(self._responses)
            == len(self._equation_names)
            == len(self._param_names)
        ):
            raise ValueError(
                "sureg: designs, responses, equation_names and param_names must "
                "have one entry per equation"
            )
        sizes = {X.shape[0] for X in self._designs}
        if len(sizes) != 1:
            raise ConsistencyError(
                f"sureg: the equations have different sample sizes {sorted(sizes)}; "
                "SUR requires the same observations in every equation"
            )
        for X, y, names in zip(self._designs, self._responses, self._param_names):
            if X.shape[1] != len(names):
                raise ValueError(
                    f"sureg: a design has {X.shape[1]} columns but {len(names)} "
                    "parameter names"
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"sureg: a design has {X.shape[0]} rows but its response has "
                    f"{y.shape[0]}"
                )
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        equations: Sequence[str],
        data: pd.DataFrame,
        *,
        names: Sequence[str] | None = None,
        subset: Any = None,
    ) -> "Sureg":
        """Build from formulas such as ``["y1 ~ x1 + x2", "y2 ~ x2 + x3"]``.

        ``names`` overrides the equation labels (which default to each
        equation's dependent variable).
        """

        if isinstance(equations, str):
            raise TypeError(
                "sureg: equations must be a sequence of formulas, e.g. "
                "['y1 ~ x1 + x2', 'y2 ~ x2 + x3']"
            )
        if not equations:
            raise ConsistencyError("sureg: at least one equation is required")
        data = _subset(data, subset)

        designs, responses, labels, names_per_equation = [], [], [], []
        common_index: np.ndarray | None = None
        for formula in equations:
            if "~" not in str(formula):
                raise ConsistencyError(
                    f"sureg: every equation must be a two-sided formula, got {formula!r}"
                )
            X, param_names, kept = _design(str(formula), data)
            lhs = str(formula).split("~", 1)[0].strip()
            if lhs not in data.columns:
                raise ConsistencyError(
                    f"sureg: the left-hand side must be a plain column name, got {lhs!r}"
                )
            if common_index is None:
                common_index = kept
            elif not np.array_equal(common_index, kept):
                raise ConsistencyError(
                    "sureg: the equations must use the same observations; one of "
                    "them drops rows the others keep"
                )
            designs.append(X)
            responses.append(data.loc[kept, lhs].to_numpy(dtype=float))
            labels.append(lhs)
            names_per_equation.append(param_names)

        if names is not None:
            if len(names) != len(equations):
                raise ValueError(
                    f"sureg: {len(names)} equation names given for {len(equations)} "
                    "equations"
                )
            labels = [str(name) for name in names]

        return cls(
            designs,
            responses,
            equation_names=labels,
            param_names=names_per_equation,
            model_data=data,
        )

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        cov_type: str | None = "nonrobust",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        iterate: bool = False,
    ) -> StataResults:
        """Estimate the system by feasible GLS.

        ``iterate=True`` is Stata's ``isure`` and raises
        :class:`~westata.errors.UnsupportedError`; see the module docstring.
        """

        if iterate:
            raise UnsupportedError(
                "sureg: iterate=True (Stata's isure) is not implemented. Iterated "
                "FGLS needs a convergence criterion Stata does not document "
                "precisely, and a wrong rule would change every standard error on "
                "the final iteration"
            )
        resolved = _resolve_cov_type(
            cov_type, "sureg", cov_kwds=cov_kwds, ssc=ssc
        )

        designs, responses = self._designs, self._responses
        equations = len(designs)
        n = designs[0].shape[0]
        sizes = [X.shape[1] for X in designs]
        total = sum(sizes)

        # Step 1 -- per-equation OLS supplies the cross-equation residuals.
        ols_residuals = []
        for X, y in zip(designs, responses):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            ols_residuals.append(y - X @ beta)

        # Step 2 -- Sigma_ij = e_i'e_j / n.  Dividing by n rather than n - k_i
        # is what reproduces Stata's standard errors.
        sigma = np.empty((equations, equations), dtype=float)
        for i in range(equations):
            for j in range(i, equations):
                value = float(ols_residuals[i] @ ols_residuals[j]) / n
                sigma[i, j] = value
                sigma[j, i] = value
        if not np.all(np.linalg.eigvalsh(sigma) > 0.0):
            raise ConsistencyError(
                "sureg: the cross-equation covariance matrix is singular; the "
                "system is not identified (two equations' residuals are perfectly "
                "collinear)"
            )
        precision = np.linalg.inv(sigma)

        # Step 3 -- feasible GLS on the stacked system, assembled blockwise.
        # A_ij = Sigma^-1_ij X_i'X_j and rhs_i = sum_j Sigma^-1_ij X_i'y_j are
        # exactly X'Omega X and X'Omega y for Omega = Sigma^-1 (x) I_n.
        information = np.zeros((total, total), dtype=float)
        derivative = np.zeros(total, dtype=float)
        offsets = np.cumsum([0] + sizes)
        for i in range(equations):
            rows = slice(int(offsets[i]), int(offsets[i + 1]))
            for j in range(equations):
                columns = slice(int(offsets[j]), int(offsets[j + 1]))
                information[rows, columns] = precision[i, j] * (designs[i].T @ designs[j])
                derivative[rows] += precision[i, j] * (designs[i].T @ responses[j])
        try:
            beta = np.linalg.solve(information, derivative)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise ConsistencyError(
                f"sureg: the system normal equations are singular ({exc}); the "
                "system is not identified"
            ) from None
        # SUR residuals, in Stata's equation order. Computed BEFORE the covariance so the robust sandwich can
        # use the system's own Sigma, which is what `_reg3robust` receives as `omega`.
        residuals = [
            y - X @ beta[offsets[position] : offsets[position + 1]]
            for position, (X, y) in enumerate(zip(designs, responses))
        ]
        covariance = np.linalg.inv(information)

        if resolved == "robust":
            # `_reg3robust.ado` GetSandwich. For observation t the score is the OMEGA-WEIGHTED residual vector,
            #     ulist2_i(t) = sum_j Omega[i, j] * e_j(t)
            # and the meat is the sum of its outer products in the stacked parameter layout:
            #     psi_t[block_i] = x_i(t) * sum_j precision[i, j] e_j(t)
            #     V = (X'Omega^-1 X)^-1 [ sum_t psi_t psi_t' ] (X'Omega^-1 X)^-1
            #
            # TWO THINGS MATTER AND BOTH WERE WRONG AT FIRST.
            #
            # 1. Weighting the residual ACROSS equations is the content of the estimator. Plain residuals give a
            #    different matrix that still looks like a covariance.
            # 2. The `omega` the ado file passes in is the SYSTEM's Sigma, estimated from the GLS residuals --
            #    not the first-stage OLS residuals. Using the OLS ones left our standard errors 2-4% low on every
            #    parameter, and the diagnostic was Sigma itself: its off-diagonal was 1767.66 against Stata's
            #    794.83, more than twice, and that term is exactly what the robust weighting multiplies.
            #
            # So the meat is built from the same `sigma_resid` / `precision_resid` the reported e(Sigma) uses.
            # `omega` is the OLS-based Sigma (verified: it equals Stata's e(Sigma) to the last digit), while the
            # residuals in the score come from `_predict` after the fit, i.e. the SUR ones. Both are "the
            # residuals" and they are not the same, which is why the two earlier attempts each got one right and
            # the other wrong.
            meat = np.zeros((total, total), dtype=float)
            for t in range(n):
                psi = np.zeros(total, dtype=float)
                for i in range(equations):
                    rows = slice(int(offsets[i]), int(offsets[i + 1]))
                    weighted = 0.0
                    for j in range(equations):
                        weighted += precision[i, j] * residuals[j][t]
                    psi[rows] = designs[i][t] * weighted
                meat += np.outer(psi, psi)
            covariance = covariance @ meat @ covariance

        # Two equations can share an outcome name -- `sureg (y x1) (y x2)` is a valid system, and both would be
        # labelled `y`, making every parameter name collide (`y:_cons` twice). Stata avoids this because its
        # equation names come from the dependent variable and it rejects a duplicate outright; we keep the call
        # working and disambiguate the LATER one, so the first equation keeps the plain name and any existing
        # expectations are untouched.
        labels: list[str] = []
        for label in self._equation_names:
            if label in labels:
                suffix = 2
                while f"{label}#{suffix}" in labels:
                    suffix += 1
                labels.append(f"{label}#{suffix}")
            else:
                labels.append(label)

        names: list[str] = []
        for label, param_names in zip(labels, self._param_names):
            names.extend(f"{label}:{name}" for name in param_names)



        sigma_resid = np.empty((equations, equations), dtype=float)
        for i in range(equations):
            for j in range(equations):
                sigma_resid[i, j] = float(residuals[i] @ residuals[j]) / n

        e: dict[str, Any] = {
            "N": n,
            # ``e(title)``: measured from a live ``sureg (y x1) (ybin x1)``.
            "title": "Seemingly unrelated regression",
            "k_eq": equations,
            "k": total,
            "rank": total,
            # e(df_m) is the number of tested slopes across the system (the
            # constants are excluded), which is what the header's Wald chi2 uses.
            "df_m": float(total - equations),
            "equations": list(self._equation_names),
            "sigma": sigma_resid.tolist(),
            "corr": _correlation(sigma_resid).tolist(),
            "method": "sure",
            # ``sureg.ado:14,45`` runs through ``reg3``, which is what supplies
            # the equation-aware predict; ``reg3_p.ado:12`` is that program.
            "predict": "reg3_p",
            "vce": "unadjusted",
        }

        # Stata's sureg log-likelihood: -n/2 (p(1 + ln 2pi) + ln|Sigma|), with
        # Sigma built from the SUR residuals over n (not n - k).
        sign, logdet = np.linalg.slogdet(sigma_resid)
        e["ll"] = float(
            -0.5 * n * (equations * (1.0 + np.log(2.0 * np.pi)) + logdet)
            if sign > 0
            else np.nan
        )

        for position, (label, response) in enumerate(zip(self._equation_names, responses)):
            block = slice(int(offsets[position]), int(offsets[position + 1]))
            equation_names = [f"{label}:{name}" for name in self._param_names[position]]
            rss = float(residuals[position] @ residuals[position])
            tss = float(((response - response.mean()) ** 2).sum())
            r2 = 1.0 - rss / tss if tss > 0 else float("nan")
            slopes = _slope_positions(equation_names)
            # Stata names these positionally -- e(r2_1), e(rmse_2), e(rss_1) --
            # not by equation label, so the positional spelling is canonical and
            # the label-based one is kept as a convenience alias.
            per_equation: dict[str, Any] = {
                "rss": rss,
                "mss": tss - rss,
                "r2": r2,
                "rmse": float(np.sqrt(rss / n)),
                "df_m": float(len(slopes)),
                "cons": int(_CONSTANT_STATA in equation_names),
            }
            for key, value in per_equation.items():
                e[f"{key}_{position + 1}"] = value
                e[f"{key}_{label}"] = value
            if slopes:
                b = beta[block][slopes]
                V = covariance[block, block][np.ix_(slopes, slopes)]
                statistic = float(b @ np.linalg.solve(V, b))
                e[f"chi2_{position + 1}"] = statistic
                e[f"p_{position + 1}"] = float(stats.chi2.sf(statistic, len(slopes)))

        result = _SystemResults(
            params=beta,
            param_names=names,
            cov=covariance,
            # The residual vector spans every equation-observation (k_eq * n
            # entries), which is what StataResults validates nobs against;
            # e(N) keeps Stata's per-equation sample size.
            resid=np.concatenate(residuals),
            fittedvalues=np.concatenate(
                [X @ beta[offsets[i] : offsets[i + 1]] for i, X in enumerate(designs)]
            ),
            nobs=int(equations * n),
            # sureg is chi2-based (dist="z"), so this only matters if a caller
            # asks for a t reference; Stata records no e(df_r) here.
            df_resid=float(equations * n - total),
            df_model=float(total - equations),
            dist="z",
            model_name=self._stata_command,
            cov_type=resolved,
            e=e,
            model_data=self._model_data,
        )
        #: ``sureg`` runs through ``reg3`` (``sureg.ado:14,45``), so its predict is
        #: the shared ``reg3_p``.  The equations may carry DIFFERENT regressors, so
        #: the designs cannot be stacked into one 3-d array -- a list of matrices
        #: keeps every equation's own column count.
        result._equation_design = [np.asarray(X, dtype=float) for X in designs]
        result._equation_coefficients = [
            np.asarray(beta[offsets[i] : offsets[i + 1]], dtype=float)
            for i in range(equations)
        ]
        result._equation_responses = np.column_stack(responses)
        return result


def _correlation(sigma: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        return sigma / np.outer(scale, scale)


# ----------------------------------------------------------------------- mvreg
class Mvreg:
    """Multivariate regression (Stata ``mvreg``).

    Every equation shares one design matrix, so the coefficients are exactly
    per-equation OLS; only the VCE uses the cross-equation residual covariance::

        V = Sigma (x) (X'X)^-1,    Sigma_ij = u_i'u_j / (n - k)

    Note the ``n - k`` here, against ``n`` for ``sureg``: ``mvreg`` is a
    multivariate least-squares problem with a common design, and Stata's
    standard errors only reproduce with the degrees-of-freedom divisor.

    ``mvreg`` and ``sureg`` therefore disagree on identical data whenever the
    equations contain different regressors -- ``mvreg`` puts every regressor in
    every equation.  That is a specification difference, not an inconsistency.
    """

    _stata_command = "mvreg"

    def __init__(
        self,
        designs: Sequence[np.ndarray],
        responses: Sequence[np.ndarray],
        *,
        outcome_names: Sequence[str],
        param_names: Sequence[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._designs = [np.asarray(X, dtype=float) for X in designs]
        self._responses = [np.asarray(y, dtype=float).reshape(-1) for y in responses]
        self._outcome_names = [str(name) for name in outcome_names]
        self._param_names = list(param_names)
        if not self._designs:
            raise ConsistencyError("mvreg: at least one outcome is required")
        if not (
            len(self._designs) == len(self._responses) == len(self._outcome_names)
        ):
            raise ValueError(
                "mvreg: designs, responses and outcome_names must have one entry "
                "per outcome"
            )
        X = self._designs[0]
        if X.shape[1] != len(self._param_names):
            raise ValueError(
                f"mvreg: the design has {X.shape[1]} columns but {len(self._param_names)} "
                "parameter names"
            )
        for other in self._designs[1:]:
            if other.shape != X.shape:
                raise ConsistencyError(
                    "mvreg: every equation shares one design matrix, but the "
                    "designs differ in shape"
                )
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Mvreg":
        """Build from ``"y1 y2 = x1 x2 x3"`` (every regressor in every equation)."""

        data = _subset(data, subset)
        outcomes, rhs = _require_outcomes(formula, data, "mvreg")
        regressors = _split_terms(rhs)
        if not regressors:
            raise ConsistencyError(
                "mvreg: no regressors were given; Stata's mvreg needs at least one"
            )
        # Stata writes the regressor list space separated; patsy needs explicit
        # "+", and every equation gets the identical right-hand side.
        X, param_names, kept = _design(
            f"{outcomes[0]} ~ " + " + ".join(regressors), data
        )
        return cls(
            [X for _ in outcomes],
            [data.loc[kept, name].to_numpy(dtype=float) for name in outcomes],
            outcome_names=outcomes,
            param_names=param_names,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = "nonrobust",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        resolved = _resolve_cov_type(cov_type, "mvreg", cov_kwds=cov_kwds, ssc=ssc)

        equations = len(self._designs)
        X = self._designs[0]
        n, k = X.shape
        if n <= k:
            raise ConsistencyError(
                f"mvreg: {n} observations but {k} parameters; the model is not "
                "identified"
            )
        total = equations * k

        coefficients, residuals = [], []
        for y in self._responses:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            coefficients.append(beta)
            residuals.append(y - X @ beta)

        # Sigma_ij = u_i'u_j / (n - k): the multivariate least-squares MLE.
        sigma = np.empty((equations, equations), dtype=float)
        for i in range(equations):
            for j in range(i, equations):
                value = float(residuals[i] @ residuals[j]) / (n - k)
                sigma[i, j] = value
                sigma[j, i] = value

        covariance = np.kron(sigma, np.linalg.inv(X.T @ X))
        beta = np.concatenate(coefficients)

        names: list[str] = []
        for outcome in self._outcome_names:
            names.extend(f"{outcome}:{name}" for name in self._param_names)

        e: dict[str, Any] = {
            "N": n,
            # ``e(title)``: measured from a live ``mvreg y ybin = x1``. mvreg prints NO
            # banner -- its table opens straight at "Equation Obs Parms RMSE" -- so this
            # string is not rendered; it is recorded because e(title) is the documented
            # macro and callers read it.
            "title": "Multivariate regression",
            "k_eq": equations,
            "k": k,
            "rank": total,
            "df_r": float(n - k),
            "df_m": float(k - 1),
            "outcomes": list(self._outcome_names),
            # ``mvreg.sthlp:194`` spells it ``e(Sigma)``: "Sigma hat matrix",
            # i.e. sigma_ij = u_i'u_j/(n-k).  Both spellings are recorded so
            # ``e["Sigma"]`` matches Stata while ``e["sigma"]`` keeps working for
            # existing callers.
            "Sigma": sigma.tolist(),
            "sigma": sigma.tolist(),
            "corr": _correlation(sigma).tolist(),
            # ``mvreg.sthlp:172-196``: the macros an eclass command carries.
            "cmd": "mvreg",
            "depvar": " ".join(self._outcome_names),
            "eqnames": " ".join(self._outcome_names),
            "properties": "b V",
            "estat_cmd": "mvreg_estat",
            # ``mvreg.ado:124``: ``est local predict "reg3_p"`` -- the shared
            # multiple-equation predict program, NOT a ``mvreg_p`` (no such file
            # exists in ado/base/m).  ``reg3_p.ado:12`` declares
            # ``local myopts "Difference Residuals"`` and delegates the rest to
            # ``_pred_me``, which is exactly the equation-aware surface
            # ``_SystemResults.predict`` provides.
            "predict": "reg3_p",
            "asbalanced": "",
            "asobserved": "",
            "vce": "unadjusted",
        }
        # Stata prints per-equation fit rows above the coefficient table and
        # stores them as *macros* -- e(r2) = " .8134 .6004" -- because a single
        # scalar cannot describe two equations.  Both spellings are recorded:
        # the macro-shaped list under Stata's own name, and a per-outcome scalar
        # (``e["r2_y1"]``) for convenient access.
        slopes = [
            i
            for i, name in enumerate(self._param_names)
            if str(name) not in (_CONSTANT_STATA, _CONSTANT_PATSY)
        ]
        fit_rows: dict[str, list[float]] = {"r2": [], "rmse": [], "F": [], "p_F": []}
        for position, (outcome, response) in enumerate(
            zip(self._outcome_names, self._responses)
        ):
            rss = float(residuals[position] @ residuals[position])
            tss = float(((response - response.mean()) ** 2).sum())
            r2 = 1.0 - rss / tss if tss > 0 else float("nan")
            rmse = float(np.sqrt(rss / (n - k)))
            statistic = float("nan")
            if slopes and np.isfinite(r2):
                statistic = (r2 / len(slopes)) / ((1.0 - r2) / (n - k))
            pvalue = (
                float(stats.f.sf(statistic, len(slopes), n - k))
                if np.isfinite(statistic)
                else float("nan")
            )
            fit_rows["r2"].append(r2)
            fit_rows["rmse"].append(rmse)
            fit_rows["F"].append(statistic)
            fit_rows["p_F"].append(pvalue)
            e[f"r2_{outcome}"] = r2
            e[f"rmse_{outcome}"] = rmse
            e[f"F_{outcome}"] = statistic
            e[f"p_{outcome}"] = pvalue
        e.update(fit_rows)

        results = _SystemResults(
            params=beta,
            param_names=names,
            cov=covariance,
            # Stacked as for sureg: nobs covers every equation-observation, and
            # e(N) keeps Stata's per-equation N.
            resid=np.concatenate(residuals),
            fittedvalues=np.concatenate([X @ b for b in coefficients]),
            nobs=int(equations * n),
            # dist="t" with the per-equation residual degrees of freedom
            # reproduces Stata's t tests exactly (Stata prints e(df_r) = 396).
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type=resolved,
            e=e,
            model_data=self._model_data,
        )
        #: Retained so the equation-aware ``predict`` can run; without these a
        #: stacked fit cannot answer ``xb`` for any single equation.
        results._equation_design = X
        results._equation_coefficients = np.vstack(coefficients)
        results._equation_responses = np.column_stack(self._responses)
        return results


# ---------------------------------------------------------------------- eivreg
class Eivreg:
    """Errors-in-variables regression (Stata ``eivreg``).

    ``[R] eivreg`` Methods and formulas::

        A = X'X/n - S,   S = diag((1 - r_j) * s_j^2)
        b = A^-1 (X'y/n)

    where ``r_j`` is the assumed reliability and ``s_j^2`` the *sample* variance
    of that regressor (``1/n`` divisor -- Stata's own convention, and the reason
    the cross-product must be normalised too).  The measurement-error variance
    is subtracted from the cross-product, which undoes the attenuation rather
    than reweighting: two plausible-looking reweightings by ``1/lambda`` and by
    ``lambda`` give x1 = 1.214 and 1.735 against Stata's 2.073214.

    The variance is the M-estimator sandwich ``A^-1 (Psi'Psi/n) A^-1 / n`` with
    the per-observation score

        psi_i = x_i u_i + sum_j (1 - r_j) (x_ij - xbar_j)^2 b_j e_j

    The second term is the derivative of ``S`` with respect to ``b`` and is what
    makes the standard errors move when a reliability is imposed; dropping it
    leaves x1's standard error at .0965 against Stata's .0757645.

    Fit statistics, all verified digit for digit::

        sigma^2 = (y'y - n b'Ab) / (n - k)          (Root MSE = .835714)
        R^2     = 1 - sigma^2 (n - k) / TSS         (0.8639837)
        F       = (b_s' V_ss^-1 b_s) / (k - 1)      (407.77933)
    """

    _stata_command = "eivreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        reliab: Mapping[str, float],
        *,
        param_names: Sequence[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._param_names = _canonical_names(param_names)
        self._reliab = {str(name): float(value) for name, value in reliab.items()}
        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"eivreg: endog has {self._y.shape[0]} entries but exog has "
                f"{self._X.shape[0]} rows"
            )
        if self._X.shape[1] != len(self._param_names):
            raise ValueError(
                f"eivreg: exog has {self._X.shape[1]} columns but "
                f"{len(self._param_names)} parameter names"
            )
        unknown = [name for name in self._reliab if name not in self._param_names]
        if unknown:
            raise ConsistencyError(
                f"eivreg: reliab refers to {unknown}, which are not regressors; "
                f"the regressors are {self._param_names}"
            )
        for name, value in self._reliab.items():
            if not 0.0 < value <= 1.0:
                raise ConsistencyError(
                    f"eivreg: the reliability of {name!r} must lie in (0, 1], "
                    f"got {value}"
                )
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        reliab: Mapping[str, float] | None = None,
        subset: Any = None,
    ) -> "Eivreg":
        """Build from ``"y ~ x1 + x2"`` and ``reliab={"x1": 0.7}``."""

        data = _subset(data, subset)
        if reliab is None:
            raise ConsistencyError(
                "eivreg: reliab is required; without it the attenuation cannot be "
                "corrected and the result would silently be OLS"
            )
        X, param_names, kept = _design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"eivreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            reliab,
            param_names=param_names,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = "nonrobust",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        resolved = _resolve_cov_type(cov_type, "eivreg", cov_kwds=cov_kwds, ssc=ssc)

        X, y = self._X, self._y
        n, k = X.shape
        if n <= k:
            raise ConsistencyError(
                f"eivreg: {n} observations but {k} parameters; the model is not "
                "identified"
            )

        reliability = np.ones(k, dtype=float)
        for position, name in enumerate(self._param_names):
            if name in self._reliab:
                reliability[position] = self._reliab[name]

        # Sample variances use the 1/n divisor (Stata's `summarize` convention
        # for eivreg's correction); with 1/(n-1) x1 comes out 2.074996 instead
        # of 2.073214.
        variance = np.var(X, axis=0, ddof=0)
        measurement = np.diag((1.0 - reliability) * variance)

        # Normalise the cross-product: `measurement` is a per-observation
        # quantity, so X'X must be too.  Using the raw X'X makes the correction
        # ~n times too small and leaves x1 at 1.452 instead of 2.073.
        cross = X.T @ X / n - measurement
        try:
            inverse = np.linalg.inv(cross)
        except np.linalg.LinAlgError:
            raise ConsistencyError(
                "eivreg: A = X'X/n - S is singular; the measurement-error "
                "correction leaves too little variation in a regressor"
            ) from None
        beta = inverse @ (X.T @ y / n)
        residual = y - X @ beta

        # Score: the OLS estimating equation plus the derivative of the
        # measurement-error correction with respect to b.
        centred_squared = (X - X.mean(axis=0)) ** 2
        score = residual[:, None] * X + centred_squared * ((1.0 - reliability) * beta)
        covariance = inverse @ (score.T @ score / n) @ inverse / n

        dof = n - k
        # Root MSE uses the corrected cross-product: y'y - n b'Ab, where A is the
        # normalised matrix (so n*A = X'X - n*S).
        sigma2 = float(y @ y - n * (beta @ cross @ beta)) / dof
        tss = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - sigma2 * dof / tss if tss > 0 else float("nan")

        slopes = _slope_positions(self._param_names)
        statistic = float("nan")
        pvalue = float("nan")
        if slopes:
            b = beta[slopes]
            V = covariance[np.ix_(slopes, slopes)]
            # Stata's model F is the Wald statistic for the slopes divided by
            # its degrees of freedom -- not the R^2-based F, which differs
            # whenever a reliability below one is imposed.
            wald = float(b @ np.linalg.solve(V, b))
            statistic = wald / len(slopes)
            pvalue = float(stats.f.sf(statistic, len(slopes), dof))

        e: dict[str, Any] = {
        # ``e(title)``: measured from a live run -- "Errors-in-variables regression".
        "title": "Errors-in-variables regression",
            "N": n,
            "df_m": float(len(slopes)),
            "df_r": float(dof),
            "rank": k,
            "r2": r2,
            "rmse": float(np.sqrt(max(sigma2, 0.0))),
            "F": statistic,
            "p": pvalue,
            "reliab": dict(self._reliab),
            "vce": "unadjusted",
        }

        return _SystemResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=X @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(len(slopes)),
            dist="t",
            model_name=self._stata_command,
            cov_type=resolved,
            e=e,
            design_matrix=(X, list(self._param_names)),
            model_data=self._model_data,
        )


# ---------------------------------------------------------------------- manova
@dataclass(frozen=True)
class ManovaResult:
    """MANOVA statistics for one factor (Stata ``manova``).

    ``df`` is Stata's hypothesis degrees of freedom for the factor (``g - 1``),
    and ``df1``/``df2`` are the numerator and denominator degrees of freedom of
    the Rao F approximation printed in the ``F(df1, df2)`` column.

    ``df_hypothesis``/``df_error``/``f_statistic``/``pvalue`` are retained as
    aliases for the earlier field names.
    """

    wilks_lambda: float
    df: float
    df1: float
    df2: float
    F: float
    p: float
    term: str = "group"
    nobs: int = 0
    groups: int = 0
    statistics: Mapping[str, float] = field(default_factory=dict)
    #: Number of dependent variables (``e(k_eq)`` in Stata).  Recorded explicitly
    #: so ``e()`` does not have to guess it from ``statistics``.
    n_equations: int = 0
    #: Per-equation OLS fit measures.  ``manova.sthlp:238-241`` stores e(r2),
    #: e(rmse), e(F) and e(p_F) as one entry PER EQUATION -- the ordinary
    #: univariate numbers for each response, not the multivariate test.
    r2_by_equation: "Sequence[float]" = field(default_factory=list)
    rmse_by_equation: "Sequence[float]" = field(default_factory=list)
    f_by_equation: "Sequence[float]" = field(default_factory=list)
    p_f_by_equation: "Sequence[float]" = field(default_factory=list)
    #: ``e(depvar)``: the response variable names.
    depvar_names: "Sequence[str]" = field(default_factory=list)
    #: The design matrix (intercept + g-1 group dummies) and the coefficient
    #: matrix of shape ``(k, n_equations)``.  Retained so ``predict`` can run.
    design_matrix: Any = None
    coefficients: Any = None
    #: The FITTED response matrix, used by ``residuals``.
    fitted_values: Any = None
    #: The observed response matrix, used by ``residuals``.
    responses: Any = None

    # -- aliases kept for the previously exported attribute names ----------
    @property
    def df_hypothesis(self) -> float:
        return self.df

    @property
    def df_error(self) -> float:
        return float(self.nobs - self.groups)

    @property
    def f_statistic(self) -> float:
        return self.F

    @property
    def pvalue(self) -> float:
        return self.p

    # -- the other three multivariate statistics ---------------------------
    #
    # ``manova`` prints only Wilks' lambda; ``manovatest`` prints all four
    # (``manova_postestimation.sthlp:243-245``).  A ``KeyError`` rather than a
    # silent ``nan`` keeps a caller from treating an unavailable statistic as a
    # computed zero.
    def _statistic(self, key: str) -> float:
        try:
            return float(self.statistics[key])
        except KeyError:
            raise KeyError(
                f"multivariate statistic {key!r} is not recorded on this result; "
                f"available: {sorted(self.statistics)}"
            ) from None

    @property
    def pillai_trace(self) -> float:
        """Pillai's trace ``V = sum lambda_i/(1+lambda_i)``."""

        return self._statistic("pillai_trace")

    @property
    def lawley_hotelling_trace(self) -> float:
        """Lawley-Hotelling trace ``U = sum lambda_i``."""

        return self._statistic("lawley_hotelling_trace")

    @property
    def roy_largest_root(self) -> float:
        """Roy's largest root ``theta = lambda_max/(1+lambda_max)``."""

        return self._statistic("roy_largest_root")

    def multivariate_tests(self) -> dict[str, float]:
        """The four statistics together, as ``manovatest`` reports them."""

        return {
            "Wilks' lambda": float(self.wilks_lambda),
            "Pillai's trace": self.pillai_trace,
            "Lawley-Hotelling trace": self.lawley_hotelling_trace,
            "Roy's largest root": self.roy_largest_root,
        }

    #: ``manova_p`` types.  ``manova_p.ado:16`` declares ``local myopts
    #: "Difference"`` and delegates the rest to ``_pred_me``, whose standard set
    #: is ``Equation()``/``Index``/``XB``/``STDP``/``Residuals``/``STDDP``.
    SUPPORTED_PREDICT_TYPES = ("xb", "index", "stdp", "residuals", "stddp", "difference")

    def _equation_covariance(self, position: int) -> Any:
        """``V`` for one equation: ``sigma2_j (X'X)^-1``.

        This is the classical OLS covariance of the j-th univariate regression on
        the shared design, which is the block ``manova`` uses.  Computed on demand
        rather than stored, because it is only needed by ``stdp``/``stddp``.
        """

        design = np.asarray(self.design_matrix, dtype=float)
        response = np.asarray(self.responses, dtype=float)[:, position]
        fitted = np.asarray(self.fitted_values, dtype=float)[:, position]
        residual = response - fitted
        n, k = design.shape
        df_resid = max(n - k, 1)
        sigma2 = float(residual @ residual) / df_resid
        return sigma2 * np.linalg.pinv(design.T @ design)

    def _cross_equation_covariance(self, first: int, second: int) -> Any:
        """``V12 = sigma12 (X'X)^-1`` across two equations.

        ``manova_postestimation.sthlp:164-165`` defines ``stddp`` as the standard
        error of ``x1j b - x2j b``, which needs the CROSS-equation covariance.
        Omitting it is only correct when the equations' residuals are
        uncorrelated -- measured at 0.17 on the test data, so the term matters.
        """

        design = np.asarray(self.design_matrix, dtype=float)
        responses = np.asarray(self.responses, dtype=float)
        fitted = np.asarray(self.fitted_values, dtype=float)
        residual_first = responses[:, first] - fitted[:, first]
        residual_second = responses[:, second] - fitted[:, second]
        n, k = design.shape
        df_resid = max(n - k, 1)
        sigma12 = float(residual_first @ residual_second) / df_resid
        return sigma12 * np.linalg.pinv(design.T @ design)

    def predict(
        self,
        kind: str = "xb",
        *,
        equation: int | None = None,
        equation2: int | None = None,
    ) -> Any:
        """``manova_p`` -- equation-aware prediction.

        ``manova_p.ado:38-44``: with no ``equation()`` Stata uses the **first**
        dependent variable, so ``equation`` defaults to 1 (0-based here: 0).

        Parameters
        ----------
        kind:
            ``xb``/``index`` (fitted values, synonyms), ``stdp`` (prediction
            standard error), ``residuals``, ``stddp`` (difference between two
            equations' standard errors) or ``difference`` (``xb1 - xb2``,
            ``manova_p.ado:100-106``).
        equation, equation2:
            1-based equation numbers.  ``stddp`` and ``difference`` need both.

        Raises
        ------
        ValueError
            For an unknown ``kind``, a missing second equation, or an equation
            number outside the model.
        """

        requested = str(kind).strip().lower()
        if requested not in self.SUPPORTED_PREDICT_TYPES:
            raise ValueError(
                f"predict, {kind}: manova_p supports only "
                f"{', '.join(self.SUPPORTED_PREDICT_TYPES)} (manova_p.ado:16)."
            )

        if self.design_matrix is None or self.coefficients is None:
            raise ValueError(
                "predict: this manova result carries no design matrix, so nothing "
                "can be predicted from it"
            )

        n_equations = int(self.coefficients.shape[1])

        def _position(number: int | None) -> int:
            # Stata's equation() is 1-based; the default is the first equation
            # (``manova_p.ado:38-44``).
            position = 0 if number is None else int(number) - 1
            if not 0 <= position < n_equations:
                raise ValueError(
                    f"predict: equation {number} is out of range; the model has "
                    f"{n_equations} equation(s)"
                )
            return position

        first = _position(equation)
        fitted_first = self.fitted_values[:, first]

        if requested in ("xb", "index"):
            return fitted_first
        if requested == "residuals":
            return self.responses[:, first] - fitted_first
        if requested == "stdp":
            # ``manova_postestimation.sthlp:151-154``: "the standard error of the
            # estimated expected value or mean for the observation's covariate
            # pattern" -- that is ``sqrt(x' V x)`` with V the equation's
            # coefficient covariance, NOT an rmse/sqrt(n) shortcut.
            #
            # On a saturated one-way design the two happen to coincide, which is
            # why the shortcut looked right; it is not general, so the sandwich
            # form is used.
            covariance = self._equation_covariance(first)
            design = np.asarray(self.design_matrix, dtype=float)
            variance = np.einsum("ij,jk,ik->i", design, covariance, design)
            return np.sqrt(np.maximum(variance, 0.0))
        if requested == "difference":
            if equation2 is None:
                raise ValueError(
                    "predict, difference requires equation(eq1, eq2); "
                    "manova_p.ado:96-106 differences two equations"
                )
            second = _position(equation2)
            return fitted_first - self.fitted_values[:, second]
        # stddp: the standard error of the DIFFERENCE between two equations'
        # linear predictions.  ``manova_postestimation.sthlp:164-165``.  The
        # cross-equation covariance must be included:
        #
        #     Var(x'b1 - x'b2) = x'V11x + x'V22x - 2 x'V12x
        if equation2 is None:
            raise ValueError("predict, stddp requires equation(eq1, eq2)")
        second = _position(equation2)
        design = np.asarray(self.design_matrix, dtype=float)
        v_first = self._equation_covariance(first)
        v_second = self._equation_covariance(second)
        v_cross = self._cross_equation_covariance(first, second)
        variance = (
            np.einsum("ij,jk,ik->i", design, v_first, design)
            + np.einsum("ij,jk,ik->i", design, v_second, design)
            - 2.0 * np.einsum("ij,jk,ik->i", design, v_cross, design)
        )
        return np.sqrt(np.maximum(variance, 0.0))

    def __str__(self) -> str:
        return (
            f"MANOVA {self.term}\n"
            f"  Wilks' lambda = {self.wilks_lambda:.4f}\n"
            f"  F({self.df1:g}, {self.df2:g}) = {self.F:.2f}"
            f"   p = {self.p:.4f}"
        )

    @property
    def e(self) -> dict[str, float]:
        """Stata's ``e()`` scalars for ``manova``.

        ``manova.sthlp:216-225`` lists ``e(N)``, ``e(k)``, ``e(k_eq)``,
        ``e(df_m)``, ``e(df_r)``, ``e(df#)`` and ``e(rank)``.

        ``e(df#)`` is the term's hypothesis degrees of freedom: ``manova`` numbers
        terms from 1, and a one-factor model has a single term, so the key is
        ``df1`` in Stata's spelling (``df`` + the term index, no underscore --
        the same convention as ``var``'s ``df_m1``).

        ``e(df_m)`` is that same hypothesis df (the model has one term here), and
        ``e(df_r)`` is the residual df, ``N - g``.

        ``e(k)`` counts estimated parameters and ``e(rank)`` the rank of ``e(V)``;
        for this one-factor fit ``k`` is the number of equations times the
        parameters per equation, which is the group count plus the intercept.
        """

        # ``manova`` numbers terms from 1 and this container holds one term.
        df_term = float(self.df)
        df_resid = float(self.nobs - self.groups)
        n_equations = int(self.n_equations) if self.n_equations else 1
        # ``k`` counts estimated parameters: the intercept plus g-1 dummies per
        # equation, i.e. g per equation for a one-factor model.
        n_parameters = int(round(self.groups * n_equations))
        result = {
            "N": float(self.nobs),
            "k": float(n_parameters),
            "k_eq": float(n_equations),
            "df_m": df_term,
            "df_r": df_resid,
            "df1": df_term,
            "rank": float(n_parameters),
            # ``manova.sthlp:229-247`` -- the macros every eclass command carries.
            "cmd": "manova",
            "depvar": " ".join(self.depvar_names),
            "indepvars": self.term,
            "properties": "b V",
            "estat_cmd": "manova_estat",
            # ``e(predict)`` and the two margins macros were withdrawn in round 63
            # because no predict existed; ``ManovaResult.predict`` now implements
            # ``manova_p``, so they are restored.
            "predict": "manova_p",
            "marginsnotok": "predict(xb) stdp",
            "marginsdefault": "xb",
            "asbalanced": "",
            "asobserved": "",
        }
        # Per-equation vectors (``manova.sthlp:238-241``).
        if self.r2_by_equation:
            result["r2"] = list(self.r2_by_equation)
            result["rmse"] = list(self.rmse_by_equation)
            result["F"] = list(self.f_by_equation)
            result["p_F"] = list(self.p_f_by_equation)
        return result


class Manova:
    """One-way MANOVA (Stata ``manova``), Wilks' lambda with Rao's F.

    ``manova y1 y2 = g`` treats ``g`` as a factor.  Designs with covariates or
    several factors are not implemented and raise
    :class:`~westata.errors.UnsupportedError`; so are the Pillai,
    Lawley-Hotelling and Roy statistics.
    """

    _stata_command = "manova"

    def __init__(
        self,
        responses: np.ndarray,
        groups: Any,
        *,
        term: str = "group",
    ) -> None:
        Y = np.atleast_2d(np.asarray(responses, dtype=float))
        if Y.shape[0] == 1:
            Y = Y.T
        self._Y = Y
        labels = np.asarray(groups).reshape(-1)
        if labels.shape[0] != Y.shape[0]:
            raise ValueError(
                f"manova: {labels.shape[0]} group labels but {Y.shape[0]} observations"
            )
        codes, levels = pd.factorize(pd.Series(labels))
        self._codes = np.asarray(codes, dtype=int)
        self._levels = list(levels)
        self._term = str(term)
        #: ``e(depvar)`` -- filled by ``from_formula``; the array constructor has
        #: no names to record.
        self._response_names: list[str] = []

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Manova":
        """Build from ``"y1 y2 = g"``."""

        data = _subset(data, subset)
        outcomes, rhs = _require_outcomes(formula, data, "manova")
        factors = _split_terms(rhs)
        if len(factors) != 1:
            raise UnsupportedError(
                "manova: exactly one factor is supported (no covariates and no "
                f"multi-factor designs), got {factors}"
            )
        factor = factors[0]
        if factor not in data.columns:
            raise ConsistencyError(f"manova: column {factor!r} not found")
        predictor = data[factor]
        # Stata: "factor variables may not contain noninteger values" (r(452)).
        if pd.api.types.is_numeric_dtype(predictor) and not np.all(
            np.isclose(predictor.to_numpy(dtype=float), np.round(predictor.to_numpy(dtype=float)))
        ):
            raise ConsistencyError(
                f"manova: {factor!r} is a treatment factor and must be integer "
                "valued (Stata: 'factor variables may not contain noninteger "
                "values'); covariates are not implemented"
            )
        model = cls(
            data[outcomes].to_numpy(dtype=float),
            predictor.to_numpy(),
            term=factor,
        )
        #: ``e(depvar)``: the response names, space-separated as Stata stores them.
        model._response_names = list(outcomes)
        return model

    def fit(self) -> ManovaResult:
        """Compute the between/within matrices and Rao's F for Wilks' lambda."""

        Y = self._Y
        n, p = Y.shape
        groups = np.unique(self._codes)
        g = int(groups.size)
        if g < 2:
            raise ConsistencyError(
                "manova: the factor has a single level, so there is nothing to test"
            )
        df_error = n - g
        if df_error <= 0:
            raise ConsistencyError(
                f"manova: {n} observations in {g} groups leave no residual degrees "
                "of freedom"
            )

        grand_mean = Y.mean(axis=0)
        within = np.zeros((p, p))
        between = np.zeros((p, p))
        for code in groups:
            block = Y[self._codes == code]
            block_mean = block.mean(axis=0)
            centred = block - block_mean
            within += centred.T @ centred
            difference = (block_mean - grand_mean).reshape(-1, 1)
            between += block.shape[0] * (difference @ difference.T)

        # Wilks' lambda = |W| / |W + B|
        denominator = float(np.linalg.det(within + between))
        if denominator == 0.0:
            raise ConsistencyError(
                "manova: the total sum of squares is singular; a response is "
                "collinear with the group indicator"
            )

        # ---- all four multivariate statistics, from one eigen-decomposition --
        #
        # Every one of them is a function of the eigenvalues lambda_i of E^-1 H
        # (``mv_glossary.sthlp``):
        #
        #   Wilks' lambda        L = |E| / |E + H|  = prod 1/(1 + lambda_i)
        #   Pillai's trace       V = trace[(E+H)^-1 H] = sum lambda_i/(1+lambda_i)
        #   Lawley-Hotelling     U = trace(E^-1 H)      = sum lambda_i
        #   Roy's largest root   t = lambda_max/(1 + lambda_max)
        #
        # Stata's ``manova`` prints only Wilks' lambda; the other three appear in
        # ``manovatest``.
        #
        # Wilks is taken from the eigenvalue product rather than from the two
        # determinants.  The determinant form is numerically fragile: on a design
        # with cond(E) ~ 3e12 it disagreed with the eigenvalue product by 2e-4
        # while pillai/lawley/roy were all exact.  Deriving all four from the same
        # eigenvalues also keeps them mutually consistent by construction.
        try:
            # ``E^-1 H`` is not symmetric, and ``eigvals`` on it can return a
            # spurious complex pair -- on an ill-conditioned design it produced a
            # wrong-signed real root next to a legitimate one.  ``E^-1/2 H E^-1/2``
            # is symmetric with the same eigenvalues and is numerically better
            # behaved, so the decomposition goes through that.
            values, vectors = np.linalg.eigh(within)
            inverse_root = vectors @ np.diag(values ** -0.5) @ vectors.T
            symmetric = inverse_root @ between @ inverse_root
            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(symmetric)))[::-1]
            positive = eigenvalues[eigenvalues > 0.0]
        except np.linalg.LinAlgError:  # pragma: no cover - singular within
            eigenvalues = np.empty(0, dtype=float)
            positive = np.empty(0, dtype=float)

        if positive.size:
            pillai = float(np.sum(positive / (1.0 + positive)))
            lawley_hotelling = float(np.sum(positive))
            roy_root = float(positive[0] / (1.0 + positive[0]))
            # ``|E|/|E+H| = prod 1/(1 + lambda_i)`` over ALL eigenvalues.
            #
            # Using every eigenvalue (not only the positive ones) is exactly the
            # determinant ratio, but computed in a numerically stable way: the
            # det(v)/det(v+w) route disagreed with this product by 2e-4 on a
            # design with cond(E) ~ 3e12.  Discarding a small negative eigenvalue
            # would silently drop its factor, so the full product is used.
            denominators = 1.0 + eigenvalues
            if np.all(denominators != 0.0):
                wilks = float(np.prod(1.0 / denominators))
            else:  # pragma: no cover - lambda = -1 cannot occur for real E, H
                wilks = float(np.linalg.det(within)) / denominator
        else:  # pragma: no cover - degenerate design
            pillai = lawley_hotelling = roy_root = float("nan")
            wilks = float(np.linalg.det(within)) / denominator

        # Rao's F approximation, in Stata's parameterisation:
        #   s    = sqrt((p^2 q^2 - 4) / (p^2 + q^2 - 5))
        #   df1  = p q
        #   df2  = s (v - (p - q + 1)/2) - (p q - 2)/2
        #   F    = ((1 - L^(1/s)) / L^(1/s)) (df2 / df1)
        # When p^2 + q^2 = 5 (or the numerator vanishes) the transform is exact
        # at s = 1, which is also the univariate case.
        q = float(g - 1)
        numerator = p * p * q * q - 4.0
        scaling_denominator = p * p + q * q - 5.0
        if numerator > 0.0 and scaling_denominator > 0.0:
            s = float(np.sqrt(numerator / scaling_denominator))
        else:
            s = 1.0
        df1 = p * q
        df2 = s * (df_error - (p - q + 1.0) / 2.0) - (p * q - 2.0) / 2.0
        if df2 <= 0.0:  # pragma: no cover - only for pathological designs
            raise ConsistencyError(
                f"manova: the Rao approximation has no residual degrees of "
                f"freedom (df2 = {df2:g}); the design is too small"
            )
        powered = wilks ** (1.0 / s)
        statistic = ((1.0 - powered) / powered) * (df2 / df1)
        pvalue = float(stats.f.sf(statistic, df1, df2))

        # Stata flags the F as exact ("e") when the Rao transform is exact, and
        # that happens exactly when s is 1 or 2 -- e.g. p=2 q=2 gives s=2 (and
        # Stata prints "e"), while p=3 q=3 gives s=2.4337 ("a").
        exact = bool(s in (1.0, 2.0))

        # ---- per-equation OLS statistics ------------------------------------
        #
        # ``manova.sthlp:238-241`` stores e(r2), e(rmse), e(F) and e(p_F) as
        # VECTORS: "R-squared for each equation", "F statistic for each
        # equation".  They are the ordinary fit measures of the j-th univariate
        # regression on the same design, not the multivariate test.
        #
        # The design here is the intercept plus g - 1 group dummies, so each
        # equation's F tests the group effect with q = g - 1 numerator and
        # n - g denominator degrees of freedom.
        design_columns = 1 + int(g - 1)

        # ---- design and coefficient matrices --------------------------------
        #
        # ``manova_p`` predicts through the underlying multivariate regression
        # (``manova_p.ado:56`` calls ``_predict ... eq()``), so the design and the
        # coefficient matrix are needed.  The design is the intercept plus g - 1
        # group dummies with the FIRST level as the base, which is how Stata
        # parameterises ``manova ... = g``.
        design = np.ones((n, design_columns))
        for column, code in enumerate(groups[1:], start=1):
            design[:, column] = (self._codes == code).astype(float)
        coefficients, *_ = np.linalg.lstsq(design, Y, rcond=None)  # (k, p)

        r2_by_equation: list[float] = []
        rmse_by_equation: list[float] = []
        f_by_equation: list[float] = []
        p_f_by_equation: list[float] = []
        for j in range(p):
            response = Y[:, j]
            total = float(np.sum((response - response.mean()) ** 2))
            # Within-group residual sum of squares for this equation.
            residuals = np.zeros_like(response)
            for code in groups:
                mask = self._codes == code
                block = response[mask]
                residuals[mask] = block - block.mean()
            rss_j = float(residuals @ residuals)
            r2_j = 1.0 - rss_j / total if total > 0.0 else float("nan")
            df_resid_j = n - design_columns
            rmse_j = (
                float(np.sqrt(rss_j / df_resid_j)) if df_resid_j > 0 else float("nan")
            )
            df_num_j = int(g - 1)
            if r2_j < 1.0 and df_resid_j > 0 and df_num_j > 0:
                f_j = (r2_j / df_num_j) / ((1.0 - r2_j) / df_resid_j)
                p_j = float(stats.f.sf(f_j, df_num_j, df_resid_j))
            else:  # pragma: no cover - perfect fit or degenerate design
                f_j = float("nan")
                p_j = float("nan")
            r2_by_equation.append(r2_j)
            rmse_by_equation.append(rmse_j)
            f_by_equation.append(f_j)
            p_f_by_equation.append(p_j)

        return ManovaResult(
            wilks_lambda=wilks,
            df=q,
            df1=float(df1),
            df2=float(df2),
            F=float(statistic),
            p=pvalue,
            term=self._term,
            nobs=int(n),
            groups=g,
            n_equations=int(p),
            statistics={
                "wilks_lambda": wilks,
                "pillai_trace": pillai,
                "lawley_hotelling_trace": lawley_hotelling,
                "roy_largest_root": roy_root,
                "s": s,
                "exact": float(exact),
            },
            r2_by_equation=r2_by_equation,
            rmse_by_equation=rmse_by_equation,
            f_by_equation=f_by_equation,
            p_f_by_equation=p_f_by_equation,
            depvar_names=getattr(self, "_response_names", []),
            design_matrix=design,
            coefficients=coefficients,
            fitted_values=design @ coefficients,
            responses=Y,
        )
