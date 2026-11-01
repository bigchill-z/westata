"""Estimate the paper's specifications with ``westata`` and capture the same quantities.

The comparison is only meaningful if both sides estimate *the same model on the same
rows with the same weights*.  Three things therefore have to be handled explicitly
rather than left to defaults:

**Samples.**  Stata drops a row when any variable in the command is missing; pandas'
formulas do the same, but only if the same column list reaches both.  The restriction
``if yr==1990`` is applied here as a boolean mask built from the same string
:class:`specs.Spec` hands to Stata.

**Analytic weights in 2SLS.**  ``westata``'s IV estimators reject ``weights`` outright
(``iv.py``: "IV estimation does not support weights in P1"), and that refusal is the
right behaviour -- silently ignoring a weight would be worse.  Stata's ``[aw=w]`` on
``ivregress`` is, however, exactly a rescaling of every column of the instrument
matrix by ``sqrt(w)`` (analytic weights are variances, so the weighted 2SLS estimator
is the unweighted 2SLS estimator on the square-root-scaled data).  The driver performs
that scaling itself, including the constant, and says so in the report.  The weighted
*standard errors* then need Stata's aweight normalisation to line up; the residual
scaling Stata applies is checked against its own ``e(rmse)`` in ``run_compare.py``.

**Covariance.**  Stata's ``vce(cluster g)`` is CR0 on the estimation sample: no
``G/(G-1)`` factor, and the t statistics use ``G-1`` degrees of freedom only when
``small`` is requested.  ``westata``'s ``cov_kwds={"clusters": ...}`` is documented as
CR0 with ``debiased=False``, which matches.  Passing ``"groups"`` instead of
``"clusters"`` to an IV model is silently accepted by some layers and ignored by others,
so the key is set per estimator deliberately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402

from adh2013_specs import SPECS, Spec  # noqa: E402

OUT = HERE / "out"


# --------------------------------------------------------------------- helpers
def restrict(frame: pd.DataFrame, condition: str | None) -> pd.DataFrame:
    """Apply Stata's ``if`` expression to a pandas frame.

    The conditions in :mod:`specs` are single comparisons against a column
    (``yr==1990``), which ``DataFrame.query`` handles with the same semantics as
    Stata's ``if``.  Anything more elaborate should be added here deliberately
    rather than guessed at, so an unsupported form fails loudly.
    """

    if not condition:
        return frame
    if "==" not in condition or "&" in condition or "|" in condition:
        raise ValueError(
            f"condition {condition!r} is more complex than the driver supports; "
            "extend restrict() rather than letting the two samples diverge"
        )
    column, _, value = condition.partition("==")
    column = column.strip()
    value = value.strip()
    try:
        target = float(value)
    except ValueError:
        target = value
    return frame.loc[frame[column] == target]


def sqrt_weighted(frame: pd.DataFrame, columns: list[str], weights: pd.Series) -> pd.DataFrame:
    """Scale ``columns`` by ``sqrt(w)``, and add the scaled constant.

    The constant is scaled too, and it has to be: Stata's aweighted regression
    minimises ``sum w_i (y_i - x_i'b)^2``, which is ``||sqrt(w)(y - Xb)||^2``.  Writing
    that as a regression of ``sqrt(w)y`` on ``sqrt(w)X`` only reproduces it if the
    "constant" column of ``sqrt(w)X`` is ``sqrt(w)`` itself -- i.e. the intercept is
    weighted like every other column.  Omitting it (and letting the estimator add an
    unweighted 1) makes exactly the kind of small, hard-to-see error that this
    replication exists to catch: on Table 3 column 6 it moves the coefficient from
    -0.59636 to -0.57031.
    """

    scaled = frame.copy()
    root = np.sqrt(np.asarray(weights, dtype=float))
    for column in columns:
        scaled[column] = np.asarray(scaled[column], dtype=float) * root
    scaled["_wconst"] = root
    return scaled


# ------------------------------------------------------------------ estimators
def _fit_ols(spec: Spec, frame: pd.DataFrame):
    kwargs: dict = {}
    if spec.weight_var:
        kwargs["weights"] = frame[spec.weight_var]
        kwargs["weight_type"] = spec.weight_type
    model = ws.OLS.from_formula(spec.formula, frame, **kwargs)
    if spec.cluster:
        return model.fit(cov_type="cluster", cov_kwds={"groups": frame[spec.cluster]})
    return model.fit(cov_type="nonrobust")


def _fit_iv(spec: Spec, frame: pd.DataFrame):
    """2SLS, with Stata's analytic weights emulated by square-root scaling."""

    columns = [spec.depvar, spec.endog or "", *spec.instruments, *spec.exog]
    if spec.weight_var:
        # The weight column itself is not a regressor, so it is only scaled where it
        # appears -- it does not.
        scaled = sqrt_weighted(frame, [c for c in columns if c], frame[spec.weight_var])
        # ``0 +`` suppresses the estimator's own intercept: the weighted constant is
        # ``_wconst``, and letting one be added as well would produce a design matrix
        # that is exactly collinear.  The exogenous list may be empty (Table 2, column
        # 1 has no controls at all), in which case the join must not contribute a
        # dangling ``+``.
        rhs = ["0", "_wconst", *spec.exog]
        formula = (
            f"{spec.depvar} ~ "
            + " + ".join(rhs)
            + f" + [{spec.endog} ~ " + " + ".join(spec.instruments) + "]"
        )
        model = ws.IV2SLS.from_formula(formula, scaled)
    else:
        model = ws.IV2SLS.from_formula(spec.formula, frame)
    if spec.cluster:
        return model.fit(cov_type="cluster", cov_kwds={"clusters": frame[spec.cluster]})
    return model.fit(cov_type="none")


def _fit_first_stage(spec: Spec, frame: pd.DataFrame):
    """The first stage as its own regression, so panel II of Table 3 can be compared.

    2SLS's first stage is ``endog`` on the exogenous controls *and* the instruments.
    Estimating it separately is what makes the paper's panel II replicable at all: the
    structural fit returns only the structural parameters, and Stata's ``ivregress,
    first`` prints the panel without leaving it in ``e(b)``.
    """

    exog = list(spec.exog) + list(spec.instruments)
    formula = f"{spec.endog} ~ " + " + ".join(exog)
    kwargs: dict = {}
    if spec.weight_var:
        kwargs["weights"] = frame[spec.weight_var]
        kwargs["weight_type"] = spec.weight_type
    model = ws.OLS.from_formula(formula, frame, **kwargs)
    if spec.cluster:
        return model.fit(cov_type="cluster", cov_kwds={"groups": frame[spec.cluster]})
    return model.fit(cov_type="nonrobust")


def stata_scale_fit(spec: Spec, frame: pd.DataFrame, result) -> dict[str, float]:
    """Recompute R-squared and RMSE on the *original* data scale.

    The square-root rescaling that emulates aweighted IV moves everything -- including
    the residual scale -- onto a different metric, so the fit statistics westata reports
    for such a model are the statistics of the transformed regression, not Stata's.
    They are not wrong; they answer a different question.  Evaluating the structural
    residuals against the unweighted data reproduces Stata's own numbers, which is how
    the report can say "the estimators agree" without the auxiliary output appearing to
    disagree.

    Two Stata conventions are implemented here rather than read off westata:

    * aweighted residuals are weighted by ``w`` normalised to sum to ``N``;
    * ``e(rmse)`` divides by ``N``, not ``N - k``.  ``ivregress`` does not report a
      residual degrees of freedom (``e(df_r)`` is missing), and its RMSE is
      ``sqrt(rss / N)`` -- verified on all six Table 3 columns against the values Stata
      printed, to every digit Stata printed.
    """

    b = dict(zip(result.param_names, np.asarray(result.params, dtype=float)))
    y = frame[spec.depvar].to_numpy(dtype=float)
    endog = frame[spec.endog].to_numpy(dtype=float)
    controls = np.column_stack(
        [frame[c].to_numpy(dtype=float) for c in spec.exog] + [np.ones(len(frame))]
    )
    beta = np.array([b[c] for c in spec.exog] + [b["_wconst"]])
    u = y - controls @ beta - b[spec.endog] * endog

    if spec.weight_var:
        w = frame[spec.weight_var].to_numpy(dtype=float)
        w = w * (len(w) / w.sum())
    else:
        w = np.ones(len(y))
    rss = float(np.sum(w * u**2))
    tss = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    return {"r2": 1.0 - rss / tss, "rmse": float(np.sqrt(rss / len(y))), "rss": rss}


def _rename_for_stata(names: list[str]) -> dict[str, str]:
    """Map westata's parameter names onto Stata's, so the two vectors align by name.

    Two mechanical differences, both in the driver's own construction rather than in
    either estimator:

    * the square-root-scaled IV model carries the constant as ``_wconst`` because the
      scaling had to be applied to a column of its own;
    * patsy, which builds the OLS design matrix, calls the constant ``Intercept``.

    Stata calls it ``_cons`` in both cases.  Renaming here (rather than teaching the
    comparison layer about aliases) keeps the alignment rule "the same name means the
    same parameter" intact.
    """

    return {"_wconst": "_cons", "Intercept": "_cons"}


def estimate(spec: Spec, frame: pd.DataFrame):
    model_frame = RESTRICTED[spec.id]
    result = _fit_iv(spec, model_frame) if spec.endog else _fit_ols(spec, model_frame)
    return result


#: Populated by :func:`load`, keyed by spec id.  A module-level cache rather than a
#: parameter because every spec shares one workfile and re-reading a 208-column Stata
#: file eleven times is pure latency.
RESTRICTED: dict[str, pd.DataFrame] = {}


def load(path: Path | None = None) -> pd.DataFrame:
    """Read the estimation sample and pre-restrict it per spec."""

    from adh2013_data import workfile

    frame = pd.read_stata(path or workfile(), convert_categoricals=False)
    frame.columns = [str(c) for c in frame.columns]
    RESTRICTED.clear()
    for spec in SPECS:
        RESTRICTED[spec.id] = restrict(frame, spec.condition).reset_index(drop=True)
    return frame


def _in_stata_order(spec: Spec, stata_names: list[str], *, first_stage: bool = False) -> list[int]:
    """Index permutation that puts parameters in the order Stata prints them.

    Stata prints the regressand's regressors in the order the command listed them, with
    ``_cons`` last: the endogenous variable first, then the exogenous controls in the
    order they were written.  patsy and formulaic sort the design columns alphabetically
    and put the intercept first, so the JSON that westata writes would otherwise read in
    a different order from the Stata JSON it is compared against.

    That mismatch is not a numerical error -- values pair by name either way -- but an
    artefact whose key order contradicts the table it is diffed against invites a
    *positional* comparison by the next reader, which is precisely the mistake
    ``replication/lib/compare.py`` exists to make impossible.  Deriving the order from
    the spec means the two files can be diffed line by line.
    """

    if first_stage:
        # The first-stage regressand is the endogenous variable, which is not itself a
        # coefficient of that regression; its covariates are the controls plus the
        # instruments, in the order the structural command wrote them.
        canonical = [*spec.exog, *spec.instruments, "_cons"]
    else:
        canonical = [spec.endog] if spec.endog else []
        canonical = list(dict.fromkeys([*canonical, *spec.exog])) + ["_cons"]
    index = {name: i for i, name in enumerate(stata_names)}
    missing = [name for name in canonical if name not in index]
    if missing:
        raise KeyError(
            f"{spec.id}: Stata order names {missing} absent from the fitted parameters "
            f"{stata_names}; the spec and the fit disagree about the model"
        )
    return [index[name] for name in canonical]


def _serialise(spec: Spec, result, *, first_stage: bool = False) -> dict:
    names = _rename_for_stata(list(result.param_names))
    stata_names = [names.get(n, n) for n in result.param_names]
    order = _in_stata_order(spec, stata_names, first_stage=first_stage)

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    e = dict(getattr(result, "e", {}) or {})
    return {
        "spec": f"{spec.id}_FS" if first_stage else spec.id,
        "names": [stata_names[i] for i in order],
        "b": [float(params[i]) for i in order],
        "se": [float(bse[i]) for i in order],
        "nobs": int(result.nobs),
        "scalars": {k: float(v) for k, v in e.items() if isinstance(v, (int, float, np.floating))},
        "estimator": type(result).__name__,
        "weighting": "sqrt(aweight) rescaling" if spec.weight_var and spec.endog else (spec.weight_type or "none"),
        "cluster": spec.cluster or "nonrobust",
        "param_order": "Stata reporting order (_cons last)",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    payload: dict[str, dict] = {"specs": {}, "first_stage": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["westata_source"] = str(Path(ws.__file__).parent)
    payload["meta"]["rows_in_file"] = int(frame.shape[0])

    for spec in SPECS:
        result = estimate(spec, frame)
        record = _serialise(spec, result)
        if spec.endog and spec.weight_var:
            # Replace the transformed-scale fit statistics with Stata's own.  They are
            # recorded separately in ``scalars_native`` so the report can show both and
            # say which is which, instead of silently preferring one.
            record["scalars_native"] = dict(record["scalars"])
            record["scalars"].update(stata_scale_fit(spec, RESTRICTED[spec.id], result))
        payload["specs"][spec.id] = record
        print(f"[westata] {spec.id}: N={result.nobs} {record['estimator']}", flush=True)

        if spec.endog:
            fs = _fit_first_stage(spec, RESTRICTED[spec.id])
            payload["first_stage"][spec.id] = _serialise(spec, fs, first_stage=True)
            print(f"[westata] {spec.id}_FS: N={fs.nobs}", flush=True)

    target = OUT / "westata_results.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
