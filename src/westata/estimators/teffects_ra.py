"""Regression adjustment treatment effects: Stata's ``teffects ra``.

``teffects`` has readable ``ado`` in ``ado/base/t/teffects.ado``, but the estimators themselves
are builtins, so the numbers were pinned by measurement rather than transcription.

Reference on a synthetic sample (n = 500, true effect 1.5)::

    teffects ra (y age educ) (treat)
        ATE        1.592426   .2430606     z 6.55
        POmean(0)  44.25704   .3107156
        POmean(1)  45.84947   .232849

    teffects ra (y age educ) (treat), atet
        ATET       1.635659   .2765799

**What is reproduced exactly**: every point estimate. Regression adjustment fits one outcome
regression per treatment arm and averages the predicted counterfactuals:

    PO(j) = mean_i  x_i' b_j
    ATE   = mean_i (x_i' b_1 - x_i' b_0)
    ATET  = mean_{i: t_i = 1} (x_i' b_1 - x_i' b_0)

which gives 1.592426 / 44.257040 / 45.849466 / 1.635659 against Stata's 1.592426 / 44.25704 /
45.84947 / 1.635659.

**What is NOT reproduced**: the robust standard errors.

Six influence-function forms were tried against three different samples, with Stata's value as
the target each time:

===========================================  =========================  ==========
form                                          result on seed 20261105   ratio
===========================================  =========================  ==========
fitted counterfactuals treated as data        0.0146934                  0.060
outcome scores + coefficient terms            0.1827790                  0.752
  same, on seeds 11 and 22                                              0.814 / 0.761
the above plus a heuristic p-hat term         0.2436296                  1.002
full GMM with the treatment-proportion moment 0.6288181                  2.587
  same, on seeds 11 and 22                                              3.005 / 2.860
===========================================  =========================  ==========

The third row is the interesting one: adding ``-(t/p - (1-t)/(1-p)) * ATE`` to the influence
function lands within 0.2% on the first seed, but 2% high on seed 11 and 2.5% low on seed 22.
A term that is right on one sample and wrong in both directions on others is a coincidence, not
a derivation -- so it is not shipped.

The right place to look is ``_teffects_ra_gmm.ado``, which IS readable -- but reading it in
round 68 shows it only SETS UP the GMM system and calls Stata's ``gmm`` command. The file is
177 lines with no occurrence of influence/variance/psi/robust/score, so it contains no variance
code at all. The variance is produced inside the builtin ``gmm``: the same kind of boundary as
``rdrobust_vce``, where the estimator is visible and the sandwich is not.

``se``/``z``/``p``/the interval therefore stay NaN, and a test pins them that way -- the same
call made for ``rdrobust``.

Round 68 re-measured the gap on nsw_dw.dta (8 covariates, ``teffects ra (re78 ...) (treat)``)
so the limitation stays quantified rather than becoming folklore:

    Stata e(b)[1,1]        = 1621.583081895851   this package: 1621.5830818958466
    Stata sqrt(e(V)[1,1])  =   679.054055509282

Every influence-function variant tried lands 2.9-3.4% LOW: the plain AIPW form gives 656.90
(-3.26%), adding Var(beta) 657.66 (-3.16%), adding the E[X|T] estimation term 659.51 (-2.88%).
Three independent refinements all move toward the target and stop short of it, which says a
term is still missing rather than mis-scaled -- so none is shipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["TefectsRaResult", "teffects_ra"]


@dataclass(frozen=True)
class TefectsRaResult:
    """``teffects ra``: regression-adjustment treatment effects."""

    ate: float
    po_0: float
    po_1: float
    atet: float
    nobs: int
    treated: int
    estimate: float
    estimand: str
    se: float = float("nan")
    z: float = float("nan")
    pvalue: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    variance_matches_stata: bool = False

    def __str__(self) -> str:
        lines = [
            "Treatment-effects estimation",
            f"  Estimator      : regression adjustment",
            f"  Outcome model  : linear",
            f"  Treatment model: none",
            f"  Number of obs  = {self.nobs}",
            "",
            f"{'':<12}{'Coefficient':>14}{'std. err.':>13}{'z':>8}{'P>|z|':>8}"
            f"{'[95% conf. interval]':>26}",
            "-" * 82,
            f"{self.estimand:<12}{self.estimate:>14.6f}"
            + (f"{self.se:>13.7f}{self.z:>8.2f}{self.pvalue:>8.3f}"
               f"{f'[{self.ci_lower:.6f}, {self.ci_upper:.6f}]':>26}"
               if np.isfinite(self.se) else f"{'(variance not reproduced)':>55}"),
            "-" * 82,
            f"{'POmean':<12}",
            f"{'treat=0':<12}{self.po_0:>14.6f}",
            f"{'treat=1':<12}{self.po_1:>14.6f}",
        ]
        return "\n".join(lines)


def teffects_ra(
    response: Any,
    treatment: Any,
    covariates: Any,
    *,
    estimand: str = "ate",
    add_constant: bool = True,
) -> TefectsRaResult:
    """Regression-adjustment treatment effect, as ``teffects ra (y x) (t)``.

    ``covariates`` is an ``(n, k)`` array; a constant column is prepended unless
    ``add_constant`` is False.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not (y.size == d.size == x.shape[0]):
        raise ConsistencyError(
            f"teffects ra: shapes disagree -- y {y.size}, treatment {d.size}, "
            f"covariates {x.shape}"
        )
    keep = np.isfinite(y) & np.isfinite(d) & np.all(np.isfinite(x), axis=1)
    y, d, x = y[keep], d[keep], x[keep]

    levels = np.unique(d)
    if levels.size != 2 or not np.array_equal(levels, np.array([0.0, 1.0])):
        raise ConsistencyError(
            f"teffects ra: treatment must take values 0 and 1; found {levels.tolist()}"
        )
    key = str(estimand).strip().lower()
    if key not in {"ate", "atet"}:
        raise UnsupportedError(
            f"teffects ra: only ate and atet are implemented; got estimand={estimand!r}"
        )

    if add_constant:
        x = np.column_stack([np.ones(x.shape[0]), x])

    treated = d > 0.5
    control = ~treated
    if int(treated.sum()) <= x.shape[1] or int(control.sum()) <= x.shape[1]:
        raise ConsistencyError(
            "teffects ra: an arm has fewer observations than covariates; the outcome "
            "regression for that arm is not identified"
        )

    def fit(mask: np.ndarray) -> np.ndarray:
        coefficients, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
        return coefficients

    beta_1 = fit(treated)
    beta_0 = fit(control)
    mu_1 = x @ beta_1
    mu_0 = x @ beta_0

    po_1 = float(mu_1.mean())
    po_0 = float(mu_0.mean())
    ate = float(po_1 - po_0)
    atet = float((mu_1 - mu_0)[treated].mean())
    estimate = ate if key == "ate" else atet

    return TefectsRaResult(
        ate=ate,
        po_0=po_0,
        po_1=po_1,
        atet=atet,
        nobs=int(y.size),
        treated=int(treated.sum()),
        estimate=estimate,
        estimand="ATE" if key == "ate" else "ATET",
    )
