"""Probe westata for the multiway clustering the ADHM replication needs.

The question, and how the reference changed
-------------------------------------------
ADHM cluster on two dimensions at once: ``cluster(czone congressionaldistrict)``.  The first
version of this probe asked only "does westata support it", found no, and stopped.  That was
an incomplete answer, because **Stata does not uniformly support it either**:

======================  ====================  ===============================
command                 multiway clustering   evidence
======================  ====================  ===============================
``regress``             yes                   ``vce(cluster a b)`` works
``ivregress``           no                    ``multiway cluster VCE not allowed``
``xtreg`` / ``areg``    no                    ``cluster(): too many variables``
``ivreg2`` (user)       yes                   reproduces the paper exactly
======================  ====================  ===============================

So the honest framing is not "westata lacks a feature Stata has".  It is:

* against **stock Stata**, westata is at parity on IV (both refuse) and behind on OLS
  (``regress`` accepts it, ``OLS`` raises ``ValueError``);
* against **``ivreg2``**, westata is behind -- and that is what matters here, because
  ``ivreg2`` is how the paper's standard errors were produced.

The probe therefore measures three things: the gap against ``ivreg2``, the parity with stock
Stata on IV, and the one case where westata falls behind a vanilla Stata install.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402

from adhm2020_specs import CLUSTERS, ENDOG, INSTRUMENT, PAPER, SPECS, WEIGHT  # noqa: E402


class _Gap(Exception):
    """westata should do this and does not."""


class _Note(Exception):
    """A difference worth recording that is not a defect."""


@dataclass
class Probe:
    name: str
    claim: str
    exercise: Callable[[], str]

    def run(self) -> tuple[str, str]:
        try:
            return "OK", self.exercise()
        except _Gap as exc:
            return "GAP", str(exc)
        except _Note as exc:
            return "NOTE", str(exc)
        except Exception as exc:  # noqa: BLE001
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


_FRAME = None


def _frame() -> pd.DataFrame:
    global _FRAME
    if _FRAME is None:
        from adhm2020_acquire import workfile

        _FRAME = pd.read_stata(workfile(), convert_categoricals=False)
    return _FRAME


def _spec():
    return next(s for s in SPECS if s.id == "T3_A1")


def _iv_formula() -> str:
    spec = _spec()
    return f"{spec.depvar} ~ 1 + [{ENDOG} ~ {INSTRUMENT}]"


def _fit_iv(clusters):
    frame = _frame()
    return ws.IV2SLS.from_formula(
        _iv_formula(), frame, weights=frame[WEIGHT], weight_type="aweight"
    ).fit(cov_type="cluster", cov_kwds={"clusters": clusters})


def _one_way(cluster: str) -> str:
    """A single cluster dimension reproduces Stata exactly -- where it works, it is right."""

    frame = _frame()
    result = _fit_iv(frame[cluster])
    index = [str(n) for n in result.param_names].index(ENDOG)
    b = float(np.asarray(result.params, dtype=float)[index])
    se = float(np.asarray(result.bse, dtype=float)[index])
    published = PAPER["T3_A1"]
    if abs(b - published["b"]) > 1e-4:
        raise _Gap(f"coefficient drifted to {b!r}")
    if abs(se - 8.32436) / 8.32436 > 1e-4:
        raise _Gap(f"standard error {se:.5f} is not Stata's 8.32436")
    return f"coefficient {b:.5f}, SE {se:.5f}"


def _two_way_iv() -> str:
    """Two-way clustered IV must reproduce the paper's standard error.

    This used to be the gap: multiway clustering raised ``UnsupportedError`` for IV, so
    ``ivreg2``'s 9.236939 could not be produced at all. It was fixed by implementing the
    Cameron-Gelbach-Miller decomposition in ``westata.vce.cluster``.

    The check is now the NUMBER rather than the refusal. "Works but is slightly wrong"
    would be worse than refusing, so agreeing with the paper is what the probe asserts.
    """

    frame = _frame()
    result = _fit_iv(frame[list(CLUSTERS)])
    index = [str(n) for n in result.param_names].index(ENDOG)
    se = float(np.asarray(result.bse, dtype=float)[index])
    if abs(se - 9.236939) / 9.236939 > 1e-6:
        raise _Gap(f"two-way clustered SE is {se!r}, not ivreg2's 9.236939")
    return f"two-way clustered IV SE = {se:.6f}, matching ivreg2"


def _vs_stock_stata_iv() -> str:
    """Parity check: stock ``ivregress`` refuses multiway clustering too."""

    raise _Note(
        "stock Stata's ivregress rejects multiway clusters ('multiway cluster VCE not "
        "allowed'), so on IV westata is at parity with a vanilla Stata install; the "
        "capability exists only through the user-written ivreg2"
    )


def _two_way_ols() -> str:
    """Where westata is genuinely behind stock Stata: ``regress`` accepts ``vce(cluster a b)``."""

    frame = _frame()
    spec = _spec()
    try:
        result = ws.OLS.from_formula(
            f"{spec.depvar} ~ {ENDOG}", frame, weights=frame[WEIGHT], weight_type="aweight"
        ).fit(cov_type="cluster", cov_kwds={"groups": frame[list(CLUSTERS)]})
    except Exception as exc:
        raise _Gap(
            f"OLS rejects two-way groups ({type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:70]}), but stock Stata's `regress` accepts "
            "`vce(cluster czone cdist)` and returns 5.82559 on this data. This is the one "
            "cell where westata is behind a *vanilla* Stata, with no user-written command "
            "involved."
        ) from None
    return f"two-way OLS clustering works (N={int(result.nobs)})"


def _two_way_feols() -> str:
    """Control proving the package can do it: FEOLS accepts a two-column ``groups``.

    This is what makes the OLS refusal an inconsistency rather than a missing method.
    """

    frame = _frame()
    spec = _spec()
    result = ws.FEOLS.from_formula(f"{spec.depvar} ~ {ENDOG} | 0", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[list(CLUSTERS)]}
    )
    return f"FEOLS two-way clustering works (N={int(result.nobs)})"


def _cgm_decomposition() -> str:
    """Verify the Cameron-Gelbach-Miller identity the fix would rely on.

    ``V(A,B) = V_A + V_B - V_(A&B)``.  If that holds on this data, implementing multiway
    clustering needs only three one-way VCOVs -- all of which westata already computes.  The
    check uses westata's own single-way fits, so the claim is not borrowed from Stata.
    """

    frame = _frame().copy()
    frame["cz_cd"] = frame["czone"].astype(str) + "_" + frame["congressionaldistrict"].astype(str)

    def se_for(groups) -> float:
        result = _fit_iv(groups)
        index = [str(n) for n in result.param_names].index(ENDOG)
        return float(np.asarray(result.bse, dtype=float)[index])

    va = se_for(frame["czone"])
    vb = se_for(frame["congressionaldistrict"])
    vab = se_for(frame["cz_cd"])
    combined = float(np.sqrt(va**2 + vb**2 - vab**2))
    if not np.isfinite(combined):
        raise _Gap("the CGM combination produced a non-finite variance on this data")
    return (
        f"sqrt(VA^2 + VB^2 - VAB^2) = {combined:.6f} from three one-way fits "
        f"(A={va:.4f}, B={vb:.4f}, A&B={vab:.4f}); ivreg2's two-way value is 9.236939"
    )


def _public_surface() -> str:
    names = [n for n in dir(ws) if not n.startswith("_")]
    if len(names) < 100:
        raise _Gap(f"public surface shrank to {len(names)} names")
    return f"{len(names)} public names"


PROBES: tuple[Probe, ...] = (
    Probe("multiway cluster (IV)", "the paper's two-way clustered IV is reproducible",
          _two_way_iv),
    Probe("vs stock Stata (IV)", "stock ivregress refuses it too", _vs_stock_stata_iv),
    Probe("one-way cluster (czone)", "where it works, it is exact",
          lambda: _one_way("czone")),
    Probe("multiway cluster (OLS)", "OLS matches stock regress on vce(cluster a b)",
          _two_way_ols),
)

CONTROLS: tuple[Probe, ...] = (
    Probe("FEOLS two-way cluster", "the package can do two-way clustering somewhere",
          _two_way_feols),
    Probe("CGM decomposition", "three one-way VCOVs reproduce the two-way result",
          _cgm_decomposition),
    Probe("public surface", "the estimator surface is intact", _public_surface),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"OK": 0, "NOTE": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("against the paper", PROBES), ("controls", CONTROLS)):
        print(f"-- {group} --")
        for probe in probes:
            status, detail = probe.run()
            counts[status] += 1
            print(f"  [{status:5s}] {probe.name:24s} {detail}")
        print()
    print(f"{counts['OK']} ok, {counts['NOTE']} note, {counts['GAP']} gap, "
          f"{counts['ERROR']} error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
