"""Shared tier helpers for the conformance sweep.

Two callers need the same three rules -- the sweep's regression check and the regrade tool -- and
keeping them in one place is what stops them from drifting apart:

  1. which retest rows belong to which registry entry (the explicit mapping);
  2. which rows are measurements rather than diagnostics;
  3. how a row's tier is formed.

Rule 2 is the one that produced the most wrong answers while being built, so it is documented here
rather than rediscovered:

  * PROBE rows document an internal mechanism the sweep investigated -- "var: VAR(1) standard
    errors: internal DOF factor (diagnostic)", "tobit /sigma SE root cause". Their tier describes
    that internal quantity, not the command's output. Counting them demoted `var` to E1 while its
    actual output measured at 1e-14 throughout.
  * ALTERNATIVE rows compare against a mapping that the repair round then CORRECTED --
    "ivregress 2sls vce(robust) vs westata HC1（文档映射）". Their tier is the evidence FOR the fix,
    so counting them penalises the code for having been repaired.
"""

from __future__ import annotations

import re

TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

_PROBE = re.compile(
    r"diagnos|internal|root cause|\[correct|after removing|scale offset"
    r"|no longer inherits|westata unsupported",
    re.I,
)
_ALTERNATIVE = re.compile(r"vs\s+westata|vs\s+FEOLS|文档映射", re.I)

#: Registry entry -> row-name prefixes that exercise it. Written out because five successive
#: heuristics each fixed one case and broke another; the two vocabularies were written
#: independently and collide (`regress, vce(robust)` is one entry but reads like an option;
#: `teffects ra` and `teffects nnmatch` share a word; `xtreg, fe` is one entry whose comma is not a
#: separator).
MAPPING: dict[str, list[str]] = {
    "regress": ["1 regress", "regress basic"],
    "regress, vce(robust)": ["3 regress vce(robust)"],
    "regress, cluster": [
        "4a regress cluster",
        "4b regress cluster",
        "4c regress cluster",
        "4d regress cluster",
    ],
    "regress, i.factor": ["2a regress i.", "2b regress i."],
    "ivregress 2sls": ["10a", "10b", "10f", "ivregress 2sls"],
    "ivregress liml": ["11", "ivregress liml"],
    "ivregress gmm": ["12", "ivregress gmm"],
    "areg": ["7b areg", "7d areg"],
    "reghdfe": ["8c reghdfe", "8d reghdfe", "8 reghdfe"],
    "eivreg": ["13 eivreg", "15f eivreg"],
    "xtreg, fe": ["xtreg, fe"],
    "xtreg, re": ["xtreg, re"],
    "xtreg, be": ["xtreg, be"],
    "xtreg, fd": ["xtreg, fd"],
    "xtlogit, fe": ["xtlogit, fe"],
    "xtlogit, pa": ["xtlogit, pa"],
    "xtprobit, re": ["xtprobit, re"],
    "xttobit": ["xttobit"],
    "xtpoisson, fe": ["xtpoisson, fe"],
    "xtpoisson, re": ["xtpoisson, re"],
    "xtnbreg, fe": ["xtnbreg, fe"],
    "xtivreg, fe": ["xtivreg, fe"],
    "xtivreg, re": ["xtivreg, re"],
    "xtgls": ["xtgls"],
    "xtpcse": ["xtpcse"],
    "xtabond": ["xtabond"],
    "logit": ["logit ("],
    "probit": ["probit ("],
    "poisson": ["poisson ("],
    "nbreg": ["nbreg ("],
    "zip": ["zip,"],
    "zinb": ["zinb,"],
    "ztp": ["ztp ("],
    "tnbreg": ["tnbreg ("],
    "ologit": ["ologit ("],
    "oprobit": ["oprobit ("],
    "tobit": ["tobit ll(0)", "tobit ul(2.5)", "tobit ll(0) ul(2)"],
    "truncreg": ["truncreg "],
    "intreg": ["intreg "],
    "heckman": ["heckman twostep"],
    "etregress": ["etregress twostep"],
    "qreg": ["qreg median", "qreg quantile"],
    "qreg, vce(robust)": ["qreg vce(robust)"],
    "rreg": ["rreg"],
    "bsqreg": ["bsqreg"],
    "sqreg": ["sqreg"],
    "frontier": ["frontier"],
    "sureg": ["sureg"],
    "mvreg": ["mvreg"],
    "manova": ["manova"],
    "nl": ["nl "],
    "lincom": ["lincom"],
    "nlcom": ["nlcom"],
    "testnl": ["testnl"],
    "lrtest": ["lrtest"],
    "hausman": ["hausman"],
    "contrast": ["contrast"],
    "pwcompare": ["pwcompare"],
    "margins": ["margins"],
    "predict": ["predict "],
    "predict, stdp/stdf": ["predict regress: stdp", "predict regress: stdf"],
    "teffects ra": ["teffects ra"],
    "teffects ipw": ["teffects ipw"],
    "teffects aipw": ["teffects aipw"],
    "teffects nnmatch": ["teffects nnmatch"],
    "didregress": ["didregress"],
    "xtdidregress": ["xtdidregress"],
    "svyset / svy: mean": ["svy: mean"],
    "svy: regress": ["svy: regress"],
    "svy: total": ["svy: total"],
    "svy: proportion": ["svy: proportion"],
    "svy: logit": ["svy: logit"],
    "jackknife": ["jackknife"],
    "bootstrap": ["bootstrap"],
    "bootstrap / permute expressions (exp_list)": ["bootstrap:"],
    "permute": ["permute:"],
    "arima": ["arima"],
    "var": ["var:"],
    "vecrank (Johansen)": ["vecrank:"],
    "dfuller": ["dfuller"],
    "pperron / kpss": ["pperron", "kpss"],
    "arch": ["arch:"],
    "xtunitroot ips": ["xtunitroot ips"],
    "xtunitroot llc": ["xtunitroot llc"],
}


def is_measurement(row: dict) -> bool:
    """True when a row grades the command's OUTPUT rather than a diagnostic or an old mapping."""

    name = row.get("name") or ""
    return not _PROBE.search(name) and not _ALTERNATIVE.search(name)


def row_tier(row: dict) -> str | None:
    """A row's tier: the worse of its coefficient tier and its standard-error tier.

    Grading on coefficients alone is how `teffects nnmatch` would be promoted while its standard
    errors sit 5-16% away -- the exact error this sweep exists to catch.
    """

    tiers = [row.get("param_tier"), row.get("se_tier")]
    usable = [tier for tier in tiers if tier in TIER_RANK]
    return max(usable, key=lambda tier: TIER_RANK[tier]) if usable else None


def entry_rows(aggregate: dict, entry_command: str) -> list[tuple[dict, str | None]]:
    """Measurement rows attributed to a registry entry, with their tiers."""

    prefixes = MAPPING.get(entry_command)
    if not prefixes:
        return []
    found: list[tuple[dict, str | None]] = []
    for entries in aggregate.get("domains", {}).values():
        for row in entries:
            if not is_measurement(row):
                continue
            name = row.get("name") or ""
            if any(name.startswith(prefix) for prefix in prefixes):
                found.append((row, row_tier(row)))
    return found
