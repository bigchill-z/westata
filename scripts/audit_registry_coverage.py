"""Audit: every estimation command in the conformance registry must either be covered by
the layout scan or have an explicit, checkable reason for not being covered.

This is the check I should have written at the start. Twice I reported a completion figure
that was measured against the wrong denominator -- once because I used the size of
_HEADER_BLOCK (53) as if it were the registry, once because three E3 commands had no scan
spec at all. This script makes that failure visible instead of trusting a summary line.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml

import westata as ws
from scan_all_layouts import _specs

#: Commands that take no estimation table. Each entry states WHY, and the claim is one a
#: reader can check in the package (the class/return type it names really is what runs).
NO_TABLE = {
    "test": "post-estimation hypothesis test: returns r(), no table",
    "lincom": "post-estimation linear combination: returns r(), no table",
    "nlcom": "post-estimation nonlinear combination: returns r(), no table",
    "testnl": "post-estimation hypothesis test: returns r(), no table",
    "lrtest": "post-estimation LR test: returns r(), no table",
    "predict": "generates variables: no table",
    "predict, stdp/stdf": "generates variables: no table",
    "margins": "post-estimation margins: returns r(), no table",
    "contrast": "NOT IMPLEMENTED in the package",
    "pwcompare": "NOT IMPLEMENTED in the package",
    "hausman": "NOT IMPLEMENTED in the package",
    "teffects ra": "NOT IMPLEMENTED in the package",
    "teffects ipw": "NOT IMPLEMENTED in the package",
    "teffects aipw": "NOT IMPLEMENTED in the package",
    "teffects nnmatch": "NOT IMPLEMENTED in the package",
    "var": "NOT IMPLEMENTED in the package",
    "dfuller": "NOT IMPLEMENTED in the package",
    "vecrank (Johansen)": "NOT IMPLEMENTED in the package",
    "vec (VECM estimation)": "NOT IMPLEMENTED in the package",
    "pperron / kpss": "NOT IMPLEMENTED in the package",
    "jackknife": (
        "prefix command: re-runs a wrapped estimation. The package exposes a Jackknife "
        "HELPER CLASS that takes an already-built design, not a jackknife: prefix, so "
        "there is no table of its own to compare"
    ),
    "bootstrap": "prefix command: re-runs a wrapped estimation",
    "permute": "prefix command: re-runs a wrapped estimation",
    "bootstrap / permute expressions (exp_list)": "prefix command",
    "etable": "renders an already-fitted model: no table of its own",
    "estat szroeter": "post-estimation: returns r()",
    "estat esize": "post-estimation: returns r()",
    "estat moran": "post-estimation: returns r()",
    "estat auc": "post-estimation: returns r()",
    "estat weakrobust": "post-estimation: returns r()",
    "estat mundlak": "post-estimation: returns r()",
    # estat hettest prints three lines of RESULT TEXT ("H0: Constant variance" /
    # "chi2(1) = 1.60" / "Prob > chi2 = 0.2063"), not a regression table, so it is outside
    # a TABLE-layout sweep. Its numbers were cross-checked anyway: westata returns
    # statistic 1.6 / p 0.2063 against Stata's "chi2(1) = 1.60" / "Prob > chi2 = 0.2063".
    "estat hettest": "prints result text, not a table; statistic and p verified equal",
    # r-class tests: the package returns a statistic, not a table, by design.
    "xtunitroot ips": "r-class test: returns PanelUnitRootResult (no table); stats verified",
    "xtunitroot llc": "r-class test: returns PanelUnitRootResult (no table); stats verified",
    "svyset / svy: mean": "covered as svy_mean",
    "svyset fpc / singleunit": "covered as svy_mean / svy_total variants",
    "regress, vce(robust)": "covered as regress_robust",
    "regress, cluster": "covered as regress_cluster",
    "regress, i.factor": "covered by regress (factor expansion not a layout difference)",
    "qreg, vce(robust)": "covered by qreg (banner differs only)",
}

registry = yaml.safe_load(open(ROOT / "tests/conformance/registry.yml", encoding="utf-8"))
specs = _specs()

# Map each scan spec to the commands it exercises: label + the Stata command it runs.
covered_stata = set()
for _label, _command, _build in specs:
    covered_stata.add(_command)
    covered_stata.add(_command.split()[0])

print(f"scan specs:                  {len(specs)}")
print(f"registry entries:            {len(registry)}")
unknown, covered = [], []
for entry in registry:
    cmd = entry["command"]
    head = cmd.split()[0].rstrip(",")
    if any(cmd == s or head == s or cmd.startswith(s) for s in covered_stata):
        covered.append(cmd)
    elif cmd in NO_TABLE:
        pass
    else:
        unknown.append((cmd, entry["level"]))

print(f"covered by the scan:         {len(covered)}")
print(f"explicitly out of scope:     {len(registry) - len(covered) - len(unknown)}")
print(f"UNACCOUNTED FOR:             {len(unknown)}")
for cmd, level in unknown:
    print(f"    {level}  {cmd}")

# Non-zero when something is unaccounted for, so this can act as a guard in CI rather than
# a report nobody reads. A registry entry must be either exercised by the layout scan or
# listed in NO_TABLE with a reason.
raise SystemExit(1 if unknown else 0)
