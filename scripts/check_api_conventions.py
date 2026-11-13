"""Check the public API against ``docs/API_CONVENTIONS.md``.

This is the mechanical half of the convention document: every rule below has an
``R`` number that matches a section of that file, so a failure here points at the
paragraph that explains why it is a rule.

It reports, it does not fix, and it is deliberately NOT wired into CI yet: the
0.3.11 tree violates several rules, and the burn-down order is in the document's
"迁移路线" section.  Once the baseline reaches zero, ``--strict`` goes into
``.github/workflows/ci.yml``.

Usage::

    python scripts/check_api_conventions.py            # human report, exit 0
    python scripts/check_api_conventions.py --strict   # exit 1 if anything fires
    python scripts/check_api_conventions.py --rule R3  # one family only
    python scripts/check_api_conventions.py --json     # machine-readable

Every reported item carries ``file:line`` so the report is actionable without a
second search.  Findings that are DELIBERATE are listed in ``ALLOWED`` with the
paragraph that allows them; they are printed as ``allowed`` and never fail
``--strict``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import pathlib
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: The import root: the directory that CONTAINS the package (R5.4).
SRC_ROOT = ROOT / "src"
#: The package directory, walked for source.
SRC = SRC_ROOT / "westata"

# Rules whose findings are known-deliberate.  Key is (rule, subject); the value is
# the reason, quoted from the convention document.  Anything not listed fails.
ALLOWED: dict[tuple[str, str], str] = {
    # R4.3: the statsmodels compatibility adapters return statsmodels shapes.
    ("R4.3", "ContrastResults.conf_int"): "statsmodels adapter: ndarray is the statsmodels shape",
    ("R4.3", "VECMResults.conf_int"): "statsmodels adapter: ndarray is the statsmodels shape",
    # R4.1: these three ARE statsmodels-shaped result objects; R4.2 covers them and a
    # ``to_dict`` would be a second, divergent serialisation of the same numbers.
    ("R4.1", "ContrastResults"): "statsmodels adapter; accessors are the statsmodels surface",
    ("R4.1", "VECMResults"): "statsmodels adapter; accessors are the statsmodels surface",
    ("R4.1", "VARResults"): "statsmodels adapter; accessors are the statsmodels surface",
    ("R4.1", "StoredResults"): "e()/r() scalar mapping, consumed by parse_exp_list",
    # R4.2: these two intentionally mirror statsmodels' `ContrastResults`, whose surface
    # is NOT the estimation-result surface (no params/llf/rsquared).  They are exempt;
    # PraisResult is not, because it presents itself as an estimation result.
    ("R4.2", "ContrastResults"): "mirrors statsmodels ContrastResults, not an estimation result",
    ("R4.2", "VECMResults"): "mirrors statsmodels VECMResults; subset of the statsmodels surface",
    # R4.2: `VARResults.params` is Stata's EQUATION BY EQUATION coefficient TABLE
    # (a DataFrame indexed [const, L1.y1, L1.y2] with one column per equation), which
    # is the shape `var` prints.  `ResultsProtocol` wants a flat 1-D `params`, and the
    # two cannot both win: flattening would destroy the table every existing VAR
    # accessor and test relies on.  A multi-equation model has no single coefficient
    # vector, so the protocol -- written for single-equation fits -- does not apply.
    ("R4.2", "VARResults"): "multi-equation: params is Stata's table, not a flat vector",
    # R2.5: the linear-algebra layer speaks mathematical notation on purpose.
    ("R2.5", "statsmodels_wald_test.R"): "statsmodels shape argument",
    ("R2.5", "run_wald_test.R"): "statsmodels shape argument",
    ("R2.5", "describe_constraint.R"): "statsmodels shape argument",
    ("R2.5", "wald_test.R"): "statsmodels shape argument",
    ("R2.5", "run_wald_test.r"): "statsmodels shape argument",
    ("R2.5", "WaldTest.wald_test.R"): "statsmodels shape argument",
    ("R2.5", "cov_cluster.X"): "linear-algebra layer: X/resid are the documented names",
    ("R2.5", "cov_cluster.resid"): "linear-algebra layer",
    ("R2.5", "cov_hc.Xw"): "linear-algebra layer",
    ("R2.5", "cov_hc.resid_w"): "linear-algebra layer",
    ("R2.5", "cov_hac.Xw"): "linear-algebra layer",
    ("R2.5", "cov_hac.resid_w"): "linear-algebra layer",
    ("R2.5", "hac_meat.Xw"): "linear-algebra layer",
    ("R2.5", "hac_meat.resid_w"): "linear-algebra layer",
    ("R2.5", "bread.X"): "linear-algebra layer",
    ("R2.5", "leverage.X"): "linear-algebra layer",
    ("R2.5", "solve_lad.X"): "LAD solver: X matches the docstring formula sum rho_tau(y - Xb)",
    ("R2.5", "check_ols_invariants.X"): "OLS invariant oracle: X is the design in X'X and residuals",
    ("R2.5", "compute_cov.X"): "linear-algebra layer",
    ("R2.5", "compute_cov.resid"): "linear-algebra layer",
    ("R2.5", "cov_multiway_cluster.X"): "linear-algebra layer",
    ("R2.5", "cov_multiway_cluster.resid"): "linear-algebra layer",
    ("R2.5", "encode_groups.groups"): "linear-algebra layer",
    # R2.6: `**kwargs` is allowed only where it is explicitly rejected.
    ("R2.6", "IV2SLS.from_formula"): "rejects unknown options via _reject_unsupported_kwargs",
    ("R2.6", "ssc"): "forwards to the SSC dataclass, which rejects unknown fields",
    # R3.2: multi-equation constructors take equations, not one formula.
    ("R3.2", "Sureg.from_formula"): "multi-equation constructor; documented in the estimator table",
    # R3.2, DEPRECATION SHIMS: `from_variables` is the real entry point now; these
    # three forward to it and are kept for one release because `retest/` has ~180 call
    # sites.  They are a TRANSITION state, not an exemption: delete these three lines
    # (and the shims) once the call sites are migrated.
    ("R3.2", "Didregress.from_formula"): "deprecated alias -> from_variables (remove one release after migration)",
    ("R3.2", "RegressionAdjustment.from_formula"): "deprecated alias -> from_variables (remove after migration)",
    ("R3.2", "IPW.from_formula"): "deprecated alias -> from_variables (remove after migration)",
    ("R3.2", "NearestNeighbourMatching.from_formula"): "deprecated alias -> from_variables (remove after migration)",
}


@dataclass
class Finding:
    rule: str
    where: str
    message: str

    def allowed(self) -> bool:
        subject = self.where.split(":", 1)[0]
        return (self.rule, subject) in ALLOWED

    def reason(self) -> str:
        subject = self.where.split(":", 1)[0]
        return ALLOWED.get((self.rule, subject), "")


@dataclass
class Report:
    findings: dict[str, list[Finding]] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.setdefault(finding.rule, []).append(finding)

    def blocking(self) -> list[Finding]:
        return [f for group in self.findings.values() for f in group if not f.allowed()]


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: pathlib.Path) -> ast.Module:
    # `utf-8-sig`, not `utf-8`: two modules in this tree carry a BOM, and analysing
    # them is exactly what R5.3 is meant to catch rather than crash on.
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _rel(path: pathlib.Path) -> str:
    """Repo-relative on the real tree, absolute elsewhere (tests use a tmp module)."""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _public_name(name: str) -> bool:
    return not name.startswith("_")


def _annotation_names(node: object) -> str:
    # Two callers: an AST node (source scan) and ``inspect``'s return annotation,
    # which is a plain string whenever the module uses ``from __future__ import
    # annotations`` -- nearly all of this tree does.
    if node is None or node is inspect.Parameter.empty:
        return ""
    if isinstance(node, ast.AST):
        return ast.unparse(node)
    return str(node)


def _params(fn: ast.FunctionDef) -> list[ast.arg]:
    return list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)


def _positional(fn: ast.FunctionDef) -> list[str]:
    names = [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args)]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def _has_var_keyword(fn: ast.FunctionDef) -> bool:
    return fn.args.kwarg is not None


def _ignores_kwargs(fn: ast.FunctionDef) -> bool:
    """True when a `**kwargs` parameter is swallowed instead of forwarded.

    Forwarding is the accepted form: the inner callable has a real signature and
    raises ``TypeError`` on a typo, which is what a caller wants.  Ignoring is not:
    ``fit(self, **_ignored)`` accepts ``cov_type="HC1"`` and returns a fit that
    silently did nothing.  A ``**kwargs`` that the body never mentions is the finding.
    """

    if fn.args.kwarg is None:
        return False
    name = fn.args.kwarg.arg
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == name:
            return False
        if isinstance(node, ast.keyword) and node.arg is None:
            return False
    return True


# --------------------------------------------------------------------------- #
# R2: parameter order and construction
# --------------------------------------------------------------------------- #
def check_r2(report: Report) -> None:
    """R2.1 fit() takes cov_type first; R2.3 DataFrame is `data`; R2.6 no **kwargs."""

    for path in _modules():
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            public_class = _public_name(node.name)
            for member in node.body:
                if not isinstance(member, ast.FunctionDef):
                    continue

                # R2.1 -- fit(): the first positional argument (after self) is cov_type.
                if member.name == "fit" and public_class:
                    positional = _positional(member)
                    if positional and positional[0] != "cov_type":
                        report.add(
                            Finding(
                                "R2.1",
                                f"{node.name}.fit",
                                f"{_rel(path)}:{member.lineno}: first positional is "
                                f"`{positional[0]}`, not `cov_type` -- `.fit(\"HC1\")` "
                                f"would be read as `{positional[0]}=\"HC1\"`",
                            )
                        )

                # R2.6 -- a public **kwargs must either forward it (the callee then
                # rejects unknown keys) or not exist.  Silently IGNORING it is the bug.
                if _has_var_keyword(member) and public_class and _ignores_kwargs(member):
                    report.add(
                        Finding(
                            "R2.6",
                            f"{node.name}.{member.name}",
                            f"{_rel(path)}:{member.lineno}: public method accepts "
                            f"`**{member.args.kwarg.arg}` and never forwards it -- "
                            f"unknown options are silently ignored",
                        )
                    )

            # R3.2 -- every public classmethod called from_formula starts (formula, data).
            for member in node.body:
                if not isinstance(member, ast.FunctionDef) or member.name != "from_formula":
                    continue
                if not public_class:
                    continue
                positional = _positional(member)
                if positional[:2] != ["formula", "data"]:
                    report.add(
                        Finding(
                            "R3.2",
                            f"{node.name}.from_formula",
                            f"{_rel(path)}:{member.lineno}: from_formula starts "
                            f"({', '.join(positional[:2]) or 'nothing'}), expected "
                            f"(formula, data)",
                        )
                    )

        # R2.3/R2.6 -- module-level public functions.
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not _public_name(node.name):
                continue
            if _has_var_keyword(node) and _ignores_kwargs(node):
                report.add(
                    Finding(
                        "R2.6",
                        node.name,
                        f"{_rel(path)}:{node.lineno}: public function accepts "
                        f"`**{node.args.kwarg.arg}` and never forwards it -- unknown "
                        f"options are silently ignored",
                    )
                )
            for arg in _params(node):
                if arg.arg != "frame":
                    continue
                # R2.3 is about the DATAFRAME argument's name.  `synth2(..., frame=)`
                # is Stata's `frame()` result-dataset option -- a string naming a
                # frame, not a DataFrame parameter -- so flagging it by name alone
                # would demand renaming a faithful transcription of Stata's syntax.
                annotation = _annotation_names(arg.annotation)
                if annotation and "DataFrame" not in annotation:
                    continue
                report.add(
                    Finding(
                        "R2.3",
                        f"{node.name}.frame",
                        f"{_rel(path)}:{node.lineno}: DataFrame parameter is named "
                        f"`frame`; the house name is `data`",
                    )
                )


# --------------------------------------------------------------------------- #
# R3: keyword polarity, units, spelling
# --------------------------------------------------------------------------- #
#: Parameters that START with "no" but are not negative switches at all.
NEGATIVE_EXEMPT = {
    "nobs",
    "node",
    "nodes",
    "noise",
    "noncentrality",
    "normalized",
    "norm",
    "notes",
    "nobs_adj",
}

#: NEGATIVE-SOUNDING parameters that are STATA'S OWN OPTION NAMES (R1.2), so R3.1
#: does not apply: the whole point of the house rule is that a caller can look the
#: Python argument up in the Stata help file.  Each entry records the option it
#: transcribes; a name that cannot cite one belongs in R3.1's defect list instead.
STATA_NEGATIVE_OPTIONS = {
    "nomiss0",      # estat bgodfrey, nomiss0     (bgodfrey.py:36,195,207)
    "nosort",       # estat vif, nosort           (vif.py:264-266)
    "noheader",     # estat summarize, noheader   (summarize.py:305, sthlp:93)
    "noconstant",   # newey/xtunitroot, noconstant (newey.py:10 quotes the syntax line)
    "nowald",       # contrast, nowald            (contrast.py:436)
    "noweights",    # estat summarize, noweights
    "nofigure",     # synth2, nofigure -- graphics; westata refuses graphics outright
    "nonrtolerance",# synth, forwarded to mlopts; pairs with nrtolerance (synth.py:804-808)
    "customV",      # synth/synth2, Stata option `customV()` (synth2.ado:253-256)
    # DEPRECATION SHIM, not a Stata option: `Areg(noabsorb=)` was renamed to
    # `absorb_test=` because it turns off the absorbed-indicator TEST, not absorption
    # itself.  Kept for one release; delete this entry with the shim.
    "noabsorb",
}
#: Stata option names that are camelCase, so R2.5's snake_case rule does not apply:
#: the Python argument must be findable in Stata's help file (R1.2).
STATA_CAMEL_CASE_OPTIONS = frozenset(
    {
        "customV",  # synth/synth2, `customV()` (synth2.ado:253-256)
    }
)



def check_r3(report: Report) -> None:
    """R3.1 no `noX` switches; R3.3 `level` is a fraction; R3.4 snake_case."""

    for path in _modules():
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                owner = "module"
                params = _params(node)
            elif isinstance(node, ast.ClassDef):
                continue
            else:
                continue
            for arg in params:
                name = arg.arg
                if name in ("self", "cls"):
                    continue

                # R3.1 -- Stata's negative options must become positive switches.
                # `startswith("no")` alone flags `normalize_counts` and `noise`; the
                # switch pattern is `no` + a WORD (`noconstant`, `noheader`), so require
                # the character after "no" to be lowercase and the name not to start
                # with a longer non-negative prefix.
                if (
                    name.startswith("no")
                    and not name.startswith(("norm", "nois"))
                    and name not in NEGATIVE_EXEMPT
                    and name not in STATA_NEGATIVE_OPTIONS
                ):
                    report.add(
                        Finding(
                            "R3.1",
                            f"{owner}.{node.name}.{name}",
                            f"{_rel(path)}:{node.lineno}: `{node.name}()` exposes the "
                            f"negative switch `{name}=`",
                        )
                    )

                # R3.3 -- a parameter called `level` holds a FRACTION in (0, 1).
                if name == "level":
                    default = _default_of(node, name)
                    if default is not None and not 0.0 < default < 1.0:
                        report.add(
                            Finding(
                                "R3.3",
                                f"{node.name}.level",
                                f"{_rel(path)}:{node.lineno}: `{node.name}(level="
                                f"{default})` -- `level` must be a fraction such as 0.95, "
                                f"not a percentage",
                            )
                        )

                # R2.5 -- a PUBLIC parameter of a PUBLIC callable is snake_case.  Private
                # helpers keep mathematical notation on purpose (see the document), and so
                # does the linear-algebra layer, so those are out of scope here.
                if (
                    _public_name(node.name)
                    and _has_upper(name)
                    and name not in STATA_CAMEL_CASE_OPTIONS
                ):
                    report.add(
                        Finding(
                            "R2.5",
                            f"{node.name}.{name}",
                            f"{_rel(path)}:{node.lineno}: parameter `{name}` is not "
                            f"snake_case",
                        )
                    )

            # R2.5 -- public function names are snake_case.
            if isinstance(node, ast.FunctionDef) and _public_name(node.name) and _has_upper(node.name):
                report.add(
                    Finding(
                        "R2.5",
                        node.name,
                        f"{_rel(path)}:{node.lineno}: public function `{node.name}` is "
                        f"not snake_case",
                    )
                )


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _default_of(fn: ast.FunctionDef, name: str) -> float | None:
    args = list(fn.args.posonlyargs) + list(fn.args.args)
    offset = len(args) - len(fn.args.defaults)
    for index, arg in enumerate(args):
        if arg.arg == name:
            default_index = index - offset
            if 0 <= default_index < len(fn.args.defaults):
                node = fn.args.defaults[default_index]
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
            return None
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if arg.arg == name and isinstance(default, ast.Constant):
            if isinstance(default.value, (int, float)):
                return float(default.value)
    return None


# --------------------------------------------------------------------------- #
# R5: export integrity
# --------------------------------------------------------------------------- #
#: Subpackages whose public names must also appear at top level (R5.2).  ``vce``,
#: ``conformance``, ``data`` and ``report`` are internal or deliberately narrow.
USER_FACING_SUBPACKAGES = ("westata.estimators", "westata.postestimation", "westata.results")


def _is_command_name(name: str) -> bool:
    """Whether a module-level function is a Stata COMMAND rather than a helper.

    The distinction decides what belongs on the public surface.  A command is what a
    user types after a fit (`estat auc`, `linktest`, `suest`, `vargranger`); a helper is
    plumbing (`resolve_names`, `refuse_panel_fit`, `ordered_probability`).
    """

    if name.startswith("estat_"):
        return True
    return name in {
        "linktest",
        "lrtest",
        "suest",
        "vargranger",
        "rcm",
        "tebalance_summarize",
        "endog_regf_test",
        "endog_score_test",
        "testnl",
        "nlcom",
        "margins",
        "predict",
        "hausman",
        "contrast",
        "pwcompare",
        "summarize",
        "dfgls",
        "kpss",
        "vecrank",
        "varsoc",
        "varstable",
        "varwle",
        "varlmar",
        "veclmar",
        "estat_dwatson",
    }


def check_r5(report: Report) -> None:
    """R5.1 `__all__` resolves; R5.2 subpackage names reach the top level; R5.3 no BOM."""

    import importlib

    packages = [
        "westata",
        "westata.estimators",
        "westata.postestimation",
        "westata.results",
        "westata.vce",
        "westata.data",
        "westata.report",
        "westata.errors",
        "westata.conformance",
    ]
    declared: dict[str, set[str]] = {}
    for name in packages:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - a broken import is its own report
            report.add(Finding("R5.1", name, f"import failed: {type(exc).__name__}: {exc}"))
            continue
        exported = getattr(module, "__all__", None)
        if exported is None:
            continue
        declared[name] = set(exported)
        for entry in exported:
            if not hasattr(module, entry):
                report.add(
                    Finding(
                        "R5.1",
                        f"{name}.{entry}",
                        f"`{name}.__all__` names `{entry}`, which the module does not "
                        f"define -- `from {name} import *` raises AttributeError",
                    )
                )

    # R5.6 -- a public classmethod's RETURN ANNOTATION must name something a caller
    # can use.  `Logit.from_formula` was annotated `-> "_DiscreteBase"`: a private base
    # class the caller never sees, so the annotation described an object they could not
    # name (11 estimators were like this).  TypeVars are allowed -- `_DiscreteT` is the
    # conventional spelling for "the concrete cls".
    import inspect as _inspect

    _westata = importlib.import_module("westata")
    for entry in declared.get("westata", set()):
        obj = getattr(_westata, entry, None)
        if not (_inspect.isclass(obj) and hasattr(obj, "from_formula")):
            continue
        try:
            annotation = str(_inspect.signature(obj.from_formula).return_annotation)
        except (ValueError, TypeError):
            continue
        name = annotation.strip("'\"")
        if not name or name in ("Any", "None", "inspect.Parameter.empty"):
            continue
        if name.endswith("T") and name.startswith("_"):
            continue  # bound TypeVar: "the concrete class"
        if name.startswith("_"):
            report.add(
                Finding(
                    "R5.6",
                    f"{entry}.from_formula",
                    f"`{entry}.from_formula` is annotated `-> {name!r}`, a PRIVATE class "
                    f"the caller cannot name; annotate the concrete class or a bound "
                    f"TypeVar",
                )
            )

    # R5.7 -- `estat_*` must exist BOTH ways.  The README's main path is
    # `result.estat_vce()`, but Stata's own spelling is `estat vce` after a fit, and a
    # caller porting commands expects `ws.estat_vce(result)`.  Six functions were
    # method-only (`estat_auc`, `estat_esize`, `estat_moran`, `estat_mundlak`,
    # `estat_szroeter`, `estat_weakrobust`) and could not be called the second way.
    _base = getattr(_westata, "StataResults", None)
    if _base is not None:
        as_methods = {name for name in dir(_base) if name.startswith("estat_")}
        as_functions = {
            name for name in declared.get("westata", set()) if name.startswith("estat_")
        }
        for name in sorted(as_methods - as_functions):
            report.add(
                Finding(
                    "R5.7",
                    name,
                    f"`StataResults.{name}()` exists as a method but `westata.{name}` "
                    f"does not; Stata's spelling works one way only",
                )
            )
        for name in sorted(as_functions - as_methods):
            report.add(
                Finding(
                    "R5.7",
                    name,
                    f"`westata.{name}` exists but `StataResults.{name}()` does not",
                )
            )

    # R5.8 -- a public method must expose a REAL signature.
    #
    # `StataResults` forwarded 29 postestimation methods through `**kwargs`, so
    # `inspect.signature(result.estat_vif)` said `(**kwargs)` and the parameters were
    # discoverable only by reading the target function.  That is the same defect
    # `test_subset_is_uniform.py` records for `IV2SLS.from_formula`: a capability that
    # exists but cannot be introspected gets reported as missing by tooling, and
    # `help()` shows the reader nothing.
    #
    # `**kwargs` is still legitimate when the method forwards to a variadic callable
    # (`test(self, *specs, **kwargs)`), so the rule fires only when the method has NO
    # named parameters at all besides self.
    for class_name, names in (("StataResults", ("westata",)),):
        cls = getattr(_westata, class_name, None)
        if cls is None:
            continue
        for member_name, member in vars(cls).items():
            if member_name.startswith("_") or not callable(member):
                continue
            try:
                params = _inspect.signature(member).parameters
            except (ValueError, TypeError):
                continue
            named = [
                p for p in params.values()
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
            if has_kwargs and not named:
                report.add(
                    Finding(
                        "R5.8",
                        f"{class_name}.{member_name}",
                        f"`{class_name}.{member_name}` is `(**kwargs)` only: the real "
                        f"parameters are invisible to `help()` and `inspect.signature`",
                    )
                )

    # R5.2 (third gap) -- a public FUNCTION in a subpackage MODULE that the
    # subpackage does not re-export is invisible twice over: `westata.estimators.pe`
    # cannot reach it and the top level never sees it.  `linktest`, `suest`,
    # `vargranger`, `rcm`, `tebalance_summarize`, `endog_regf_test` and
    # `endog_score_test` were all like this -- real commands, exported nowhere.
    #
    # Only `postestimation` and `estimators` are checked: they are the user-facing
    # command layers.  Helper modules (`names`, `ts_lags`, `predict`) intentionally keep
    # internal utilities private.
    for package_name, module_dir in (
        ("westata.postestimation", SRC / "postestimation"),
        ("westata.estimators", SRC / "estimators"),
    ):
        exported = declared.get(package_name, set())
        for path in sorted(module_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            try:
                tree = _parse(path)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                if node.name in exported or node.name in declared.get("westata", set()):
                    continue
                # Only COMMANDS belong on the public surface.  A module also holds
                # helpers whose names are plain English (`resolve_names`,
                # `refuse_panel_fit`, `ordered_probability`); exporting those would
                # bury the 227 real names among implementation details.  The test
                # below is "does the name read as a Stata command?", applied by
                # checking it against the command vocabulary the package uses:
                # `estat_*`, `test*`, and the plain command names Stata itself has.
                if not _is_command_name(node.name):
                    continue
                report.add(
                    Finding(
                        "R5.2",
                        f"{package_name}.{path.stem}.{node.name}",
                        f"{_rel(path)}:{node.lineno}: public function `{node.name}` is "
                        f"not re-exported by `{package_name}`",
                    )
                )

    top = declared.get("westata", set())
    # R5.2 applies only to the USER-FACING subpackages.  ``vce``, ``conformance``,
    # ``data`` and ``report`` are internal layers (or a deliberately narrow helper
    # surface); requiring their names at top level would clutter ``ws.`` with
    # machinery no reader should call directly.
    for package in USER_FACING_SUBPACKAGES:
        for entry in sorted(declared.get(package, set()) - top):
            report.add(
                Finding(
                    "R5.2",
                    f"{package}.{entry}",
                    f"`{entry}` is exported by `{package}` but not by `westata`",
                )
            )

    # R5.5 -- every module must COMPILE from its raw bytes.
    #
    # This is not a substitute for the test suite; it is a guard against a batch
    # rewrite corrupting a file in a way `.pyc` caching hides from `import westata`.
    # Measured cost of not having it: a `\x0c` in a comment was rewritten as a
    # newline, `panel.py` stopped parsing, and the failure only surfaced 15 minutes
    # into a full test run (47 failed + 20 errors).
    for path in _modules():
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:
            report.add(
                Finding(
                    "R5.5",
                    _rel(path),
                    f"{_rel(path)}:{exc.lineno}: does not compile from its raw bytes "
                    f"({exc.msg}); a stale .pyc may still make `import westata` succeed",
                )
            )

    for path in _modules():
        if path.read_bytes().startswith(b"\xef\xbb\xbf"):
            report.add(
                Finding(
                    "R5.3",
                    _rel(path),
                    f"{_rel(path)} starts with a UTF-8 BOM; `ast.parse` on the raw bytes "
                    f"fails and every reader must remember `utf-8-sig`",
                )
            )


# --------------------------------------------------------------------------- #
# R4: result surface
# --------------------------------------------------------------------------- #
def _protocol_requirements() -> list[str]:
    """The attribute names ``ResultsProtocol`` requires, read from its annotations."""

    try:
        from westata.protocols import ResultsProtocol
    except Exception:  # pragma: no cover - a broken import is reported by R5.1
        return []
    names = list(getattr(ResultsProtocol, "__annotations__", {}))
    # `e` is declared as `Mapping`, and the methods are declared separately.
    names += ["cov_params", "conf_int", "summary"]
    return names


def _declares(cls: type, name: str) -> bool:
    return hasattr(cls, name)


def _looks_like_estimate(cls: type) -> bool:
    """An estimation result exposes coefficients; a table result does not."""

    return hasattr(cls, "params") or hasattr(cls, "conf_int")


def check_r4(report: Report) -> None:
    """R4.1 result classes carry a conversion method; R4.3 conf_int returns a frame.

    Two discovery traps this rule has already fallen into, both fixed here:

    * **Only ``__all__`` was scanned.** The statsmodels adapter ``VECMResults`` is
      defined in ``estimators/tseries.py`` and never exported, so the ``conf_int``
      branch could not fire -- a rule that reports a confident "0 findings" forever.
      Modules are now walked for the classes they *define*.
    * **The class-name filter dropped ``Results``.** ``VARResults``/``VECMResults``/
      ``StoredResults`` end in ``Results``, not ``Result``.
    """

    import importlib
    import inspect

    #: Machine-readable extraction.  ``render``/``summary`` are DISPLAY and do not
    #: count: a reader who wants the numbers must not have to parse a table.
    result_methods = {"to_dict", "to_frame"}
    #: Estimator results implement the statsmodels surface instead (R4.2); they are
    #: not required to carry a converter, and `StataResults` deliberately does not.
    protocol_implementors = {"StataResults"}

    candidates: list[tuple[str, type]] = []
    modules = [
        "westata",
        "westata.estimators",
        "westata.postestimation",
        "westata.results",
        "westata.estimators.tseries",
        "westata.results.ecosystem",
        "westata.expressions.explist",
    ]
    for name in modules:
        module = importlib.import_module(name)
        for entry in getattr(module, "__all__", []):
            obj = getattr(module, entry, None)
            if inspect.isclass(obj) and obj.__module__.startswith("westata"):
                candidates.append((entry, obj))
        for entry, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == getattr(module, "__name__", ""):
                candidates.append((entry, obj))

    # A result class the package RETURNS but never exports is invisible to the loop
    # above, and that is how `PanelUnitRootResult` (21 such classes in total) escaped
    # R4.1/R5.2 for two rounds: it was reachable from `.fit()` yet absent from every
    # `__all__`.  Walk the source for public `*Result`-style classes and require each to
    # be exported by some package.
    declared: set[str] = set()
    for module_name in ("westata", "westata.estimators", "westata.postestimation", "westata.results"):
        declared |= set(getattr(importlib.import_module(module_name), "__all__", []))
    for path in _modules():
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith(("Result", "Results", "Table", "Matrix")):
                continue
            if node.name.startswith("_"):
                continue  # private implementation container, not public surface
            if node.name in declared:
                continue
            report.add(
                Finding(
                    "R5.2",
                    node.name,
                    f"{_rel(path)}:{node.lineno}: `{node.name}` is a public result class "
                    f"that no package exports; callers can receive it from a command but "
                    f"cannot name it",
                )
            )

    seen: set[type] = set()
    for entry, obj in candidates:
        if not entry.endswith(("Result", "Results", "Table", "Matrix")):
            continue
        if obj in seen:
            continue  # exported by both `westata` and its subpackage; report once.
        seen.add(obj)

        # R4.1 -- a public result object must be extractable without a renderer.
        if entry not in protocol_implementors and not (result_methods & set(dir(obj))):
            report.add(
                Finding(
                    "R4.1",
                    entry,
                    f"`{entry}` offers none of {sorted(result_methods)}",
                )
            )

        # R4.3 -- conf_int: same name, same shape (a DataFrame), everywhere.
        conf_int = getattr(obj, "conf_int", None)
        if conf_int is not None and callable(conf_int):
            annotation = _annotation_names(inspect.signature(conf_int).return_annotation)
            if "ndarray" in annotation:
                report.add(
                    Finding(
                        "R4.3",
                        f"{entry}.conf_int",
                        f"`{entry}.conf_int()` returns {annotation}; the house shape "
                        f"is a DataFrame with `lower`/`upper`",
                    )
                )

        # R4.2 -- an ESTIMATION result implements the statsmodels duck type.
        # This is a DECLARATION check: `ResultsProtocol` is @runtime_checkable, but
        # instantiating a result class to test `isinstance` would run real work (and
        # some results are only constructible from a fitted model).  The protocol's
        # own annotations name the required attributes, so those are what we compare.
        if entry not in protocol_implementors and _looks_like_estimate(obj):
            required = _protocol_requirements()
            missing = sorted(name for name in required if not _declares(obj, name))
            if missing:
                report.add(
                    Finding(
                        "R4.2",
                        entry,
                        f"`{entry}` does not satisfy ResultsProtocol; missing "
                        f"{', '.join(missing)}",
                    )
                )


# --------------------------------------------------------------------------- #
# R6: no duplicated private helpers
# --------------------------------------------------------------------------- #
#: Helpers whose several definitions are DELIBERATE because each family resolves a
#: different vocabulary.  ``_normalize_cov_type`` looks like a duplicate by name but is
#: not one: ``did`` maps ``robust`` to HC1, ``panel`` to a clustered VCE, ``hdfe``
#: forwards to ``_COV_ALIASES``, and ``dispatch`` takes a ``robust_as`` argument.  Those
#: are four different ANSWERS, not one function written four times -- merging them would
#: recreate the single-global-default defect that R3.7.0 removed.
#:
#: The test for membership here is "is this genuinely the same function?", and the way
#: to earn an entry is the same as for ALLOWED: say why in the document.
PER_FAMILY_HELPERS = frozenset(
    {
        "_normalize_cov_type",  # one vocabulary per estimator family (R3.7.0)
        "_normalize_cov_type_for",  # noqa: kept for future family splits
    }
)


def _shapes(entries: list[tuple[str, str]]) -> set[str]:
    """Each definition's body with LITERAL ARGUMENTS abstracted away.

    Two wrappers that both ``return shared(result, "estat x")`` are the same shape: the
    only difference is the string they pass.  That is the desired end-state for a
    delegating shim, so R6.1 must not flag it -- but a body that genuinely computes
    something else has a different shape and still is flagged.
    """

    import re as _re

    shapes = set()
    for _, body in entries:
        # Replace string and numeric literals with a placeholder; keep call structure.
        shapes.add(_re.sub(r"""(['"]).*?\1|\b\d+(?:\.\d+)?\b""", "LIT", body))
    return shapes


def _delegates_to_owner(entries: list[tuple[str, str]]) -> bool:
    """True when one definition is the implementation and the others forward to it.

    The end-state R6.1 asks for is "one implementation, N thin wrappers".  Wrappers
    necessarily have a different body from the implementation -- that is what makes
    them wrappers -- so a plain body comparison would keep flagging the very shape the
    rule is trying to produce.  A definition counts as a wrapper when its body is a
    single ``return`` (optionally behind an import) whose call names the shared
    function.
    """

    import re as _re

    def is_wrapper(body: str) -> bool:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return False
        returns = [line for line in lines if line.startswith("return ")]
        if len(returns) != 1:
            return False
        # Every non-return line must be an import naming the shared module/function.
        others = [line for line in lines if line not in returns]
        return all(
            line.startswith(("import ", "from ")) for line in others
        ) and bool(_re.search(r"\b[a-z_][a-z0-9_]*\s*\(", body))

    # A wrapper forwards to ANY shared function, so the test cannot be a name
    # whitelist -- that is what made this miss `numerical_hessian` and
    # `slope_positions` after they were unified.  The shared function is the one
    # every other definition calls; here we only require "some call, no computation".
    wrappers = sum(1 for _, body in entries if is_wrapper(body))
    # Either one module holds the implementation and the rest wrap it, or ALL of them
    # are wrappers (the implementation lives under a public name elsewhere, as with
    # `results.stars.stars`).  Both are the end-state R6.1 asks for.
    return wrappers >= len(entries) - 1


def check_r6(report: Report) -> None:
    """R6.1 a helper needed by three modules lives in one module.

    Membership is by NAME and BODY SHAPE: a helper whose body is byte-identical in
    several modules is a copy; a helper that only shares a name but computes a
    family-specific answer is listed in ``PER_FAMILY_HELPERS`` and exempt.
    """

    owners: dict[str, list[tuple[str, str]]] = {}
    for path in _modules():
        tree = _parse(path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
                # The body, minus the docstring and the signature, is what decides
                # whether two definitions are the same function.
                body = "\n".join(
                    ast.unparse(statement)
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str)
                    )
                )
                owners.setdefault(node.name, []).append(
                    (f"{_rel(path)}:{node.lineno}", body)
                )

    for name, entries in sorted(owners.items()):
        if len(entries) < 3:
            continue
        if name in PER_FAMILY_HELPERS:
            continue
        bodies = {body for _, body in entries}
        locations = [where for where, _ in entries]
        if len(bodies) == 1:
            detail = "byte-identical copies"
        elif len(_shapes(entries)) == 1:
            detail = (
                "same shape, differing only in literal arguments (delegating wrappers)"
            )
            # AND THAT IS FINE: do not report it.  See the comment above.
            continue
        elif _delegates_to_owner(entries):
            # One module holds the IMPLEMENTATION and the rest forward to it.  That is
            # the fix R6.1 asks for, so it must not be reported as the problem: the
            # wrapper bodies differ from the implementation's body by construction.
            continue
        else:
            # Different bodies under one name is a NAMING collision rather than a
            # copy: the reader cannot tell which one a call reaches, but merging them
            # would be wrong.  Still reported, with the distinction spelled out.
            detail = f"{len(_shapes(entries))} DIFFERENT implementations sharing one name"
            report.add(
                Finding(
                    "R6.1",
                    name,
                    f"`{name}` is redefined in {len(entries)} modules ({detail}): "
                    + ", ".join(locations[:6])
                    + (" ..." if len(locations) > 6 else ""),
                )
            )
            continue
        report.add(
            Finding(
                "R6.1",
                name,
                f"`{name}` is redefined in {len(entries)} modules ({detail}): "
                + ", ".join(locations[:6])
                + (" ..." if len(locations) > 6 else ""),
            )
        )


# --------------------------------------------------------------------------- #
# R3.7: `cov_type` must be canonicalised per family, never echoed back
# --------------------------------------------------------------------------- #
#: Every family, and what its ``cov_type="robust"`` MUST resolve to (R3.7.1).  The
#: value is the canonical name the RESULT OBJECT must report, not the Stata option.
#: What a family's `cov_type="robust"` must resolve to.  `"REFUSED"` means Stata has no
#: such VCE for that command, so `robust` must raise.
#:
#: R3.7.0 is the governing rule: **if Stata's command has `vce(robust)`, `robust` must be
#: accepted.**  So ``"REFUSED"`` is only legitimate for commands with no `vce()` at all
#: (``rreg``) or where westata genuinely has not implemented the estimator yet -- and
#: ``REFUSAL_IS_MISSING_IMPLEMENTATION`` records which is which, because "not supported"
#: and "not implemented yet" are different promises to the user.
COV_TYPE_ROBUST_TARGETS: dict[str, str] = {
    "OLS": "HC1",
    "IV2SLS": "HC0",
    "IVLIML": "HC0",
    "IVGMM": "HC0",
    "Logit": "ML_ROBUST",
    "Probit": "ML_ROBUST",
    "Poisson": "ML_ROBUST",
    "NegativeBinomial": "ML_ROBUST",
    "OrderedLogit": "ML_ROBUST",
    "OrderedProbit": "ML_ROBUST",
    "ZeroInflatedPoisson": "ML_ROBUST",
    "ZeroInflatedNegativeBinomial": "ML_ROBUST",
    "TruncatedPoisson": "ML_ROBUST",
    "TruncatedNegativeBinomial": "ML_ROBUST",
    "PanelOLS": "cluster",
    "RandomEffects": "cluster",
    "RandomEffectsMLE": "cluster",
    "BetweenOLS": "cluster",
    "FirstDifferenceOLS": "cluster",
    "Sureg": "SYSTEMS_ROBUST",
    "Mvreg": "SYSTEMS_ROBUST",
    "QuantileRegression": "QREG_ROBUST",
    # `Sqreg`/`Bsqreg` take their covariance FROM the bootstrap: `cov_type="bootstrap"`
    # is a real, distinct value here (not an HC alias), so `robust` has no separate
    # meaning and must be refused rather than silently mapped to one.
    "Sqreg": "bootstrap",
    "Bsqreg": "bootstrap",
    # `RobustRegression` is `rreg`: its covariance comes from the biweight iteration,
    # and Stata's `rreg` has no `vce()` at all.  It currently ACCEPTS any cov_type and
    # silently returns the default -- which is worse than refusing.
    "RobustRegression": "REFUSED",
    # `FEOLS`/`Areg` currently REFUSE `robust` and demand `HC1`.  That is R3.7.0a's
    # canonical "fake refusal": `reghdfe, robust` IS HC1, the implementation already
    # exists, and only the alias table is missing a key.  Recorded as HC1, not REFUSED,
    # so the checker demands the fix rather than blessing the workaround.
    "FEOLS": "HC1",
    "Areg": "HC1",
}

#: Refusals that are HONEST (Stata has no `vce()` for that command).  A refusal here
#: is correct behaviour.  Everything else that refuses is a missing implementation,
#: recorded in ``REFUSAL_IS_MISSING_IMPLEMENTATION`` so the error message can say
#: "not implemented yet" instead of implying the user asked for something invalid.
REFUSAL_IS_CORRECT = frozenset({"RobustRegression"})

#: Families where Stata DOES have `vce(robust)` but westata has not implemented it.
#: `robust` must still raise today, but the message must say "not implemented yet"
#: and the entry is a standing TODO, tracked here so it cannot be forgotten.
REFUSAL_IS_MISSING_IMPLEMENTATION = frozenset(
    {
        "Tobit",
        "Truncreg",
        "Intreg",
        "Heckman",
        "Etregress",
        "XtIvreg",
        "XtGls",
        "XtPcse",
    }
)

#: Families where `robust` is legitimately REFUSED (R3.7.1's "reject" rows).  A
#: refusal is correct behaviour, so these must raise rather than resolve.
COV_TYPE_ROBUST_REFUSED = frozenset(
    {
        "Tobit",
        "Truncreg",
        "Intreg",
        "Heckman",
        "Etregress",
        "Eivreg",
        "XtIvreg",
        "XtGls",
        "XtPcse",
        "XtUnitroot",
        "XtLogit",
        "XtProbit",
        "XtTobit",
        "XtPoisson",
        "XtNbreg",
        "Didregress",
    }
)


def check_r3_cov_type(report: Report) -> None:
    """R3.7 -- `cov_type` is canonicalised per family and echoed as the canonical value.

    This rule can only be checked BEHAVIOURALLY: a family may canonicalise with a dict,
    a match statement, or a helper in another module, and inspecting source for the
    literal ``"robust"`` finds the wrong thing in all three cases.  So each family is
    actually fitted twice and the reported ``cov_type`` compared -- the same technique
    that caught ``PanelOLS.fit("HC1")`` being silently ignored.
    """

    import numpy as np
    import pandas as pd

    import westata as ws

    rng = np.random.default_rng(20260101)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=240),
            "x": rng.normal(size=240),
            "yb": (rng.normal(size=240) > 0).astype(int),
            "ycnt": rng.poisson(2.0, size=240),
            "firm": np.repeat(np.arange(24), 10),
            "year": np.tile(np.arange(1, 11), 24),
        }
    )
    frame["z"] = frame["x"] * 0.6 + rng.normal(size=240)

    builders = {
        "OLS": lambda: ws.OLS.from_formula("y ~ x", frame).fit(cov_type="robust"),
        "IV2SLS": lambda: ws.IV2SLS.from_formula(
            "y ~ [x ~ z]", frame
        ).fit(cov_type="robust"),
        "IVLIML": lambda: ws.IVLIML.from_formula("y ~ [x ~ z]", frame).fit(cov_type="robust"),
        "Logit": lambda: ws.Logit.from_formula("yb ~ x", frame).fit(cov_type="robust"),
        "Probit": lambda: ws.Probit.from_formula("yb ~ x", frame).fit(cov_type="robust"),
        "Poisson": lambda: ws.Poisson.from_formula("ycnt ~ x", frame).fit(cov_type="robust"),
        "OrderedLogit": lambda: ws.OrderedLogit.from_formula("yb ~ x", frame).fit(
            cov_type="robust"
        ),
        "PanelOLS": lambda: ws.PanelOLS.from_formula(
            "y ~ x", frame, entity="firm", time="year"
        ).fit(cov_type="robust"),
        "RandomEffects": lambda: ws.RandomEffects.from_formula(
            "y ~ x", frame, entity="firm", time="year"
        ).fit(cov_type="robust"),
        "QuantileRegression": lambda: ws.QuantileRegression.from_formula(
            "y ~ x", frame
        ).fit(cov_type="robust"),
        "Sureg": lambda: ws.Sureg.from_formula(["y ~ x", "x ~ y"], frame).fit(
            cov_type="robust"
        ),
        # `y ~ x + z | firm` has k=2 (x, z) plus the absorbed firms: k>=2, so the
        # n/(n-1) vs n/(n-k) difference is visible (R3.7.5).
        "FEOLS": lambda: ws.FEOLS.from_formula("y ~ x + z | firm", frame).fit(
            cov_type="robust"
        ),
        # `rreg` has no `vce()` at all.  It currently ACCEPTS cov_type and returns the
        # default, which is a silent wrong-variance bug rather than a hard error.
        "RobustRegression": lambda: ws.RobustRegression.from_formula("y ~ x", frame).fit(
            cov_type="robust"
        ),
    }

    for family, build in builders.items():
        expected = COV_TYPE_ROBUST_TARGETS.get(family)
        if expected is None:
            continue
        try:
            result = build()
        except Exception as exc:  # noqa: BLE001 - the family's refusal IS the answer
            if expected == "REFUSED":
                continue
            # A family that cannot even fit is its own finding.
            report.add(
                Finding(
                    "R3.7",
                    family,
                    f"`{family}` could not be fitted for the cov_type check: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if expected == "REFUSED":
            report.add(
                Finding(
                    "R3.7",
                    family,
                    f"`{family}(cov_type='robust')` was ACCEPTED; Stata's family has no "
                    f"`robust` VCE here (or westata has not implemented it), so the "
                    f"request must be refused rather than resolved to some number",
                )
            )
            continue
        reported = getattr(result, "cov_type", None)
        if reported != expected:
            report.add(
                Finding(
                    "R3.7",
                    f"{family}.cov_type",
                    f"`{family}(cov_type='robust')` reports `{reported!r}`; the family "
                    f"table (R3.7.1) says it must be `{expected}`. A user copying "
                    f"`cov_type='robust'` between families gets numerically different "
                    f"variances under the same label.",
                )
            )

        # R3.7.5 -- the fixture must have k >= 2, or this whole check is vacuous.
        # `n/(n-1)` and `n/(n-k)` are EQUAL at k=1, so a single-parameter fixture
        # cannot tell ML_ROBUST from HC1 no matter what the code does.  Guarding the
        # fixture keeps the check honest if someone later "simplifies" the formulas.
        params = np.asarray(getattr(result, "params", []), dtype=float).reshape(-1)
        if params.size < 2:
            report.add(
                Finding(
                    "R3.7.5",
                    f"{family}.fixture",
                    f"the `{family}` fixture has k={params.size}; the robust check is "
                    f"VACUOUS below k=2 because n/(n-1) == n/(n-k) there",
                )
            )


def check_r3_cov_type_panel_hc1(report: Report) -> None:
    """R3.7.3 -- `HC1` on a panel estimator must not be reported as `HC1`.

    `xtreg` has no heteroskedasticity-only VCE: Stata's `vce(robust)` IS
    `vce(cluster panelvar)`.  westata accepts `cov_type="HC1"` on `PanelOLS` and
    computes a clustered covariance, but the result still says `HC1`, so a reader
    comparing it with `OLS(cov_type="HC1")` is comparing two different estimators
    under one label.  Either refuse `HC1` or report `cluster`; reporting `HC1` is
    the one option that is definitely wrong.
    """

    import numpy as np
    import pandas as pd

    import westata as ws

    rng = np.random.default_rng(20260102)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=240),
            "x": rng.normal(size=240),
            "firm": np.repeat(np.arange(24), 10),
            "year": np.tile(np.arange(1, 11), 24),
        }
    )
    for family in ("PanelOLS", "RandomEffects"):
        model = getattr(ws, family).from_formula(
            "y ~ x", frame, entity="firm", time="year"
        )
        try:
            result = model.fit(cov_type="HC1")
        except Exception:  # noqa: BLE001 - refusing HC1 is the recommended fix
            continue
        reported = getattr(result, "cov_type", None)
        if str(reported).upper() == "HC1":
            report.add(
                Finding(
                    "R3.7.3",
                    f"{family}.HC1",
                    f"`{family}(cov_type='HC1')` reports `HC1` but computes a CLUSTERED "
                    f"covariance (xtreg's vce(robust) is vce(cluster panelvar)); the "
                    f"label makes it look comparable with OLS's HC1, which it is not",
                )
            )


CHECKS = {
    "R2": check_r2,
    "R3": check_r3,
    "R3.7": check_r3_cov_type,
    "R3.7.3": check_r3_cov_type_panel_hc1,
    "R4": check_r4,
    "R5": check_r5,
    "R6": check_r6,
}


def collect(only: str | None = None) -> Report:
    """Run every check, then filter by RULE ID.

    Filtering by check function is wrong: ``check_r2`` legitimately emits ``R3.2``
    (``from_formula`` order is a shape rule about a constructor, and it is grouped
    with the other constructor rules).  Filtering on the finding's own rule id keeps
    ``--rule R2`` honest -- the test suite asserts it.
    """

    report = Report()
    for check in CHECKS.values():
        check(report)
    if only:
        wanted = only.upper()
        report.findings = {
            rule: group
            for rule, group in report.findings.items()
            if rule.upper().startswith(wanted)
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 if any rule fires")
    parser.add_argument("--rule", default=None, help="only this rule family, e.g. R2")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    # `src`, not `src/westata`: putting the PACKAGE directory on sys.path makes
    # `westata/random` shadow the standard library's `random`, and the next numpy
    # import dies with `cannot import name 'SystemRandom'`.  That is R5.4.
    sys.path.insert(0, str(SRC_ROOT))
    report = collect(args.rule)

    if args.json:
        payload = {
            rule: [
                {
                    "where": f.where,
                    "message": f.message,
                    "allowed": f.allowed(),
                    "reason": f.reason(),
                }
                for f in group
            ]
            for rule, group in sorted(report.findings.items())
        }
        print(json.dumps(payload, indent=1))
    else:
        print("westata public-API conventions (docs/API_CONVENTIONS.md)\n")
        total = 0
        allowed = 0
        for rule, group in sorted(report.findings.items()):
            print(f"{rule}  {len(group)} finding(s)")
            for finding in group:
                total += 1
                if finding.allowed():
                    allowed += 1
                    print(f"  ok    {finding.where}: {finding.message}")
                    print(f"        allowed: {finding.reason()}")
                else:
                    print(f"  FAIL  {finding.where}: {finding.message}")
            print()
        blocking = len(report.blocking())
        print(f"{total} finding(s), {allowed} allowed by a documented exception, "
              f"{blocking} to fix")
        if not report.findings:
            print("no findings")

    if args.strict and report.blocking():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
