"""Probe westata's public API for gaps that are problems *on its own terms*.

What this module is for
-----------------------
``REPORT.md`` section 6.4 lists seven things westata could improve.  Those claims are
worth exactly as much as the probe behind them, so the probe lives here rather than in a
throwaway script: each entry states an expectation, exercises the API, and reports
whether it held.  Re-running it after a westata release answers "is this list still
true?" without re-reading the report.

What counts as a gap
--------------------
westata's API is deliberately statsmodels-shaped -- that is the design, stated in the
README, not a shortfall.  So "differs from Stata's syntax" is **not** a gap and the probe
contains no such entry.  Two earlier drafts of the report got this wrong; the deleted
items are listed at the bottom of this file so the reasoning is auditable.

A gap must satisfy one of:

1. **Capability** -- a whole class of model cannot be estimated at all.
2. **Inconsistency** -- the same statsmodels-shaped call means different things, or takes
   different keyword names, depending on which estimator it is made against.
3. **Contract** -- a signature accepts something and then does not honour it.

Design rule: every claim must be executable.  A gap that cannot be shown with a five-line
snippet is a matter of taste and does not belong in the report.

Three outcomes, not two
-----------------------
``GAP`` (the limitation is there), ``FIXED`` (the call works now, so the report is
stale), ``ERROR`` (unexpected -- probe bug or a changed exception type).  Collapsing the
last two would let a *closed* gap keep being reported as open, which is the kind of stale
claim a probe exists to prevent.

Usage::

    .venv/Scripts/python.exe replication/papers/adh2013/probe_gaps.py
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402
from westata.report import to_latex, to_markdown  # noqa: E402


class Limit(Exception):
    """Raised by a probe when the limitation it describes is present."""


class Works(Exception):
    """Raised by a probe when the feature it describes works, so the entry is stale.

    Controls raise this on success rather than returning normally, so one ``except``
    chain can classify every outcome without the control needing an inverted notion of
    which branch means what.
    """


@dataclass
class Gap:
    """One probe: a claim about the API, and the code that settles it."""

    priority: str
    name: str
    claim: str
    exercise: Callable[[], str]

    def run(self) -> tuple[str, str]:
        """Return ``(status, detail)`` where status is GAP, FIXED or ERROR."""

        try:
            return "FIXED", self.exercise()
        except Limit as exc:
            return "GAP", str(exc)
        except Works as exc:
            return "FIXED", str(exc)
        except Exception as exc:  # noqa: BLE001 - a probe must never crash the sweep
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


def _frame(seed: int = 97, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, n),
            "c1": np.repeat(np.arange(20), n // 20),
            "c2": np.repeat(np.arange(10), n // 10),
            "firm": np.repeat(np.arange(n // 10), 10),
            "year": np.tile(np.arange(10), n // 10),
        }
    )
    # ``x2`` is endogenous and ``z`` a genuine instrument for it: a random ``z`` would
    # trigger westata's weak-instrument warning on every probe that fits an IV model,
    # which is noise in a probe about API shape rather than instrument quality.
    df["x2"] = 0.8 * df["z"] + rng.normal(size=n)
    df["y"] = 1.0 + 0.5 * df["x1"] + 0.7 * df["x2"] + rng.normal(size=n)
    return df



def ws_dataframe() -> pd.DataFrame:
    """A small frame with an id column, for the controls that need a panel."""

    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": np.repeat(np.arange(n // 10), 10),
            "year": np.tile(np.arange(10), n // 10),
        }
    )
    frame["y"] = 1.0 + 0.5 * frame["x1"] + rng.normal(size=n)
    return frame

DF = _frame()
FIT = ws.OLS.from_formula("y ~ x1 + x2", DF).fit()


def _must_fail(label: str, call: Callable[[], object]) -> str:
    """The limitation is present iff ``call`` raises; otherwise the entry is stale."""

    try:
        call()
    except Limit:
        raise
    except Exception as exc:
        raise Limit(f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}") from exc
    raise Works(f"{label} now succeeds")


def _must_be_absent(label: str, present: bool) -> str:
    if present:
        raise Works(f"{label} now exists")
    raise Limit(f"{label} is absent")


def _must_work(label: str, call: Callable[[], object]) -> str:
    """The capability works iff ``call`` does NOT raise. Used by the controls.

    The inverse of ``_must_fail``: a control asserts that something the report now credits
    actually happens, so a regression -- not an improvement -- is what makes it fail.
    """

    try:
        call()
    except Exception as exc:  # noqa: BLE001 - any failure means the capability regressed
        raise Limit(f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}") from exc
    raise Works(f"{label} works")


def _etable_cannot_write() -> str:
    """Can ``etable`` write the table to a file, or must the caller do it?

    Checked by CALLING it and looking for the file, not by reading a signature. The earlier
    version inspected ``to_latex``/``to_markdown`` for a ``filename=`` parameter -- but the
    gap list's own recommendation was to add the capability to ``etable`` rather than to
    change those two functions, so the probe was looking in a place the fix was never going
    to touch. It would have reported the gap as open forever.
    """

    import pathlib
    import tempfile

    directory = pathlib.Path(tempfile.mkdtemp())
    target = directory / "probe_table.csv"
    try:
        ws.etable([FIT], file=target)
    except TypeError as error:
        raise Limit(f"etable(file=...) is rejected: {error}") from error
    except Exception:  # noqa: BLE001 - accepted the argument, failed later
        raise Works("etable accepts file=") from None
    if target.exists() and target.stat().st_size > 0:
        raise Works("etable(file=...) writes the table")
    raise Limit("etable accepted file= but wrote nothing")


def _cluster_key_inconsistent() -> str:
    """Which single key works per estimator, for ``cov_type="cluster"``?"""

    calls = {
        "OLS": lambda kw: ws.OLS.from_formula("y ~ x1", DF).fit(
            cov_type="cluster", cov_kwds=kw
        ),
        "IV2SLS": lambda kw: ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z]", DF).fit(
            cov_type="cluster", cov_kwds=kw
        ),
        "FEOLS": lambda kw: ws.FEOLS.from_formula("y ~ x1 | firm", DF).fit(
            cov_type="cluster", cov_kwds=kw
        ),
        "PanelOLS": lambda kw: ws.PanelOLS.from_formula(
            "y ~ x1", DF, entity="firm", time="year"
        ).fit(cov_type="cluster", cov_kwds=kw),
    }
    accepted: dict[str, list[str]] = {}
    for name, call in calls.items():
        ok = []
        for key in ("groups", "cluster", "clusters"):
            try:
                call({key: DF["c1"]})
                ok.append(key)
            except Exception:  # noqa: BLE001 - rejection is the signal
                pass
        accepted[name] = ok

    # "groups" works everywhere, so a *single* supported key is not the problem.  The
    # problem is that the three aliases disagree about which estimators accept them:
    # a user who writes `clusters=` gets a working model on IV2SLS and an error on OLS,
    # for the same semantic request.  Measure that spread.
    aliases = ("cluster", "clusters")
    spread = {key: [name for name, ok in accepted.items() if key in ok] for key in aliases}
    uniform = all(len(ok) in (0, len(accepted)) for ok in spread.values())
    if uniform:
        raise Works(
            "the alias keys are accepted uniformly; entry is stale "
            f"(groups everywhere; {spread})"
        )
    detail = ", ".join(f"{key} accepted by {names or 'none'}" for key, names in spread.items())
    raise Limit(f"'groups' works everywhere, but the aliases do not: {detail}")


def _subset_inconsistent() -> str:
    """Which estimators ACCEPT ``subset=`` on ``from_formula``?

    The check CALLS ``from_formula`` with ``subset=`` rather than reading the signature.
    That distinction matters: ``IV2SLS.from_formula`` takes ``**kwargs`` and pops
    ``subset`` out of it, so ``inspect.signature`` never lists the parameter even though
    the call works. An earlier version read the signature, reported the IV family as
    missing ``subset=``, and would have sent someone to "fix" a working path.
    """

    names = ("OLS", "IV2SLS", "IVLIML", "FEOLS", "PanelOLS", "Logit", "Poisson")
    has, lacks = [], []
    for name in names:
        cls = getattr(ws, name, None)
        if cls is None:
            continue
        try:
            _probe_subset_call(cls)
        except TypeError as error:
            # Only a SIGNATURE refusal counts as absent; any other TypeError means the
            # argument was accepted and something else went wrong.
            (lacks if "subset" in str(error) else has).append(name)
        except Exception:  # noqa: BLE001 - accepted the argument, failed on the data
            has.append(name)
        else:
            has.append(name)

    if not lacks:
        # Uniformly present: the entry is stale, which is what ``Works`` reports.
        raise Works(f"subset= is accepted by every estimator (checked by calling it)")
    if not has:
        raise Limit("subset= is accepted by none of the estimators")
    raise Limit(f"present on {has}, absent on {lacks}")


def _probe_subset_call(cls: object) -> None:
    """Call ``from_formula`` with ``subset=`` on a tiny frame; raise only on refusal."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "e": rng.normal(size=n),
            "grp": np.repeat(np.arange(6), 10),
            "t": np.tile(np.arange(1, 11), 6),
        }
    )
    mask = frame["x"] > -10.0
    name = getattr(cls, "__name__", "")

    if name in ("IV2SLS", "IVLIML", "IVGMM"):
        cls.from_formula("y ~ 1 + [e ~ z]", frame, subset=mask)
    elif name == "FEOLS":
        cls.from_formula("y ~ x | grp", frame, subset=mask)
    elif name == "PanelOLS":
        cls.from_formula("y ~ x", frame, entity="grp", time="t", subset=mask)
    else:
        cls.from_formula("y ~ x", frame, subset=mask)


# ---------------------------------------------------------------- the seven entries
GAPS: tuple[Gap, ...] = (
    # --- 1. capability -----------------------------------------------------------
    # ``weighted IV`` USED to be the P0 gap (IV rejected analytic weights). It was fixed
    # upstream while this replication was being written, so it is now a CONTROL below --
    # and the control checks the number, not just that the call succeeds, because a
    # weighting bug leaves the coefficient right and the standard error wrong.
    # ``FEOLS has no predict`` USED to be here. It was implemented upstream (01c3ce5's
    # sibling work), so it is a CONTROL below -- asserting the fitted values are right, not
    # merely that the call stopped raising.
    # --- 2. inconsistency --------------------------------------------------------
    # ``cluster kwds key`` USED to live here. It was fixed (all three spellings now work
    # on every estimator), so it moved to CONTROLS below: a gap list that keeps listing a
    # closed gap deters work that would now succeed, which the guard test is designed to
    # prevent. See ../../gaps/adh2013_gaps.md §6.1.
    # ``margins at= scalar only`` USED to live here and is a CONTROL below.
    # ``predict exog shape`` and ``etable file export`` likewise -- see §6.
)


# ------------------------------------------------------------------- controls
def _control_two_way_hdfe() -> str:
    """Two-way clustering on HDFE works, so this raises ``Works`` and reports FIXED."""

    ws.FEOLS.from_formula("y ~ x1 | firm + year", DF).fit(
        cov_type="cluster", cov_kwds={"cluster": ["c1", "c2"]}
    )
    raise Works("two-way clustering on HDFE works, as expected")


def _control_bootstrap_seed() -> str:
    """Bootstrap accepts ``reps``/``seed`` and reproduces under a fixed seed."""

    first = ws.Bootstrap.from_formula("y ~ x1", DF, reps=25, seed=12345).fit()
    again = ws.Bootstrap.from_formula("y ~ x1", DF, reps=25, seed=12345).fit()
    other = ws.Bootstrap.from_formula("y ~ x1", DF, reps=25, seed=999).fit()
    if not np.allclose(first.bse, again.bse):
        raise Works("bootstrap is NOT reproducible under a fixed seed")
    if np.allclose(first.bse, other.bse):
        raise Works("changing the seed did not change the result, so it is ignored")
    raise Works("bootstrap honours reps= and seed=, and reproduces exactly")


def _control_ols_subset() -> str:
    """``subset=`` works on OLS, so the P1 entry is about *coverage*, not absence."""

    result = ws.OLS.from_formula("y ~ x1", DF, subset=DF["x1"] > 0).fit()
    if result.nobs >= len(DF):
        raise Works("OLS subset= was silently ignored")
    raise Works(f"OLS subset= restricts the sample as documented (nobs={result.nobs})")


CONTROL = Gap(
    "control", "HDFE two-way cluster", "control: two-way HDFE clustering works",
    _control_two_way_hdfe,
)
CONTROL_BOOTSTRAP = Gap(
    "control", "bootstrap seed", "control: bootstrap honours reps=/seed=",
    _control_bootstrap_seed,
)
CONTROL_SUBSET = Gap(
    "control", "subset= coverage",
    "control: subset= is accepted uniformly (was a P1 gap, now fixed)",
    _subset_inconsistent,
)

#: Was a P1 gap until the three key spellings were accepted uniformly on every estimator.
#: Kept as a control so a REGRESSION (one estimator silently dropping an alias again) is
#: caught: ``_cluster_key_inconsistent`` returns FIXED when uniform, and the guard test
#: requires every control to be FIXED.
CONTROL_CLUSTER_KEYS = Gap(
    "control", "cluster kwds keys",
    "control: groups/clusters/cluster all work on every estimator",
    _cluster_key_inconsistent,
)

def _control_weighted_iv() -> str:
    """Weighted IV reproduces Stata's coefficient AND standard error on real data.

    This used to be the report's P0 gap; the support was then implemented and committed
    (01c3ce5).  It is a control now, and the assertion is on
    the *number* rather than on "the call no longer raises": a weighting bug typically
    leaves the point estimate exact and corrupts only the covariance, so a probe that
    stopped at "it ran" would certify a broken estimator.
    """

    import pandas as pd

    dta = (
        REPO / "replication" / "papers" / "adh2013" / "data"
        / "Autor-Dorn-Hanson-ChinaSyndrome-FileArchive" / "dta" / "workfile_china.dta"
    )
    if not dta.exists():
        raise Works("workfile_china.dta absent; weighted-IV control skipped")
    frame = pd.read_stata(dta, convert_categoricals=False)
    regs = [c for c in frame.columns if c.startswith("reg")]
    ctrls = [
        "l_shind_manuf_cbp", "l_sh_popedu_c", "l_sh_popfborn", "l_sh_empl_f",
        "l_sh_routine33", "l_task_outsource", *regs, "t2",
    ]
    formula = (
        "d_sh_empl_mfg ~ " + " + ".join(ctrls)
        + " + [d_tradeusch_pw ~ d_tradeotch_pw_lag]"
    )
    result = ws.IV2SLS.from_formula(
        formula, frame, weights=frame["timepwt48"], weight_type="aweight"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["statefip"]})
    index = list(result.param_names).index("d_tradeusch_pw")
    coef = float(np.asarray(result.params, dtype=float)[index])
    se = float(np.asarray(result.bse, dtype=float)[index])
    if abs(coef - (-0.5963600661342254)) > 1e-9:
        raise Works(f"weighted IV coefficient drifted to {coef!r}")
    if abs(se - 0.09877388299580893) > 1e-6:
        raise Works(
            f"weighted IV standard error is {se!r}, not Stata's 0.09877388299; the "
            "coefficient is right but the covariance is wrong"
        )
    # The implementation must be in HEAD, not only in the working tree: the gap list is
    # read by people who will `git clone`, and they get HEAD.  The support was uncommitted
    # for a while during this replication, which is exactly the state this assertion
    # distinguishes -- "works here" is not the same claim as "works for a reader".
    import subprocess

    committed = subprocess.run(
        ["git", "show", "HEAD:src/westata/estimators/iv.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    if committed.returncode == 0 and "normalize_weights" not in committed.stdout:
        raise Works(
            "weighted IV works in the working tree but is NOT committed; a fresh clone "
            "still raises UnsupportedError, so the gap list must keep calling it open"
        )
    raise Works("weighted IV reproduces Stata's b and se; support is in HEAD")


#: Weighted IV works in THIS working tree but the change is not committed, so the control
#: asserts two things: the numbers reproduce Stata's, AND the source that makes them
#: reproduce is actually in HEAD.  Without the second assertion a reader who clones the
#: repo gets an `UnsupportedError` while this file says the gap is closed.
CONTROL_WEIGHTED_IV = Gap(
    "control", "weighted IV", "control: weighted IV reproduces Stata's b and se",
    _control_weighted_iv,
)

def _control_feols_predict() -> str:
    """``reghdfe`` now predicts, and the fitted values include the absorbed effects.

    The assertion is numeric rather than "it no longer raises": a fixed-effects model can
    return ``xb`` that silently omits the absorbed intercept shift, which looks plausible
    and is wrong.  ``xb`` is checked against ``y - residuals`` on the same rows, which is
    a within-model identity and holds only if both parts are computed consistently.
    """

    frame = ws_dataframe()
    result = ws.FEOLS.from_formula("y ~ x1 | firm", frame).fit()
    # Same distinction as the weighted-IV control: "works here" is weaker than "works for
    # a reader who clones HEAD". The check reads HEAD at RUN time rather than asserting a
    # fixed state, so it stays correct as commits land -- an earlier version of the prose
    # said "not committed" and went stale the moment the fix was pushed.
    import subprocess

    committed = subprocess.run(
        ["git", "show", "HEAD:src/westata/estimators/hdfe.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    in_head = committed.returncode == 0 and "_register_hdfe_predict" in committed.stdout

    xb = np.asarray(result.predict(kind="xb"), dtype=float)
    resid = np.asarray(result.predict(kind="residuals"), dtype=float)
    y = frame["y"].to_numpy(dtype=float)[: len(xb)]
    if not np.allclose(xb + resid, y, atol=1e-8):
        raise Works("FEOLS predict: xb + residuals does not equal y; the absorbed part is off")
    state = "committed" if in_head else "working tree only, NOT committed"
    raise Works(f"FEOLS predict works and xb + residuals == y ({state})")


CONTROL_FEOLS_PREDICT = Gap(
    "control", "FEOLS predict", "control: reghdfe predicts fitted values and residuals",
    _control_feols_predict,
)

#: The three P1/P2 entries that closed last. Kept as controls so a REGRESSION reopens the
#: suite rather than passing quietly -- each asserts the capability still works, which is
#: the same assertion the gap used to invert.
CONTROL_MARGINS_AT = Gap(
    "control", "margins at= sequence",
    "control: margins(at=) accepts a sequence of values",
    lambda: _must_work(
        "margins at= with a list", lambda: FIT.margins(at={"x1": [0, 1, 2]})
    ),
)
CONTROL_PREDICT_EXOG = Gap(
    "control", "predict exog DataFrame",
    "control: predict(exog=DataFrame) selects the model columns by name",
    lambda: _must_work(
        "predict(exog=DataFrame)", lambda: FIT.predict(exog=DF.head(5))
    ),
)
CONTROL_ETABLE_FILE = Gap(
    "control", "etable file export",
    "control: etable(file=...) writes the table",
    _etable_cannot_write,
)

CONTROLS = (CONTROL, CONTROL_BOOTSTRAP, CONTROL_SUBSET, CONTROL_WEIGHTED_IV,
            CONTROL_CLUSTER_KEYS, CONTROL_FEOLS_PREDICT,
            CONTROL_MARGINS_AT, CONTROL_PREDICT_EXOG, CONTROL_ETABLE_FILE)


#: Entries deleted from an earlier draft, kept as a record of *why* the framing changed.
#: Their removal is the point: they asked westata to resemble Stata, not to be better on
#: its own terms.
REMOVED = (
    ("i./ib#./c./##/L./D. syntax",
     "westata uses patsy by design; C(x, base=) expresses i. fully"),
    ("lincom _b[x]",
     "_b[] is Stata's display syntax; westata's parameter names are the natural spelling"),
    ("predict kind aliases (resid/se)",
     "serves Stata habits, not statsmodels users"),
    ("global set seed",
     "RNG state belongs to the estimation call, not the library's globals"),
    ("log using",
     "users have logging/redirect_stdout; the library need not reimplement it"),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"GAP": 0, "FIXED": 0, "ERROR": 0}
    for gap in GAPS:
        status, detail = gap.run()
        counts[status] += 1
        print(f"  [{status:5s}] {gap.priority:2s}  {gap.name:26s} {detail}")

    print()
    controls_ok = True
    for control in CONTROLS:
        status, detail = control.run()
        print(f"  control [{status}] {control.name}: {detail}")
        controls_ok = controls_ok and status == "FIXED"

    print(
        f"\n{counts['GAP']} gap(s) present, {counts['FIXED']} stale report entr(y/ies), "
        f"{counts['ERROR']} unexpected failure(s)."
    )
    if not controls_ok:
        print("  WARNING: a control probe misbehaved; distrust the list above.")

    print("\nRemoved from an earlier draft (asked for Stata resemblance, not better API):")
    for name, reason in REMOVED:
        print(f"  - {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
