"""The single place that answers "was this result fitted with a robust VCE?".

R3.7.2 moved the label on a result object from the caller's input to a canonical
name: ``cov_type="robust"`` on the ML families now reports ``ML_ROBUST``, on the
panel family ``cluster``, on ``qreg`` ``QREG_ROBUST``.  That change broke every
postestimation module that had grown its OWN membership test against the literal
string ``"robust"`` -- five of them:

    hettest.py:198      cov_type in ("hc0", "hc1", "hc2", "hc3", "robust", "cluster")
    imtest.py:313       cov_type in ("hc0", "hc1", "hc2", "hc3", "robust") or "cluster" in ...
    endogenous.py:293   cov_type in ("hc0", "hc1", "hc2", "hc3", "robust")
    overid.py:383       cov_type in ("hc0", "hc1", "hc2", "hc3", "robust")
    ts_lags.py:131      cov_type in robust_names or ("robust" in cov_type and ...)

Each would have silently answered "not robust" for the new labels, and the failure
mode is the worst kind: `estat hettest` would have STOPPED refusing after a robust
fit, printing a test Stata declines to compute.  Nothing in the numeric fixtures
would have caught it, because the numbers are only produced on the wrong path.

So the test lives here, once, and is expressed as a property of the label rather
than as a list of literals that has to be edited whenever a family gains a name.
"""

from __future__ import annotations

#: Labels that mean "a heteroskedasticity-robust or clustered VCE was used".
#:
#: Listed explicitly rather than matched by substring so a future label cannot be
#: swept in -- or left out -- by accident.  Every entry is a value some estimator
#: can actually store (see ``docs/API_CONVENTIONS.md`` R3.7.1).
ROBUST_COV_TYPES = frozenset(
    {
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "hc4",
        "hc5",
        "robust",          # Stata's own spelling, still produced by some paths
        "ml_robust",       # discrete/count ML sandwich (n/(n-1)); R3.7.2
        "qreg_robust",     # qreg's tau(1-tau) sandwich; R3.7.2
        "systems_robust",  # sureg's cross-equation sandwich; R3.7.2
        "cluster",         # xtreg's vce(robust) IS cluster(panelvar); R3.7.3
        "clustered",
    }
)

#: Labels that mean a HAC / Newey-West VCE.
HAC_COV_TYPES = frozenset({"hac", "newey", "neweywest"})


def normalize_cov_label(value: object) -> str:
    """Lower-case, trimmed label of a result's ``cov_type``."""

    return str(value or "").strip().lower()


def is_robust(cov_type: object) -> bool:
    """Whether ``cov_type`` denotes a robust (heteroskedasticity or clustered) VCE."""

    label = normalize_cov_label(cov_type)
    if label in ROBUST_COV_TYPES:
        return True
    # Every clustered label contains "cluster" (``cluster``, ``clustered``,
    # ``cluster_2way``); no NON-robust label does, so the substring test is safe here
    # and keeps a future ``cluster_threeway`` from being missed.
    return "cluster" in label


def is_clustered(cov_type: object) -> bool:
    """Whether ``cov_type`` denotes a clustered VCE."""

    return "cluster" in normalize_cov_label(cov_type)


def is_hac(cov_type: object) -> bool:
    """Whether ``cov_type`` denotes a HAC / Newey-West VCE."""

    label = normalize_cov_label(cov_type)
    return label in HAC_COV_TYPES or "newey" in label
