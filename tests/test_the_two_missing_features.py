"""The report rows that had no guard, and why.

At the time this file was written, two of the 90 scored rows were missing
FEATURES rather than unverified ones::

    C16-02   rdrobust bandwidth selection (bwselect, msetwo)   0.0
    C18-02   synth2                                             0.0

Those two still do not exist in the package, and the guards below pin that.

``synth`` (C18-01) is DIFFERENT: it landed on 2026-11-09 as the B-grade default
path (see docs/superpowers/specs/2026-11-09-synth-b-design.md).  The original
guard asserted ``not hasattr(ws, "synth")`` with the explicit note that
"if someone later implements synth, a test asserting it fails would be
actively wrong".  It has therefore been converted to a presence + behaviour
guard rather than deleted: the estimator really is exported now.

``synth2`` remains absent on purpose -- it is a wrapper adding figures,
placebo/LOO and graphs, all outside westata's scope.
"""

import warnings

warnings.filterwarnings("ignore")
import pytest

import westata as ws


def test_bwselect_is_not_implemented() -> None:
    """Automatic bandwidth selection is options.md T1; nothing is exported under this name."""

    assert not hasattr(ws, "bwselect")


def test_synth_is_now_implemented() -> None:
    """B-grade synth landed; the former absence guard was deliberately retired."""

    assert hasattr(ws, "synth")
    assert callable(ws.synth)
    assert hasattr(ws, "SynthResult")


def test_synth2_is_not_implemented() -> None:
    """synth2 is a figure/placebo wrapper, deliberately out of scope."""

    assert not hasattr(ws, "synth2")


def test_rdrobust_itself_works_with_explicit_bandwidths() -> None:
    """The command exists; only the automatic selector is missing, and that is the distinction."""

    import numpy as np

    rng = np.random.default_rng(1)
    n = 300
    running = rng.uniform(-1.0, 1.0, n)
    response = 0.5 * (running >= 0) + 0.3 * running + rng.normal(scale=0.3, size=n)

    fitted = ws.rdrobust(response, running, cutoff=0.0, bandwidth=0.3)
    assert np.isfinite(float(fitted.tau))


def test_rdrobust_refuses_without_a_bandwidth() -> None:
    """The refusal is the evidence that the selector rather than the estimator is absent."""

    import numpy as np

    rng = np.random.default_rng(1)
    n = 300
    running = rng.uniform(-1.0, 1.0, n)
    response = 0.5 * (running >= 0) + 0.3 * running + rng.normal(scale=0.3, size=n)

    with pytest.raises(Exception):
        ws.rdrobust(response, running, cutoff=0.0)
