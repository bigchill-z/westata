"""The report rows that had no guard, and why.

At the time this file was written, two of the 90 scored rows were missing
FEATURES rather than unverified ones::

    C16-02   rdrobust bandwidth selection (bwselect, msetwo)   0.0
    C18-02   synth2                                             0.0

``bwselect`` still does not exist.  Both ``synth`` (C18-01) and ``synth2``
(C18-02) have since landed, so their absence guards were deliberately
converted to presence + behaviour guards rather than deleted -- a test
asserting a landed estimator fails would be actively wrong.

* ``synth`` landed 2026-11-09 (B-grade default path, then the extended
  non-graphics feature set): docs/superpowers/specs/2026-11-09-synth-b-design.md
* ``synth2`` landed 2026-11-11 (non-graphics feature set: fit statistics,
  balance table, in-space/in-time placebo, LOO):
  docs/superpowers/specs/2026-11-11-synth2-design.md

Graphics remain out of scope for both.
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


def test_synth2_is_now_implemented() -> None:
    """synth2's NON-GRAPHICS feature set landed; graphics remain excluded."""

    assert hasattr(ws, "synth2")
    assert callable(ws.synth2)
    assert hasattr(ws, "Synth2Result")


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
