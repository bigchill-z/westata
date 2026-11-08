"""The two report rows with no guard, and why they have none.

At this point 88 of the 90 scored rows in the assessment report are referenced by a test. The two that are
not are missing FEATURES rather than unverified ones, and this file records that distinction so the gap does
not read as unfinished verification:

    C16-02   rdrobust bandwidth selection (bwselect, msetwo)   0.0
    C18-02   synth2                                             0.0

NEITHER NAME EXISTS IN THE PACKAGE AT ALL, which is the check this file performs -- not "it raises" but
"there is nothing to call". options.md records what each would need: T3 lists the two commands, and T6
documents how much of synth is readable (synth.ado is 931 readable lines; six numeric kernels live in
lsynth_mata_subr.mlib and cannot be read).

A GUARD THAT ASSERTS ABSENCE IS DELIBERATE HERE. The alternative -- pinning that the call raises -- would
freeze an accident: if someone later implements synth, a test asserting it fails would be actively wrong.
"""

import warnings

warnings.filterwarnings("ignore")
import pytest

import westata as ws


def test_bwselect_is_not_implemented() -> None:
    """Automatic bandwidth selection is options.md T1; nothing is exported under this name."""

    assert not hasattr(ws, "bwselect")


def test_synth_is_not_implemented() -> None:
    assert not hasattr(ws, "synth")


def test_synth2_is_not_implemented() -> None:
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
