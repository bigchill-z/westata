"""``e(N_cds)`` / ``e(N_cdf)`` -- the completely-determined observations.

``logit.sthlp:243-244`` gives the definitions outright:

    {cmd:e(N_cds)}  number of completely determined successes
    {cmd:e(N_cdf)}  number of completely determined failures

These are not decoration. Stata prints them so a user can see that the likelihood was
driven by degenerate covariate patterns -- ``e(N) == e(N_cds) + e(N_cdf)`` means EVERY
observation is perfectly predictable, which is what a separation warning is about. This
package stores them and previously only asserted that they exist.

The test below builds a sample where the answer is known by construction: one covariate
pattern that is all successes, one that is all failures, and one that is mixed. A correct
implementation counts the first two and ignores the third.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def test_the_definitions_are_in_the_help_file() -> None:
    """Read the authority rather than trusting the transcription."""

    from pathlib import Path

    text = Path(r"D:\Program Files\StataNow19\ado\base\l\logit.sthlp").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "{cmd:e(N_cds)}" in text
    assert "{cmd:e(N_cdf)}" in text
    assert "completely determined successes" in text
    assert "completely determined failures" in text


def _constructed_frame():
    """Three covariate patterns: all-success, all-failure, and mixed."""

    covariate = np.array([0.0] * 5 + [1.0] * 5 + [2.0] * 10)
    outcome = np.array([1] * 5 + [0] * 5 + [1, 0] * 5)
    return pd.DataFrame({"y": outcome, "x": covariate}), covariate, outcome


def test_all_success_patterns_are_counted() -> None:
    frame, _, _ = _constructed_frame()
    fitted = ws.Logit.from_formula("y ~ x", frame).fit()

    assert float(fitted.e["N_cds"]) == 5.0


def test_all_failure_patterns_are_counted() -> None:
    frame, _, _ = _constructed_frame()
    fitted = ws.Logit.from_formula("y ~ x", frame).fit()

    assert float(fitted.e["N_cdf"]) == 5.0


def test_the_mixed_pattern_is_not_counted() -> None:
    """The mixed pattern's ten observations are determined by neither label.

    Both counters would be higher if the implementation counted "has a success" and
    "has a failure" instead of "is all success" / "is all failure".
    """

    frame, _, _ = _constructed_frame()
    fitted = ws.Logit.from_formula("y ~ x", frame).fit()

    assert float(fitted.e["N_cds"]) + float(fitted.e["N_cdf"]) < float(fitted.e["N"])


def test_a_fully_separated_sample_is_all_completely_determined() -> None:
    """``e(N) == e(N_cds) + e(N_cdf)`` when every pattern is degenerate.

    This is the condition Stata's separation diagnostic is about, so the identity is the
    meaningful check rather than the individual counts.
    """

    covariate = np.array([0.0] * 5 + [1.0] * 5)
    outcome = np.array([1] * 5 + [0] * 5)
    frame = pd.DataFrame({"y": outcome, "x": covariate})

    fitted = ws.Logit.from_formula("y ~ x", frame).fit()

    total = float(fitted.e["N_cds"]) + float(fitted.e["N_cdf"])
    assert total == float(fitted.e["N"])


def test_a_fully_mixed_sample_has_no_determined_pattern() -> None:
    """With every pattern mixed, both counters are zero."""

    covariate = np.array([0.0, 0.0, 1.0, 1.0])
    outcome = np.array([1, 0, 1, 0])
    frame = pd.DataFrame({"y": outcome, "x": covariate})

    fitted = ws.Logit.from_formula("y ~ x", frame).fit()

    assert float(fitted.e["N_cds"]) == 0.0
    assert float(fitted.e["N_cdf"]) == 0.0


def test_the_counters_are_present_for_the_binary_family() -> None:
    """``logit``/``probit`` both document them; both must record them."""

    frame, _, _ = _constructed_frame()

    for estimator in (ws.Logit, ws.Probit):
        e = estimator.from_formula("y ~ x", frame).fit().e
        assert "N_cds" in e, estimator.__name__
        assert "N_cdf" in e, estimator.__name__


def test_the_counters_are_absent_where_the_concept_does_not_apply() -> None:
    """A count model has no "completely determined success/failure".

    Guard on the other side: if the keys leaked into every ML family, the per-command
    distinction would be lost.
    """

    rng = np.random.default_rng(6902)
    frame = pd.DataFrame(
        {"yc": rng.poisson(2.0, size=200), "x": rng.normal(size=200)}
    )
    fitted = ws.Poisson.from_formula("yc ~ x", frame).fit()

    assert "N_cds" not in fitted.e
    assert "N_cdf" not in fitted.e