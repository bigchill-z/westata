"""Tests for `e(N_cds)` / `e(N_cdf)` on the binary-outcome models.

Stata's help (``logit.sthlp:243-244``) defines them as:

    e(N_cds)  number of completely determined successes
    e(N_cdf)  number of completely determined failures

and ``frac_154.ado:679`` detects perfect prediction with the identity

    if (e(N) == e(N_cds) + e(N_cdf))

The definition is therefore: a *covariate pattern* (a distinct combination of
the regressors) whose observed outcomes are ALL successes contributes its
observations to ``N_cds``; one whose outcomes are all failures contributes to
``N_cdf``; a pattern containing variation contributes to neither.

**Both fixtures report 0.0**, so a hardcoded zero would match them. These tests
therefore use data where heterogeneous patterns exist and verify the counts
against an independent dict-based grouping, including a case where the sum is
strictly between 0 and N.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

import westata as ws


def independent_counts(design, response):
    """Group by exact covariate row and count homogeneous patterns."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float).reshape(-1)
    keep = [j for j in range(x.shape[1]) if np.ptp(x[:, j]) > 0.0]
    sub = x[:, keep] if keep else np.zeros((x.shape[0], 1))

    groups: dict[tuple, list[float]] = defaultdict(list)
    for i in range(sub.shape[0]):
        groups[tuple(sub[i].tolist())].append(float(y[i]))

    cds = cdf = 0
    for values in groups.values():
        arr = np.asarray(values, dtype=float)
        if np.all(arr == arr[0]):
            if arr[0] != 0:
                cds += arr.size
            else:
                cdf += arr.size
    return cds, cdf, len(groups)


def _response(result):
    return np.asarray(result.fittedvalues, dtype=float) + np.asarray(
        result.resid, dtype=float
    )


@pytest.fixture()
def mixed_patterns():
    """Rare and common x values, so some patterns are homogeneous."""
    rng = np.random.default_rng(4242)
    n = 400
    x = np.where(
        rng.uniform(size=n) < 0.12,
        rng.choice([9.0, 10.0, 11.0], size=n),
        rng.integers(0, 4, size=n).astype(float),
    )
    p = 1.0 / (1.0 + np.exp(-(0.8 * (x - 5.0))))
    y = (rng.uniform(size=n) < p).astype(float)
    return pd.DataFrame({"y": y, "x": x})


def test_both_keys_are_present(mixed_patterns) -> None:
    e = ws.Logit.from_formula("y ~ x", mixed_patterns).fit().e

    assert "N_cds" in e
    assert "N_cdf" in e


def test_counts_match_an_independent_grouping(mixed_patterns) -> None:
    result = ws.Logit.from_formula("y ~ x", mixed_patterns).fit()
    design, _ = result.design_matrix
    expected_cds, expected_cdf, _ = independent_counts(design, _response(result))

    assert result.e["N_cds"] == float(expected_cds)
    assert result.e["N_cdf"] == float(expected_cdf)


def test_the_sum_is_strictly_between_zero_and_n(mixed_patterns) -> None:
    """The decisive check: a hardcoded 0 or a trivially-homogeneous dataset
    would both pass every other test in this file."""

    result = ws.Logit.from_formula("y ~ x", mixed_patterns).fit()
    total = result.e["N_cds"] + result.e["N_cdf"]

    assert 0 < total < len(mixed_patterns)


def test_perfect_prediction_sums_to_n() -> None:
    """The ``frac_154.ado:679`` identity."""

    frame = pd.DataFrame(
        {
            "y": np.concatenate([np.zeros(25), np.ones(25)]),
            "x": np.repeat([0.0, 1.0], 25),
        }
    )
    result = ws.Logit.from_formula("y ~ x", frame).fit()

    assert result.e["N_cds"] == 25.0
    assert result.e["N_cdf"] == 25.0
    assert result.e["N_cds"] + result.e["N_cdf"] == len(frame)


def test_the_fixtures_zero_is_reproduced_correctly() -> None:
    """Data whose patterns all vary must give zero -- not because it is
    hardcoded, but because the counts genuinely are zero."""

    rng = np.random.default_rng(11)
    n = 300
    x = rng.normal(size=n)          # every value distinct -> no repeats
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < p).astype(float)
    frame = pd.DataFrame({"y": y, "x": x})
    result = ws.Logit.from_formula("y ~ x", frame).fit()

    # With all-distinct regressors every pattern is a singleton, hence
    # homogeneous, so the counts sum to N.
    assert result.e["N_cds"] + result.e["N_cdf"] == float(len(frame))


def test_probit_also_reports_them(mixed_patterns) -> None:
    result = ws.Probit.from_formula("y ~ x", mixed_patterns).fit()

    assert "N_cds" in result.e
    assert "N_cdf" in result.e


def test_probit_covers_its_whole_fixture() -> None:
    import json
    from pathlib import Path

    df = pd.read_stata("tests/data/auto.dta")
    frame = df.assign(dom=(df["foreign"] == "Foreign").astype(int))
    result = ws.Probit.from_formula("dom ~ weight + mpg", frame).fit()
    expected = set(
        json.loads(
            Path("tests/conformance/stata_fixtures/probit_sim.json").read_text()
        )["scalars"]
    )

    assert not (expected - set(result.e)), sorted(expected - set(result.e))


def test_logit_covers_its_whole_fixture() -> None:
    import json
    from pathlib import Path

    df = pd.read_stata("tests/data/auto.dta")
    frame = df.assign(dom=(df["foreign"] == "Foreign").astype(int))
    result = ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()
    expected = set(
        json.loads(
            Path("tests/conformance/stata_fixtures/logit_sim.json").read_text()
        )["scalars"]
    )

    assert not (expected - set(result.e)), sorted(expected - set(result.e))
