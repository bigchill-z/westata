"""``e(vce)`` and ``e(vcetype)`` -- two DIFFERENT strings for the same request.

``regress.sthlp:437-438`` defines both:

    {synopt:{cmd:e(vce)}}     {it:vcetype} specified in {cmd:vce()}
    {synopt:{cmd:e(vcetype)}} title used to label Std. err.

They are not synonyms, and the difference is load-bearing in Stata's own source --
``arch.ado:1470`` and ``archlm.ado:16`` both write::

    if "`e(vcetype)'" == "Robust" { ... }

So ``e(vcetype)`` is the CAPITALISED DISPLAY TITLE while ``e(vce)`` holds the lowercase
option token. Recording the same string for both would make any ported comparison fail
silently, which is the failure mode this file guards.

The HC forms are TWO WORDS, taken verbatim from the ados rather than built:

    didregress.ado:477   ereturn local vcetype "Robust HC2"
    nl.ado:1115          ereturn local vcetype "Robust HC2"
    nl.ado:1119          ereturn local vcetype "Robust HC3"

A naive ``token.capitalize()`` would produce ``"Hc2"``, which matches nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.linear import _stata_vce_label, _stata_vcetype_label

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


def _frame(n: int = 250, seed: int = 7400):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "x": covariate,
            "g": np.arange(n) % 20,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


def _fit(frame, cov_type=None, cov_kwds=None):
    if cov_type is None:
        return ws.OLS.from_formula("y ~ x", frame).fit()
    return ws.OLS.from_formula("y ~ x", frame).fit(
        cov_type=cov_type, cov_kwds=cov_kwds or {}
    )


# ------------------------------------------------------------------ e(vce)
def test_the_default_is_ols_not_nonrobust(frame) -> None:
    """``regress.ado:68``: ``local vce = cond(..., "ols")``.

    This package's internal name for the default is ``nonrobust``; Stata's token is
    ``ols``. Passing the internal name through would be wrong.
    """

    assert _fit(frame).e["vce"] == "ols"


def test_cluster_is_recorded(frame) -> None:
    fitted = _fit(frame, "cluster", {"groups": frame["g"]})

    assert fitted.e["vce"] == "cluster"


@pytest.mark.parametrize("hc", ["HC2", "HC3"])
def test_the_hc_tokens_are_lowercase(frame, hc: str) -> None:
    """``hc2``/``hc3`` keep their own names.

    ``HC1`` is deliberately NOT in this list: Stata has no ``vce(hc1)``. Its HC1 is
    spelled ``vce(robust)`` and stores ``e(vce)="robust"``, so asserting "hc1" here would
    pin a token Stata cannot produce. That case is covered separately below.
    """

    assert _fit(frame, hc).e["vce"] == hc.lower()


def test_hc1_is_spelled_robust(frame) -> None:
    """The package's ``HC1`` IS Stata's ``vce(robust)`` -- verified numerically and by
    the macro. Stata rejects ``vce(hc1)`` as a non-existent option."""

    fitted = _fit(frame, "HC1")

    assert fitted.e["vce"] == "robust"
    assert fitted.e["vcetype"] == "Robust"


# --------------------------------------------------------------- e(vcetype)
def test_the_vcetype_is_capitalised(frame) -> None:
    """The value Stata's own ``if`` statements compare against."""

    # Empty, not "Ols" -- see the note on _stata_vcetype_label.
    assert _fit(frame).e["vcetype"] == ""


def test_cluster_vcetype_is_capitalised(frame) -> None:
    fitted = _fit(frame, "cluster", {"groups": frame["g"]})

    assert fitted.e["vcetype"] == "Cluster"


@pytest.mark.parametrize("hc", ["HC2", "HC3"])
def test_the_hc_vcetype_is_two_words(frame, hc: str) -> None:
    """``"Robust HC2"`` -- capitalised, with the ``HC`` part upper-case as well.

    ``HC1`` is excluded because its vcetype is the single word ``"Robust"``; only the
    hc2/hc3 estimators are named after their option.
    """

    assert _fit(frame, hc).e["vcetype"] == f"Robust {hc.upper()}"


def test_the_hc_vcetype_is_not_a_naive_capitalisation(frame) -> None:
    """Guard: ``"Hc2"`` is what ``capitalize()`` would give, and it is wrong."""

    recorded = _fit(frame, "HC2").e["vcetype"]

    assert recorded != "Hc2"
    assert recorded == "Robust HC2"


# --------------------------------------------------- the two are NOT the same
def test_the_two_macros_differ_for_every_request(frame) -> None:
    """THE reason both exist: one is a token, the other a display title."""

    cases = [
        (None, None),
        ("HC1", None),
        ("HC2", None),
        ("cluster", {"groups": frame["g"]}),
    ]
    for cov_type, cov_kwds in cases:
        e = _fit(frame, cov_type, cov_kwds).e
        assert e["vce"] != e["vcetype"], cov_type


def test_the_robust_string_matches_what_stata_compares(frame) -> None:
    """``arch.ado:1470`` compares ``e(vcetype)`` against the literal ``"Robust"``.

    An HC request must therefore produce a value that STARTS with "Robust".
    """

    for hc in ("HC1", "HC2", "HC3"):
        assert _fit(frame, hc).e["vcetype"].startswith("Robust")


# ------------------------------------------------------------ the evidence
def test_the_ados_compare_vcetype_capitalised() -> None:
    for relative in ("a/arch.ado", "a/archlm.ado"):
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert 'e(vcetype)\'"\' == "Robust"' in text or '"Robust"' in text, relative


def test_the_hc_vcetype_literals_are_in_the_ados() -> None:
    for relative, literal in (
        ("d/didregress.ado", 'ereturn local vcetype "Robust HC2"'),
        ("n/nl.ado", 'ereturn local vcetype "Robust HC3"'),
    ):
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert literal in text, relative


def test_the_help_file_lists_both_macros() -> None:
    text = (BASE / "r/regress.sthlp").read_text(encoding="utf-8", errors="ignore")

    assert "{cmd:e(vce)}" in text
    assert "{cmd:e(vcetype)}" in text


# ------------------------------------------------------------- the helpers
@pytest.mark.parametrize(
    "raw,expected_token",
    [
        ("nonrobust", "ols"),
        # Stata has NO ``vce(hc1)``: the option is ``vce(robust)``, and that is the
        # estimator this package calls HC1. Checked against StataNow 19.5 --
        # ``regress, vce(hc1)`` is rejected as a non-existent option, while
        # ``vce(robust)`` stores e(vce)="robust". Reporting "hc1" put a token in the
        # macro that Stata can never produce.
        ("HC1", "robust"),
        ("cluster", "cluster"),
        ("hac", "hac"),
    ],
)
def test_the_token_helper(raw: str, expected_token: str) -> None:
    assert _stata_vce_label(raw) == expected_token


@pytest.mark.parametrize(
    "raw,expected_title",
    [
        # e(vcetype) is EMPTY for least squares: ``regress y s`` leaves it blank, and
        # Stata's own ado compares it against "Robust" to choose a display path, so a
        # fabricated "Ols" is actively misleading.
        ("nonrobust", ""),
        # ``vce(robust)`` yields exactly "Robust"; only the hc2/hc3 forms are two words.
        ("HC1", "Robust"),
        ("cluster", "Cluster"),
        ("hac", "Hac"),
    ],
)
def test_the_title_helper(raw: str, expected_title: str) -> None:
    assert _stata_vcetype_label(raw) == expected_title