"""Tests for Stata's ``estat classification`` (binary-outcome classification table).

Transcribed from ``lstat.ado`` v3.2.0, which ``logit_estat`` forwards
``classification`` to. The procedure is a 2x2 table of observed against predicted
at ``cutoff()`` (default 0.5), plus nine percentages on a 0-100 scale.

Besides the direct transcription, this file asserts four *complementary
identities* that must hold on any data:

    sensitivity          + false - rate given D   == 100
    specificity          + false + rate given ~D  == 100
    pos. predictive value + false + given +       == 100
    neg. predictive value + false - given -       == 100

Each pair partitions one conditional distribution, so a swapped numerator or
denominator in any of the eight fails here even if the direct comparison were
somehow transcription-consistent.

A cutoff of 0 must classify everything positive (sensitivity 100, specificity 0),
which pins the direction of the ``>=`` comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def logit_fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df.assign(dom=(df["foreign"] == "Foreign").astype(int))
    return frame, ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()


def _counts(result, cutoff=0.5):
    p = np.asarray(result.predict(kind="pr"), dtype=float)
    y = np.asarray(result.fittedvalues, dtype=float) + np.asarray(result.resid, dtype=float)
    observed = y != 0
    classified = p >= cutoff
    return (
        int(np.sum(observed & classified)),
        int(np.sum(~observed & classified)),
        int(np.sum(observed & ~classified)),
        int(np.sum(~observed & ~classified)),
    )


def test_the_table_matches_the_four_counts(logit_fit) -> None:
    _, result = logit_fit
    a, b, c, d = _counts(result)

    table = result.estat_classification().table.to_numpy()

    assert table[0, 0] == a
    assert table[0, 1] == b
    assert table[1, 0] == c
    assert table[1, 1] == d


def test_table_margins_are_the_row_and_column_sums(logit_fit) -> None:
    _, result = logit_fit
    a, b, c, d = _counts(result)

    table = result.estat_classification().table.to_numpy()

    assert table[0, 2] == a + b  # classified + total
    assert table[1, 2] == c + d  # classified - total
    assert table[2, 0] == a + c  # true D total
    assert table[2, 1] == b + d  # true ~D total
    assert table[2, 2] == a + b + c + d


def test_the_nine_percentages(logit_fit) -> None:
    _, result = logit_fit
    a, b, c, d = _counts(result)
    cr = result.estat_classification()

    assert cr.percent_correct == pytest.approx((a + d) / (a + b + c + d) * 100)
    assert cr.sensitivity == pytest.approx(a / (a + c) * 100)
    assert cr.specificity == pytest.approx(d / (b + d) * 100)
    assert cr.false_positive_rate == pytest.approx(b / (b + d) * 100)
    assert cr.false_negative_rate == pytest.approx(c / (a + c) * 100)
    assert cr.positive_predictive_value == pytest.approx(a / (a + b) * 100)
    assert cr.negative_predictive_value == pytest.approx(d / (c + d) * 100)
    assert cr.false_positive_classified == pytest.approx(b / (a + b) * 100)
    assert cr.false_negative_classified == pytest.approx(c / (c + d) * 100)


def test_complementary_identities(logit_fit) -> None:
    """Each pair must sum to exactly 100 on any data."""

    cr = logit_fit[1].estat_classification()

    assert cr.sensitivity + cr.false_negative_rate == pytest.approx(100.0)
    assert cr.specificity + cr.false_positive_rate == pytest.approx(100.0)
    assert (
        cr.positive_predictive_value + cr.false_positive_classified
        == pytest.approx(100.0)
    )
    assert (
        cr.negative_predictive_value + cr.false_negative_classified
        == pytest.approx(100.0)
    )


def test_cutoff_zero_classifies_everything_positive(logit_fit) -> None:
    """Pins the direction of the ``>=`` comparison."""

    cr = logit_fit[1].estat_classification(cutoff=0.0)

    assert cr.sensitivity == pytest.approx(100.0)
    assert cr.specificity == pytest.approx(0.0)


def test_raising_the_cutoff_reduces_positive_classifications(logit_fit) -> None:
    frame, result = logit_fit

    low = result.estat_classification(cutoff=0.2)
    mid = result.estat_classification(cutoff=0.5)
    high = result.estat_classification(cutoff=0.8)

    assert low.table.to_numpy()[0, 2] >= mid.table.to_numpy()[0, 2]
    assert mid.table.to_numpy()[0, 2] >= high.table.to_numpy()[0, 2]
    assert low.sensitivity >= mid.sensitivity >= high.sensitivity


def test_cutoff_outside_the_unit_interval_is_refused(logit_fit) -> None:
    _, result = logit_fit

    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            result.estat_classification(cutoff=bad)


def test_non_binary_models_are_refused(logit_fit) -> None:
    frame, _ = logit_fit
    ols = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="binary-outcome"):
        ols.estat_classification()


def test_probit_is_supported(logit_fit) -> None:
    frame, _ = logit_fit
    cr = ws.Probit.from_formula("dom ~ weight + mpg", frame).fit().estat_classification()

    assert 0.0 <= cr.percent_correct <= 100.0
    assert cr.table.to_numpy()[2, 2] == len(frame)


def test_ologit_is_refused_and_the_evidence_is_recorded() -> None:
    """``ologit`` must NOT offer ``estat classification``.

    It is the tempting case: a multi-category classification table looks like a
    natural generalisation of the binary one, so adding it "for consistency" is an easy
    mistake.  The ado settles it --

    * ``ologit.ado`` never sets ``e(estat_cmd)``, and
    * ``ologit_postestimation.sthlp`` has no ``classification`` section,

    whereas ``probit.ado:259`` records ``e(estat_cmd) = "probit_estat"`` and
    ``logit_estat.ado:11`` dispatches ``classification`` from it.

    The test asserts BOTH the refusal and the source evidence, so a future addition
    has to argue with the ado rather than with this comment.
    """

    from pathlib import Path

    rng = np.random.default_rng(521)
    n = 300
    frame = pd.DataFrame(
        {"rep": rng.integers(2, 4, size=n), "x1": rng.normal(size=n)}
    )
    ordered = ws.OrderedLogit.from_formula("rep ~ x1", frame).fit()

    with pytest.raises(NotImplementedError):
        ordered.estat_classification()

    base = Path(r"D:\Program Files\StataNow19\ado\base")
    ologit_ado = (base / "o" / "ologit.ado").read_text(
        encoding="utf-8", errors="ignore"
    )
    ologit_help = (base / "o" / "ologit_postestimation.sthlp").read_text(
        encoding="utf-8", errors="ignore"
    )
    probit_ado = (base / "p" / "probit.ado").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "estat_cmd" not in ologit_ado
    assert "classification" not in ologit_help.lower()
    # ... while the binary family does route estat through a handler.
    assert "probit_estat" in probit_ado


def test_to_dict_uses_stata_r_names(logit_fit) -> None:
    payload = logit_fit[1].estat_classification().to_dict()

    assert set(payload) == {
        "P_corr",
        "P_p1",
        "P_n0",
        "P_p0",
        "P_n1",
        "P_1p",
        "P_0n",
        "P_0p",
        "P_1n",
    }


def test_explicit_inputs_match_the_defaults(logit_fit) -> None:
    _, result = logit_fit
    p = np.asarray(result.predict(kind="pr"), dtype=float)
    y = np.asarray(result.fittedvalues, dtype=float) + np.asarray(result.resid, dtype=float)

    explicit = result.estat_classification(y=y, probabilities=p)
    default = result.estat_classification()

    assert explicit.to_dict() == default.to_dict()


def test_output_mentions_the_cutoff_and_nine_lines(logit_fit) -> None:
    text = str(logit_fit[1].estat_classification())

    assert "Classified + if predicted Pr(D) >= 0.5" in text
    for label in (
        "Sensitivity",
        "Specificity",
        "Positive predictive value",
        "Negative predictive value",
        "False + rate for true ~D",
        "False - rate for true D",
        "False + rate for classified +",
        "False - rate for classified -",
        "Correctly classified",
    ):
        assert label in text, label
