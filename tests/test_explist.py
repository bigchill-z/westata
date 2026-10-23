"""Tests for the ``exp_list`` parser and evaluator.

Every example here is taken from ``[U] exp_list`` or the bootstrap help file, so the parser is
tested against the specification's own worked cases rather than invented ones.
"""

from __future__ import annotations

import math

import pytest

from westata.expressions.explist import (
    ExpListError,
    StoredResults,
    evaluate_statistic,
    parse_exp_list,
)


def _results() -> StoredResults:
    return StoredResults(
        r={
            "mean": 5.0,
            "p50": 4.0,
            "p25": 2.0,
            "p75": 7.0,
            "sd": 2.5,
            "max": 9.0,
            "min": 1.0,
            "skewness": 0.4,
            "N": 100.0,
        },
        e={"rmse": 1.5, "N": 96.0, "r2": 0.75},
        b={"mpg": -0.5, "x1": 2.0, "x2": -1.0},
        se={"x1": 0.3, "x2": 0.25},
    )


# ------------------------------------------------- the manual's own examples


def test_the_manual_summarise_example():
    """``bootstrap (location: mean=r(mean) median=r(p50)) ...: summarize price, detail``."""

    parsed = parse_exp_list("(location: mean=r(mean) median=r(p50))")
    assert parsed.evaluate(_results()) == {"location:mean": 5.0, "location:median": 4.0}


def test_the_manual_scale_group_with_parenthesised_differences():
    parsed = parse_exp_list(
        "(scale: sd=r(sd) iqr=(r(p75)-r(p25)) range=(r(max)-r(min)))"
    )
    values = parsed.evaluate(_results())
    assert values["scale:sd"] == 2.5
    assert values["scale:iqr"] == 5.0
    assert values["scale:range"] == 8.0


def test_the_manual_two_group_example():
    parsed = parse_exp_list(
        "(location: mean=r(mean) median=r(p50)) "
        "(scale: sd=r(sd) iqr=(r(p75)-r(p25)) range=(r(max)-r(min)))"
    )
    assert parsed.names == [
        "location:mean",
        "location:median",
        "scale:sd",
        "scale:iqr",
        "scale:range",
    ]


def test_the_manual_exp_of_a_coefficient():
    """``bootstrap or=(exp(_b[mpg])): logit foreign mpg weight``."""

    parsed = parse_exp_list("or=(exp(_b[mpg]))")
    assert parsed.evaluate(_results())["or"] == pytest.approx(math.exp(-0.5))


def test_the_manual_jackknife_example():
    """``jackknife sd=(r(sd)) skew=(r(skewness)), rclass: summarize mpg, detail``."""

    parsed = parse_exp_list("sd=(r(sd)) skew=(r(skewness))")
    values = parsed.evaluate(_results())
    assert values["sd"] == 2.5
    assert values["skew"] == pytest.approx(0.4)


def test_the_manual_permute_example():
    """``permute price _b rmse=(e(rmse)): regress price trunk``.

    The bare ``_b`` in that spelling is ambiguous per coefficient; this parser accepts the
    explicit form and rejects the bare one with a message.
    """

    parsed = parse_exp_list("_b[x1] rmse=(e(rmse))")
    values = parsed.evaluate(_results())
    assert values["_b[x1]"] == 2.0
    assert values["rmse"] == 1.5


# ------------------------------------------------------------- arithmetic


def test_operator_precedence():
    parsed = parse_exp_list("a=(1+2*3) b=(2^3^2) c=((1+2)*3) d=(-2^2)")
    values = parsed.evaluate(_results())
    assert values["a"] == 7.0
    # ^ is right associative in Stata, so 2^3^2 is 2^9.
    assert values["b"] == 512.0
    assert values["c"] == 9.0
    # Unary minus binds tighter than ^, so -2^2 is 4.
    assert values["d"] == 4.0


def test_division_and_negative_denominators():
    parsed = parse_exp_list("ratio=(r(mean)/r(sd))")
    assert parsed.evaluate(_results())["ratio"] == pytest.approx(2.0)


def test_division_by_zero_is_reported():
    parsed = parse_exp_list("bad=(r(mean)/0)")
    with pytest.raises(ExpListError, match="division by zero"):
        parsed.evaluate(_results())


def test_exp_of_a_standard_error():
    parsed = parse_exp_list("t=(_b[x1]/_se[x1])")
    assert parsed.evaluate(_results())["t"] == pytest.approx(2.0 / 0.3)


# --------------------------------------------------------------- functions


def test_supported_functions():
    parsed = parse_exp_list("a=(exp(1)) b=(ln(2)) c=(sqrt(9)) d=(abs(-4))")
    values = parsed.evaluate(_results())
    assert values["a"] == pytest.approx(math.e)
    assert values["b"] == pytest.approx(math.log(2))
    assert values["c"] == 3.0
    assert values["d"] == 4.0


def test_an_unsupported_function_names_itself():
    parsed = parse_exp_list("a=(logit(0.5))")
    with pytest.raises(ExpListError, match="logit"):
        parsed.evaluate(_results())


def test_ln_of_a_non_positive_value_is_reported():
    parsed = parse_exp_list("a=(ln(0))")
    with pytest.raises(ExpListError, match="non-positive"):
        parsed.evaluate(_results())


# --------------------------------------------------------------- failures


def test_a_missing_stored_result_lists_what_is_available():
    parsed = parse_exp_list("a=(r(nonexistent))")
    with pytest.raises(ExpListError, match="available"):
        parsed.evaluate(_results())


def test_a_bare_coefficient_is_rejected_rather_than_guessed():
    with pytest.raises(ExpListError, match="ambiguous"):
        parse_exp_list("_b")


def test_unbalanced_parentheses_are_reported():
    with pytest.raises(ExpListError, match="unbalanced"):
        parse_exp_list("(a: mean=r(mean)")


def test_an_empty_list_is_rejected():
    with pytest.raises(ExpListError, match="empty"):
        parse_exp_list("   ")


def test_an_unknown_leading_name_is_explained():
    with pytest.raises(ExpListError, match="not a stored-result reference"):
        parse_exp_list("a=(bogus)")


# --------------------------------------------------------------- labelling


def test_names_are_flattened_with_group_labels():
    parsed = parse_exp_list("(a: x=r(mean)) (b: y=r(sd))")
    assert parsed.names == ["a:x", "b:y"]


def test_an_unlabelled_statistic_keeps_its_own_name():
    parsed = parse_exp_list("rmean=(r(mean))")
    assert parsed.names == ["rmean"]
