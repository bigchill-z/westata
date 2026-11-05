"""Pin ``_stata_number``'s fixed-vs-exponential rule.

Stata prefers FIXED notation and falls back to an exponent only when the fixed form cannot
fit the field. C's ``%g`` (and Python's) switches to an exponent once the exponent drops
below -4, which is why a standard error of .0000249 came out as ``2.49e-05`` where Stata
prints ``.0000249``.

That single cell in ``arch dinf, arch(1)`` was the visible symptom:

    before:  _cons |   .0001802   2.49e-05     7.25   ...
    Stata :  _cons |   .0001802   .0000249     7.25   ...

The fix walks DECIMAL PLACES and keeps the longest fixed form that fits the field, so
2.486e-05 becomes .0000249 (7 places, 8 characters) rather than being formatted at full
precision (.12f -> .000024861206, 13 characters, rejected as too wide).
"""

from __future__ import annotations

import pytest

from westata.results.summary import _stata_number


def test_small_values_use_fixed_notation() -> None:
    """The arch case: .0000249, not 2.49e-05."""

    assert _stata_number(2.48612055e-05, 9).strip() == ".0000249"
    assert _stata_number(-2.48612055e-05, 9).strip() == "-.0000249"


def test_the_documented_percent9_0g_cases_still_hold() -> None:
    """The original rule must not regress -- these were verified against Stata's display."""

    cases = [
        (0.1236468764, ".1236469"),
        (0.0133632566, ".0133633"),
        (0.0009470056, ".000947"),
        (-0.02220731, "-.0222073"),
        (1.0231784, "1.023178"),
    ]
    for value, expected in cases:
        assert _stata_number(value, 9).strip() == expected, value


def test_a_value_too_small_for_fixed_form_keeps_its_exponent() -> None:
    """1.23e-12 cannot be written in fixed form within nine characters, so it stays.

    An earlier attempt collapsed this to a bare "0" -- losing the value entirely -- which is
    worse than any formatting difference.
    """

    rendered = _stata_number(1.23e-12, 9).strip()
    assert "e" in rendered
    assert "1.23" in rendered


def test_the_arch_standard_error_is_fixed_notation() -> None:
    """The exact value from the reference fit, pinned end to end."""

    bse = 2.4861205555986357e-05
    assert _stata_number(bse, 9).strip() == ".0000249"
