"""``regress``'s ``e()`` keys, checked item by item against ``regress.sthlp``.

The help file's stored-results block (``regress.sthlp:406-425``) lists exactly these
scalars:

    N  mss  df_m  rss  df_r  r2  r2_a  F  rmse  ll  ll_0  sum_w  N_clust  rank

plus the macros (``cmd``, ``cmdline``, ``depvar``, ``properties``, ``predict``,
``marginsok``, ``model``, ``estat_cmd``, ``title``, ``vce``, ``vcetype``, ``wexp``,
``wtype``, ``clustvar``, ``beta``, ``sample``, ``kcluster``, ``asbalanced``,
``asobserved``, ``adj_df``).

Three keys this package exposes are NOT in that block:

* ``sigma`` -- ``e(rmse)`` already carries the root mean squared error;
* ``aic`` / ``bic`` -- ``regress`` stores no information criteria at all (their values
  here are on the Gaussian-deviance scale and omit ``N(1 + log 2pi)``).

They are kept for caller compatibility but must not be mistaken for Stata returns; the
Stata-conformant route is ``estat ic``. ``tss`` is deliberately ABSENT -- ``regress``
stores ``rss`` and ``mss`` and no total sum of squares.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

HELP = Path(r"D:\Program Files\StataNow19\ado\base\r\regress.sthlp")

#: The scalars ``regress.sthlp:409-425`` lists, transcribed.
STATA_SCALARS = frozenset(
    {
        "N",
        "mss",
        "df_m",
        "rss",
        "df_r",
        "r2",
        "r2_a",
        "F",
        "rmse",
        "ll",
        "ll_0",
        "sum_w",
        "N_clust",
        "rank",
    }
)

#: Keys this package adds that Stata's ``regress`` does not store.
EXTRAS = frozenset({"sigma", "aic", "bic"})


def _frame(n: int = 250, seed: int = 5000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.OLS.from_formula("y ~ x1 + x2", _frame()).fit()


# -------------------------------------------------------------- the documented set
@pytest.mark.parametrize("key", sorted(STATA_SCALARS - {"N_clust"}))
def test_every_documented_scalar_is_present(fitted, key: str) -> None:
    """All but ``N_clust``, which only appears under ``vce(cluster)``."""

    assert key in fitted.e, key
    assert fitted.e[key] is not None, key
    assert np.isfinite(float(fitted.e[key])), key


def test_the_documented_scalars_really_are_in_the_help_file() -> None:
    """Read the authority rather than trusting the transcription above.

    A future Stata upgrade that adds or renames a scalar should make this fail, which is
    the point -- the matrix is checked against the file, not against memory.
    """

    text = HELP.read_text(encoding="utf-8", errors="ignore")
    for key in STATA_SCALARS - {"N_clust"}:
        assert f"{{cmd:e({key})}}" in text, key


# -------------------------------------------------------------- what we add
@pytest.mark.parametrize("key", sorted(EXTRAS))
def test_the_extra_keys_are_documented_as_extras(fitted, key: str) -> None:
    """They exist (callers rely on them) but Stata does not store them."""

    assert key in fitted.e

    text = HELP.read_text(encoding="utf-8", errors="ignore")
    assert f"{{cmd:e({key})}}" not in text, key


def test_sigma_duplicates_rmse(fitted) -> None:
    """``sigma`` is this package's alias for the root mean squared error.

    Stata names it ``e(rmse)``; the alias exists because other Stata commands (``xtreg``)
    do store ``e(sigma)``, and callers move between the two families.
    """

    assert float(fitted.e["sigma"]) == pytest.approx(float(fitted.e["rmse"]), rel=1e-12)


def test_the_extra_aic_gap_is_the_omitted_constant(fitted) -> None:
    """``estat ic``'s AIC minus ``e(aic)`` is exactly ``N (1 + log 2pi)``."""

    e = fitted.e
    n = float(e["N"])

    assert fitted.estat_ic().aic - float(e["aic"]) == pytest.approx(
        n * (1.0 + np.log(2.0 * np.pi)), rel=1e-10
    )


# -------------------------------------------------------------- what we omit
def test_tss_is_absent(fitted) -> None:
    """``regress`` stores ``rss`` and ``mss``, never a total sum of squares.

    Round 160's firststage work and several earlier rounds relied on
    ``tss = mss + rss`` being reconstructed rather than stored -- that reconstruction is
    correct precisely because Stata does not provide the key.
    """

    assert "tss" not in fitted.e
    assert float(fitted.e["mss"]) + float(fitted.e["rss"]) > 0.0


def test_the_macros_are_present(fitted) -> None:
    """The macro block exists too, at least the ones a caller reads first."""

    for key in ("cmd", "depvar", "properties"):
        assert key in fitted.e, key


def test_properties_names_the_estimates_it_carries(fitted) -> None:
    """``e(properties)`` lists which estimate matrices exist -- ``b V`` here."""

    assert "b" in str(fitted.e["properties"])
    assert "V" in str(fitted.e["properties"])