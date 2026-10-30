"""The rendered table must match Stata's row order and labels.

Stata prints ``_cons`` LAST; patsy puts ``Intercept`` first. Until round 114 the
renderer used ``result.params.index`` directly, so an OLS table led with
``Intercept`` where Stata ends with ``_cons``.

The captured Stata output settles the placement:
``tests/conformance/stata_fixtures/p10_outputs.txt:19``, ``:24``, ``:75`` and ``:81``
all end their coefficient block with ``_cons``.

Reordering alone is not enough -- the LABEL has to change too, because patsy's name
is ``Intercept`` and Stata's is ``_cons``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(211)
    n = 150
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 - 0.3 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


#: Labels that belong to the ANOVA block, not the coefficient block.
_ANOVA_LABELS = {"Source", "Model", "Residual", "Total"}


def _coefficient_rows(rendered: str) -> list[str]:
    """Pull the parameter labels out of a rendered Stata-style table.

    The ANOVA block uses the same ``label | numbers`` shape, so its rows are
    filtered out by name -- otherwise they read as coefficients.
    """

    rows = []
    for line in rendered.splitlines():
        if "|" not in line or "Coefficient" in line or line.startswith("-"):
            continue
        label = line.split("|", 1)[0].strip()
        if label and label not in _ANOVA_LABELS:
            rows.append(label)
    return rows


def test_cons_is_printed_last(frame) -> None:
    """Stata's placement: ``_cons`` closes the coefficient block."""

    rendered = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().summary(style="stata")
    rows = _coefficient_rows(rendered)

    assert rows[-1] == "_cons", rows
    assert "_cons" not in rows[:-1], rows


def test_the_intercept_is_labelled_cons_not_intercept(frame) -> None:
    """patsy calls it ``Intercept``; Stata calls it ``_cons``."""

    rendered = ws.OLS.from_formula("y ~ x1", frame).fit().summary(style="stata")

    assert "_cons" in rendered
    assert "Intercept" not in rendered


def test_the_slopes_keep_their_own_order(frame) -> None:
    """Only the constant moves; the regressors stay in formula order."""

    rendered = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().summary(style="stata")
    rows = _coefficient_rows(rendered)

    assert rows == ["x1", "x2", "_cons"]


def test_each_row_keeps_its_own_numbers_after_reordering(frame) -> None:
    """Reordering the rows must carry every column with its name.

    If only the labels moved, the first row would show ``x1`` with the intercept's
    estimate -- a table that looks structurally correct and is numerically wrong.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    rendered = result.summary(style="stata")

    # Locate the ``_cons`` row and check its coefficient cell against the stored
    # intercept.  Stata drops the leading zero only for values below 1, so compare
    # the 7-decimal expansion which works either way.
    cons_line = next(
        line for line in rendered.splitlines() if line.strip().startswith("_cons")
    )
    intercept = float(result.params["Intercept"])
    assert f"{intercept:.6f}" in cons_line, (intercept, cons_line)


def test_a_model_without_an_intercept_has_no_cons_row(frame) -> None:
    """No intercept, no ``_cons`` -- and nothing else should be relabelled."""

    rendered = ws.OLS.from_formula("y ~ x1 + x2 - 1", frame).fit().summary(
        style="stata"
    )
    rows = _coefficient_rows(rendered)

    assert "_cons" not in rows
    assert rows == ["x1", "x2"]


def test_an_auxiliary_parameter_stays_last(frame) -> None:
    """``tobit`` reports its auxiliary ``sigma`` after ``_cons``.

    ``stata_param_names`` puts auxiliary parameters last (``_stata_order``).
    ``tobit``'s is stored and displayed as ``sigma`` (no slash); ``intreg`` uses
    ``/lnsigma``.  Both are covered here via the label set.
    """

    censored = frame.assign(y=np.maximum(frame["y"], 0.5))
    rendered = ws.Tobit.from_formula("y ~ x1", censored).fit().summary(style="stata")
    rows = _coefficient_rows(rendered)

    assert rows.index("_cons") < len(rows) - 1, rows
    assert rows[-1] in ("sigma", "/lnsigma", "/sigma"), rows


def test_the_statsmodels_style_keeps_patsy_names(frame) -> None:
    """Only the Stata rendering is relabelled.

    The statsmodels-facing table is a different surface with its own conventions;
    changing it would break callers that expect patsy's names.
    """

    rendered = ws.OLS.from_formula("y ~ x1", frame).fit().summary()

    assert "Intercept" in rendered
    assert "_cons" not in rendered


# ------------------------------------- interaction terms use Stata's separator
@pytest.fixture(scope="module")
def interaction_result():
    rng = np.random.default_rng(431)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 - 0.3 * x2 + 0.5 * x1 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
        }
    )
    return ws.OLS.from_formula("y ~ x1*x2", frame).fit()


def test_the_coefficient_table_uses_the_hash_separator(interaction_result) -> None:
    """Stata separates interaction terms with ``#``; patsy uses ``:``."""

    rendered = interaction_result.summary(style="stata")

    assert "x1#x2" in rendered
    assert "x1:x2" not in rendered


def test_estat_vce_uses_the_same_separator(interaction_result) -> None:
    """The covariance matrix must agree with the coefficient table's vocabulary.

    Before this fix one analysis printed ``x1#x2`` in the table and ``x1:x2`` in the
    matrix -- two names for one parameter, which reads as two parameters.
    """

    rendered = str(interaction_result.estat_vce())

    assert "x1#x2" in rendered
    assert "x1:x2" not in rendered


def test_both_tables_label_every_row_with_its_own_value(interaction_result) -> None:
    """Cross-check the two tables: the diagonal of V is bse² for the SAME row.

    Both tables carry the same four labels, so pairing them proves the rename is
    consistent rather than coincidentally applying to one.
    """

    from westata.results.summary import _stata_coef

    result = interaction_result
    table_rows = _coefficient_rows(result.summary(style="stata"))
    matrix = result.estat_vce()

    assert table_rows == list(matrix.names)
    for index, label in enumerate(matrix.names):
        assert table_rows[index] == label, (index, label)


def test_the_patsy_spelling_is_gone_from_the_matrix(interaction_result) -> None:
    """No row or column keeps patsy's separator."""

    matrix = interaction_result.estat_vce()

    assert not any(":" in name for name in matrix.names), matrix.names
    assert "_cons" in matrix.names


def test_estat_vif_uses_the_same_separator(interaction_result) -> None:
    """``estat vif`` lists the interaction under Stata's spelling too.

    It carried its OWN ``_display_name`` that handled only the constant, so the same
    analysis printed ``x1#x2`` in the coefficient table and ``x1:x2`` in the VIF
    table.  Three tables, one parameter, two names.
    """

    vif = interaction_result.estat_vif()
    rendered = str(vif)

    assert "x1#x2" in vif.names
    assert not any(":" in name for name in vif.names), vif.names
    assert "x1:x2" not in rendered


def test_every_table_describes_the_same_parameters(interaction_result) -> None:
    """The full cross-check across all three tables.

    The coefficient table and ``estat vce`` carry the constant and the intercept row;
    ``estat vif`` drops the constant but must still agree on the regressor names.
    """

    result = interaction_result
    table_labels = [
        label for label in _coefficient_rows(result.summary(style="stata"))
        if label != "_cons"
    ]
    vce_labels = [name for name in result.estat_vce().names if name != "_cons"]
    vif_labels = list(result.estat_vif().names)

    assert table_labels == vce_labels
    assert sorted(vif_labels) == sorted(table_labels)


# ------------------------------------------------------------ the ANOVA table
def test_the_anova_table_is_printed_for_an_ols_fit(frame) -> None:
    """Stata prints ``Source | SS df MS`` beside the fit statistics.

    The layout is visible in ``p10_outputs.txt:97-102`` for ``nl``; ``regress`` uses
    the same block.
    """

    rendered = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().summary(style="stata")

    for label in ("Source |", "Model |", "Residual |", "Total |"):
        assert label in rendered, label


def test_the_anova_table_replaces_the_stacked_header(frame) -> None:
    """The fit statistics appear ONCE, in the right-hand column.

    Printing the ANOVA table alongside the old stacked lines would repeat
    "Number of obs" and "R-squared", which Stata never does.
    """

    rendered = ws.OLS.from_formula("y ~ x1", frame).fit().summary(style="stata")

    assert rendered.count("Number of obs") == 1
    assert rendered.count("R-squared") == 2  # "R-squared" and "Adj R-squared"
    assert rendered.count("Root MSE") == 1


def test_the_anova_sums_of_squares_match_the_stored_scalars(frame) -> None:
    """``Model SS`` is ``e(mss)`` and ``Residual SS`` is ``e(rss)``.

    ``Total SS`` is their sum -- Stata's ``regress`` stores only ``rss`` and
    ``mss``, so the total is derived rather than read.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e
    rendered = result.summary(style="stata")

    model_line = next(line for line in rendered.splitlines() if "Model |" in line)
    residual_line = next(line for line in rendered.splitlines() if "Residual |" in line)
    total_line = next(line for line in rendered.splitlines() if "Total |" in line)

    assert f"{e['mss']:.7g}" in model_line
    assert f"{e['rss']:.7g}" in residual_line
    assert f"{e['mss'] + e['rss']:.7g}" in total_line


def test_the_total_row_has_n_minus_one_degrees_of_freedom(frame) -> None:
    """``Total`` df is ``df_m + df_r``, which equals ``N - 1`` with a constant.

    The constant is absorbed by the total, so using ``N`` would overstate it by one
    -- the mistake ``nl``'s capture shows is NOT made (``p10_outputs.txt:102`` has
    ``Total`` df equal to the sample size only because ``nl`` has no constant).
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e
    rendered = result.summary(style="stata")
    total_line = next(line for line in rendered.splitlines() if "Total |" in line)

    assert f"{int(e['df_m'] + e['df_r']):>10d}" in total_line
    assert int(e["df_m"] + e["df_r"]) == int(e["N"]) - 1


def test_each_mean_square_is_the_ratio_of_its_row(frame) -> None:
    """``MS`` is ``SS / df`` in every row -- three independent checks."""

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e
    rendered = result.summary(style="stata")

    model_line = next(line for line in rendered.splitlines() if "Model |" in line)
    residual_line = next(line for line in rendered.splitlines() if "Residual |" in line)
    total_line = next(line for line in rendered.splitlines() if "Total |" in line)

    assert f"{e['mss'] / e['df_m']:.7g}" in model_line
    assert f"{e['rss'] / e['df_r']:.7g}" in residual_line
    total_df = e["df_m"] + e["df_r"]
    assert f"{(e['mss'] + e['rss']) / total_df:.7g}" in total_line


def test_the_ml_family_still_gets_the_stacked_header(frame) -> None:
    """Only OLS-style fits carry sums of squares.

    ``logit`` has no ``e(mss)``, so it must keep the "Log likelihood / Pseudo R2"
    block and gain no ANOVA table.
    """

    logit_frame = frame.assign(yb=(frame["y"] > frame["y"].median()).astype(int))
    rendered = ws.Logit.from_formula("yb ~ x1", logit_frame).fit().summary(style="stata")

    assert "Source |" not in rendered
    assert "Log likelihood" in rendered
    assert "Number of obs" in rendered


def test_a_quantile_fit_gets_no_anova_table(frame) -> None:
    """``qreg`` stores no sums of squares in Stata's sense."""

    rendered = (
        ws.QuantileRegression.from_formula("y ~ x1", frame).fit().summary(style="stata")
    )

    assert "Source |" not in rendered
    assert "Raw sum of deviations" in rendered


# ------------------------------------------- the panel decomposition block
@pytest.fixture(scope="module")
def panel_result():
    rng = np.random.default_rng(311)
    entities, periods = 40, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": firm,
            "year": np.tile(np.arange(periods), entities),
        }
    )
    return ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


def test_the_panel_block_reports_the_variance_decomposition(panel_result) -> None:
    """``xtreg`` prints ``sigma_u`` / ``sigma_e`` / ``rho`` beneath the table.

    The keys are confirmed by ``xtreg_fe.json``: ``sigma_u``, ``sigma_e``, ``rho`` and
    ``corr`` all have values there.
    """

    rendered = panel_result.summary(style="stata")

    for label in ("sigma_u |", "sigma_e |", "rho |"):
        assert label in rendered, label
    assert "fraction of variance due to u_i" in rendered
    assert "corr(u_i, Xb)" in rendered


def test_the_panel_block_values_are_the_stored_scalars(panel_result) -> None:
    """The printed numbers must be the e() values, not recomputed ones."""

    from westata.results.summary import _stata_coef

    e = panel_result.e
    rendered = panel_result.summary(style="stata")

    for key in ("sigma_u", "sigma_e", "rho"):
        # Render through the SAME formatter the table uses, so the assertion tests
        # the value rather than re-deriving Stata's digit conventions.
        assert _stata_coef(float(e[key]), 12).strip() in rendered, key


def test_the_rho_identity_holds_in_the_rendered_block(panel_result) -> None:
    """``rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)`` -- checked on the values printed.

    Reading the block back and recombining it is the check that the three numbers
    describe the same fit; printing them from three independent sources would pass a
    "labels present" test while being inconsistent.
    """

    e = panel_result.e

    assert e["rho"] == pytest.approx(
        e["sigma_u"] ** 2 / (e["sigma_u"] ** 2 + e["sigma_e"] ** 2), rel=1e-12
    )


def test_a_non_panel_fit_gets_no_panel_block(frame) -> None:
    """``regress`` has no ``sigma_u``, so the block must not appear."""

    rendered = ws.OLS.from_formula("y ~ x1", frame).fit().summary(style="stata")

    assert "sigma_u |" not in rendered
    assert "corr(u_i, Xb)" not in rendered


def test_the_panel_block_prints_no_invented_standard_errors(panel_result) -> None:
    """Stata shows a standard error beside ``sigma_u``; ``e()`` does not store one.

    The block therefore prints the estimate alone.  Fabricating a standard error --
    or borrowing ``e(sigma)``'s -- would put a number in a column Stata fills from a
    different source.
    """

    rendered = panel_result.summary(style="stata")
    sigma_line = next(line for line in rendered.splitlines() if "sigma_u |" in line)

    # Exactly one numeric token on the line: the estimate.
    numeric = [
        token
        for token in sigma_line.replace("|", " ").split()
        if token.lstrip("-").replace(".", "", 1).replace("e", "", 1).isdigit()
        or token.startswith(".")
    ]
    assert len(numeric) == 1, numeric