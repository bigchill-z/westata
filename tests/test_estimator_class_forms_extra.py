"""R2.7 的后半程：其余 7 个估计命令的类形式，必须与函数形式逐个数字一致。

`newey`/`prais` 的类形式在 `test_estimator_class_forms.py`；这里覆盖后来补上的
`synth`/`synth2`/`rcm`/`rdrobust` 与三个 `teffects_*` 子命令。

核心断言始终是**两个入口给出同一组数字**——类是一个薄壳，函数保留了旧调用，
任何一边单独改动都会立刻暴露。另有一条测试钉住它们的构造入口必须是
`from_variables` 而非 `from_formula`：这些命令在 Stata 里用变量列表，写成
`from_formula` 就是 R3.2 要消除的「名字骗人」。
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def cross_section() -> pd.DataFrame:
    """300 行截面：一个处理指示变量和两个协变量。"""

    rng = np.random.default_rng(20261114)
    n = 300
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": (rng.normal(size=n) > 0).astype(int),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """12 个单位 × 20 期，单位 1 从第 12 期起有效应。"""

    rng = np.random.default_rng(20261114)
    rows = [(unit, period, float(rng.normal())) for unit in range(1, 13) for period in range(1, 21)]
    frame = pd.DataFrame(rows, columns=["unit", "year", "y"])
    frame["y"] = frame["y"] + np.where(
        (frame["unit"] == 1) & (frame["year"] >= 12), 3.0, 0.0
    )
    return frame


@pytest.fixture(scope="module")
def regression_discontinuity() -> pd.DataFrame:
    """500 行，running variable 在 0 处有一个跳跃。"""

    rng = np.random.default_rng(20261114)
    n = 500
    running = rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 0.5 * (running >= 0) + 0.3 * running + rng.normal(size=n) * 0.2,
            "x": running,
        }
    )


# --------------------------------------------------------------- teffects 三个
def test_teffects_ra_class_matches_the_function(cross_section: pd.DataFrame) -> None:
    frame = cross_section
    by_function = ws.teffects_ra(
        frame["y"].to_numpy(), frame["d"].to_numpy(), frame[["x1", "x2"]].to_numpy()
    )
    by_class = ws.TeffectsRa.from_variables(
        "y", "d", frame, covariates=["x1", "x2"]
    ).fit()

    assert np.isclose(float(by_function.ate), float(by_class.ate))


def test_teffects_psmatch_class_matches_the_function(cross_section: pd.DataFrame) -> None:
    frame = cross_section
    by_function = ws.teffects_psmatch(
        frame["y"].to_numpy(), frame["d"].to_numpy(), frame[["x1", "x2"]].to_numpy()
    )
    by_class = ws.TeffectsPsmatch.from_variables(
        "y", "d", frame, covariates=["x1", "x2"]
    ).fit()

    assert np.isclose(float(by_function.ate), float(by_class.ate))


def test_teffects_nnmatch_class_matches_the_function(cross_section: pd.DataFrame) -> None:
    frame = cross_section
    by_function = ws.teffects_nnmatch(
        frame["y"].to_numpy(), frame["d"].to_numpy(), frame[["x1", "x2"]].to_numpy()
    )
    by_class = ws.TeffectsNnmatch.from_variables(
        "y", "d", frame, covariates=["x1", "x2"]
    ).fit()

    assert np.isclose(float(by_function.ate), float(by_class.ate))


# ------------------------------------------------------------------ rdrobust
def test_rdrobust_class_matches_the_function(regression_discontinuity: pd.DataFrame) -> None:
    frame = regression_discontinuity
    by_function = ws.rdrobust(frame["y"].to_numpy(), frame["x"].to_numpy(), bandwidth=0.5)
    by_class = ws.Rdrobust.from_variables("y", "x", frame, bandwidth=0.5).fit()

    assert np.isclose(float(by_function.tau), float(by_class.tau))
    assert np.isclose(float(by_function.se), float(by_class.se))


# --------------------------------------------------------------------- synth
def test_synth_class_matches_the_function(panel: pd.DataFrame) -> None:
    args = dict(unit="unit", time="year", depvar="y", predictors=["y"], trunit=1, trperiod=12)
    by_function = ws.synth(panel, **args)
    by_class = ws.Synth.from_variables(panel, **args).fit()

    assert np.isclose(float(by_function.RMSPE), float(by_class.RMSPE))


def test_synth2_class_matches_the_function(panel: pd.DataFrame) -> None:
    args = dict(unit="unit", time="year", depvar="y", predictors=["y"], trunit=1, trperiod=12)
    by_function = ws.synth2(panel, **args)
    by_class = ws.Synth2.from_variables(panel, **args).fit()

    # `Synth2Result` 的数值字段都在 dataclass 上；逐个比对，避免只看一个数字。
    for name in ("N", "T", "T0", "T1", "K", "J", "rmse", "r2"):
        assert np.isclose(
            float(getattr(by_function, name)), float(getattr(by_class, name))
        ), f"{name} differs between the function and the class"


# ----------------------------------------------------------------------- rcm
def test_rcm_class_matches_the_function() -> None:
    """`rcm` 需要一个可预测的单位-时间面板。"""

    rng = np.random.default_rng(20261114)
    rows = []
    for unit in range(1, 11):
        for period in range(1, 16):
            treated = unit == 1 and period >= 10
            rows.append(
                (
                    unit,
                    period,
                    float(rng.normal() + (2.0 if treated else 0.0)),
                    float(rng.normal()),
                )
            )
    frame = pd.DataFrame(rows, columns=["unit", "year", "y", "x"])
    args = dict(
        predictors=["x"],
        unit="unit",
        time="year",
        treated_unit=1,
        treatment_period=10,
    )
    by_function = ws.rcm(
        frame["y"].to_numpy(), frame[["x"]].to_numpy(),
        frame["unit"].to_numpy(), frame["year"].to_numpy(),
        **{k: v for k, v in args.items() if k not in ("predictors", "unit", "time")},
    )
    by_class = ws.Rcm.from_variables("y", frame, **args).fit()

    assert np.isclose(float(by_function.att), float(by_class.att))


# --------------------------------------------- 构造入口必须是 from_variables
@pytest.mark.parametrize(
    "cls",
    [ws.TeffectsRa, ws.TeffectsPsmatch, ws.TeffectsNnmatch, ws.Rdrobust,
     ws.Synth, ws.Synth2, ws.Rcm],
)
def test_non_formula_command_uses_from_variables(cls: type) -> None:
    """这些命令在 Stata 里用变量列表，因此入口是 `from_variables`（R3.2）。

    写成 `from_formula` 会让名字撒谎：调用者以为能传 `"y ~ x"`，实际要传列名。
    """

    assert hasattr(cls, "from_variables"), f"{cls.__name__} must offer from_variables"
    assert not hasattr(cls, "from_formula"), (
        f"{cls.__name__} takes variable names in Stata, so it must NOT expose "
        f"from_formula -- that name would lie about what it accepts (R3.2)"
    )


# ------------------------------------------------------ fit() 不接受 cov_type
@pytest.mark.parametrize(
    "cls",
    [ws.TeffectsRa, ws.TeffectsPsmatch, ws.TeffectsNnmatch, ws.Rdrobust,
     ws.Synth, ws.Synth2, ws.Rcm],
)
def test_fit_offers_no_cov_type(cls: type) -> None:
    """这些命令没有 `vce()` 选项，所以 `fit()` 不该承诺这个选择（R1.5）。"""

    assert "cov_type" not in inspect.signature(cls.fit).parameters, (
        f"{cls.__name__}.fit() must not accept cov_type"
    )
