"""`newey`/`prais` 的类形式（R2.7）与它们保留的函数形式必须完全等价。

这两个命令曾是全包 19 个估计命令里仅有的两个"只能当函数调"的：同一件事在别处是
`ws.OLS.from_formula(...).fit()`，在这里是 `ws.newey(...)`。加入类形式时最大的风险
不是新代码写错，而是**两份实现悄悄分叉**——包装函数和新类各算一遍，某天其中一份
被改动，数字就开始不一致而没人发现。

所以这组测试的核心是「两种入口给出逐位相同的数字」，而不是「两种入口都能跑」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """120 行、三个回归元的平稳数据。"""

    rng = np.random.default_rng(20261114)
    n = 120
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


# ---------------------------------------------------------------- 实现只有一份
def test_newey_class_and_function_agree(frame: pd.DataFrame) -> None:
    """`ws.newey(...)` 与 `ws.Newey.from_formula(...).fit()` 必须逐位相同。"""

    by_function = ws.newey("y ~ x1 + x2", frame, lag=2)
    by_class = ws.Newey.from_formula("y ~ x1 + x2", frame, lag=2).fit()

    assert np.array_equal(by_function.params, by_class.params)
    assert np.array_equal(by_function.bse, by_class.bse)
    assert by_function.cov_type == by_class.cov_type == "HAC"


def test_prais_class_and_function_agree(frame: pd.DataFrame) -> None:
    """`ws.prais(...)` 与 `ws.Prais.from_formula(...).fit()` 必须逐位相同。"""

    by_function = ws.prais(frame, "y ~ x1 + x2")
    by_class = ws.Prais.from_formula("y ~ x1 + x2", frame).fit()

    assert np.array_equal(by_function.params, by_class.params)
    assert np.array_equal(by_function.bse, by_class.bse)
    assert by_function.rho == by_class.rho


@pytest.mark.parametrize("corc", [False, True])
def test_prais_corc_switches_the_same_way_in_both_forms(
    frame: pd.DataFrame, corc: bool
) -> None:
    """`corc` 是模型选项，两种入口必须一起切到同一个估计量。"""

    by_function = ws.prais(frame, "y ~ x1", corc=corc)
    by_class = ws.Prais.from_formula("y ~ x1", frame, corc=corc).fit()

    assert np.array_equal(by_function.params, by_class.params)
    assert by_function.rho == by_class.rho


# ------------------------------------------------------- 位置参数的两种顺序
def test_newey_positional_construction_keeps_the_function_order(
    frame: pd.DataFrame,
) -> None:
    """`Newey(data, formula)` 沿用函数的位置顺序，因为 `newey(formula, data)` 是标准。

    两个位置参数都是"看起来对"的类型（DataFrame 与 str 不同型，所以交换会报错
    而不是静默算错），但仍要钉住，否则重构时容易随手调换。
    """

    by_position = ws.Newey(frame, "y ~ x1", lag=1).fit()
    by_standard = ws.Newey.from_formula("y ~ x1", frame, lag=1).fit()

    assert np.array_equal(by_position.params, by_standard.params)


def test_prais_positional_construction_keeps_the_reversed_order(
    frame: pd.DataFrame,
) -> None:
    """`Prais(data, formula)` 是**反的**——这是刻意的，52 处调用都按位置传。

    如果哪天有人为了"统一"把它改成 `(formula, data)`，这条测试会失败，提醒他
    52 处 `ws.prais(frame, FORMULA)` 会静默错位（DataFrame 被当成公式解析）。
    """

    by_position = ws.Prais(frame, "y ~ x1").fit()
    by_standard = ws.Prais.from_formula("y ~ x1", frame).fit()

    assert np.array_equal(by_position.params, by_standard.params)


def test_from_formula_takes_the_formula_first(frame: pd.DataFrame) -> None:
    """R3.2：`from_formula` 的前两个位置参数固定是 `(formula, data)`。"""

    import inspect

    for cls in (ws.Newey, ws.Prais):
        parameters = list(inspect.signature(cls.from_formula).parameters)
        assert parameters[:2] == ["formula", "data"], (
            f"{cls.__name__}.from_formula must take (formula, data), "
            f"got {parameters[:2]}"
        )


# ------------------------------------------------- `fit()` 不接受 cov_type
@pytest.mark.parametrize("cls", [ws.Newey, ws.Prais])
def test_fit_does_not_offer_a_cov_type(cls: type) -> None:
    """Stata 的这两个命令没有 `vce()` 选项，所以 `fit()` 不该承诺这个选择。

    加了 `cov_type` 参数会让调用者以为可以换方差估计；实际换不了，或者换掉之后
    得到的就不再是 `newey`/`prais`。这是"接口承诺了不存在的东西"（R1.5）。
    """

    import inspect

    assert "cov_type" not in inspect.signature(cls.fit).parameters, (
        f"{cls.__name__}.fit() must not accept cov_type: Stata's command has no vce() "
        f"option, so the parameter would promise a choice that cannot be honoured"
    )


# ------------------------------------------------------- lag 属于模型不属于 fit
def test_newey_lag_belongs_to_construction() -> None:
    """`lag` 决定估计量本身（HAC 求和的阶数），因此是构造参数。"""

    import inspect

    assert "lag" in inspect.signature(ws.Newey.__init__).parameters
    assert "lag" in inspect.signature(ws.Newey.from_formula).parameters


def test_newey_lag_can_be_overridden_for_one_fit(frame: pd.DataFrame) -> None:
    """单次拟合可以覆盖 lag，方便比较不同截断阶。"""

    model = ws.Newey.from_formula("y ~ x1", frame, lag=1)
    shorter = model.fit()
    longer = model.fit(lag=3)

    assert np.array_equal(shorter.params, longer.params)  # 系数不随 lag 变
    assert not np.array_equal(shorter.bse, longer.bse)  # 方差会变
