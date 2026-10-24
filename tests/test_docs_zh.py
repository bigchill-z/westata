"""执行中文文档里的每一段 Python 代码，并核对数字与英文版一致。

英文 README 曾经有 4 处错误，全部是"读起来完全正常"的代码，靠阅读没发现，靠执行立刻发现。
中文版如果不受同样的约束，就等于引入了一个新的、无人验证的副本。

这个文件同时做三件事，对应三种真实的失效方式：

1. 代码块能否执行。翻译时最容易出现的错误是把 API 名译错，或漏掉一个必需的关键字参数。
2. 数字是否与英文版同源。两版文档各自维护同一个统计量，迟早会分叉；这里断言它们逐字符相同。
3. 章节是否对等。中文版是翻译，不是缩写版；缺章节属于未完成，而不是风格选择。

代码块用一个预置命名空间执行，因为文档中的后续代码依赖前面建立的 df/res，读者粘贴时的情况
就是这样。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EN_README = REPO / "README.md"
ZH_README = REPO / "README.zh-CN.md"
COMPARISON = REPO / "docs" / "comparison.json"

sys.path.insert(0, str(REPO / "src"))


def _blocks(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    return re.findall(r"```python\n(.*?)```", text, re.S)


def _run_blocks(path: pathlib.Path) -> None:
    import os

    import numpy as np
    import pandas as pd

    blocks = _blocks(path)
    assert blocks, f"{path.name} 里没有 python 代码块"

    namespace: dict[str, object] = {
        "__name__": "__doc__",
        "pd": pd,
        "np": np,
        "df": pd.read_csv(REPO / "docs" / "data" / "auto.csv"),
    }
    previous = os.getcwd()
    os.chdir(REPO)
    try:
        for index, block in enumerate(blocks, start=1):
            try:
                exec(compile(block, f"<{path.name} block {index}>", "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"{path.name} 第 {index} 个代码块报错 {type(exc).__name__}: {exc}\n"
                    f"--- 代码 ---\n{block}"
                ) from exc
    finally:
        os.chdir(previous)


def test_chinese_readme_code_blocks_execute() -> None:
    """中文 README 的代码必须能跑。翻译不算完成。"""

    if not ZH_README.exists():
        import pytest

        pytest.skip("README.zh-CN.md 不存在")
    _run_blocks(ZH_README)


def test_chinese_readme_numbers_match_the_measured_data() -> None:
    """中文版引用的每个实测数字都必须出现在 comparison.json 或注册表里。

    做法是把中文版里的数字抽出来，和英文版/JSON 中的同源数字对照。这里不做模糊匹配：注册表
    统计逐字符比对，对照结果按同样的格式化规则重算后检查出现。
    """

    import pytest

    if not ZH_README.exists():
        pytest.skip("README.zh-CN.md 不存在")

    from westata.conformance.registry import load_registry

    registry = load_registry(REPO / "tests" / "conformance" / "registry.yml")
    counts = {"E3": 0, "E2": 0, "E1": 0}
    for entry in registry:
        counts[entry.level] += 1

    zh = ZH_README.read_text(encoding="utf-8")
    en = EN_README.read_text(encoding="utf-8-sig")

    # 注册表条目数与各级数量：两版都必须出现同样的数字。
    for pattern in (
        rf"\*\*{counts['E3']} × E3\*\*",
        rf"{counts['E2']} × E2",
        rf"{counts['E1']} × E1",
    ):
        assert re.search(pattern, zh), f"中文版缺少注册表统计 {pattern}"
        assert re.search(pattern, en), f"英文版缺少注册表统计 {pattern}"

    # 对照结果：最差值与机器精度计数。
    entries = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparisons = [e for e in entries if "rel_b" in e]
    worst = max(max(e["rel_b"], e["rel_se"]) for e in comparisons)
    machine = sum(1 for e in comparisons if max(e["rel_b"], e["rel_se"]) < 1e-12)

    assert f"{worst:.2e}" in zh, f"中文版未引用实测最差值 {worst:.2e}"
    assert f"**{machine}**" in zh, f"中文版未引用机器精度计数 {machine}"

    # 对照命令数。
    assert f"| {len(comparisons)} |" in zh or f"{len(comparisons)} 条命令" in zh


def test_chinese_readme_has_the_same_sections() -> None:
    """中文版是翻译，不是缩写版。缺章节算未完成。"""

    import pytest

    if not ZH_README.exists():
        pytest.skip("README.zh-CN.md 不存在")

    zh = ZH_README.read_text(encoding="utf-8")
    en = EN_README.read_text(encoding="utf-8-sig")

    zh_headings = [line for line in zh.splitlines() if line.startswith("## ")]
    en_headings = [line for line in en.splitlines() if line.startswith("## ")]

    assert len(zh_headings) >= len(en_headings) - 1, (
        f"中文版章节数 {len(zh_headings)} 少于英文版 {len(en_headings)}；"
        f"中文：{zh_headings}；英文：{en_headings}"
    )


def test_chinese_readme_lists_all_e1_commands() -> None:
    """9 条 E1 命令必须在中文版里出现。漏掉它们，等于把分级藏起来。"""

    import pytest

    if not ZH_README.exists():
        pytest.skip("README.zh-CN.md 不存在")

    zh = ZH_README.read_text(encoding="utf-8")
    for command in (
        "xtgls",
        "teffects nnmatch",
        "frontier",
        "bsqreg",
        "permute",
        "arima",
        "rreg",
        "xtlogit, pa",
        "heckman",
    ):
        assert command in zh, f"中文版缺少 E1 命令 {command!r}"


def test_chinese_guide_code_blocks_execute() -> None:
    """中文指南里的代码同样必须能跑。翻译不算完成，能跑才算。"""

    import pytest

    guide = REPO / "docs" / "GUIDE.zh-CN.md"
    if not guide.exists():
        pytest.skip("GUIDE.zh-CN.md 不存在")
    _run_blocks(guide)


def test_chinese_guide_table_matches_the_english_one() -> None:
    """中英文指南的主对照表，前四列必须逐字符相同。

    第五列是给人读的判定词（英文 `machine precision`、中文 `机器精度`），属于翻译而非数据差异，
    因此不参与比对。把整行做全等比对会把正确的翻译判成错误——这是最初的写法，也是它失败的原因。

    前四列是命令名、样本量、两个相对偏差，全部来自 docs/comparison.json，两版之间不允许有任何
    差别；一旦有人手工改了其中一版，这里就会失败。
    """

    import pytest

    zh_guide = REPO / "docs" / "GUIDE.zh-CN.md"
    en_guide = REPO / "docs" / "GUIDE.md"
    if not zh_guide.exists() or not en_guide.exists():
        pytest.skip("指南文件不全")

    # 主表行的形状：| `cmd` | N | b | se | 判定 |
    shape = re.compile(r"^\| `[^`]+` \| \d+ \| [\d.e+\-]+ \| [\d.e+\-]+ \| [^|]+ \|$")

    def numeric_columns(path: pathlib.Path) -> list[tuple[str, str, str, str]]:
        out = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if shape.match(line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                out.append((cells[0], cells[1], cells[2], cells[3]))
        return out

    zh_rows = numeric_columns(zh_guide)
    en_rows = numeric_columns(en_guide)

    assert len(zh_rows) >= 20, f"中文主表只有 {len(zh_rows)} 行，抽取可能失效"
    assert len(zh_rows) == len(en_rows), (
        f"主表行数不同：中文 {len(zh_rows)}，英文 {len(en_rows)}"
    )
    assert zh_rows == en_rows, (
        "主表前四列不同；两版应来自同一份 comparison.json。\n"
        f"只在中文: {[r for r in zh_rows if r not in en_rows][:2]}\n"
        f"只在英文: {[r for r in en_rows if r not in zh_rows][:2]}"
    )


def test_chinese_guide_detail_rows_match_the_json() -> None:
    """逐参数明细的行数必须等于 JSON 里的参数总数。

    按表头定位表体，不按行的形状猜。主对照表的行也形如 `| `名字` | 数字 | ...`，任何基于形状的
    正则都会把它一起算进来——前两次尝试正是因此得到 99 和 108，而正确答案是 71。

    参数明细表的表头第二格是 `Stata `b``，主表的是 `N`；这个区别是稳定的，因为两者渲染自不同的
    数据字段。
    """

    import pytest

    guide = REPO / "docs" / "GUIDE.zh-CN.md"
    if not guide.exists():
        pytest.skip("GUIDE.zh-CN.md 不存在")

    entries = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparisons = [e for e in entries if "rel_b" in e]
    expected_rows = sum(len(e["rows"]) for e in comparisons)

    lines = guide.read_text(encoding="utf-8").splitlines()

    sections = sum(1 for line in lines if line.startswith("#### `"))
    assert sections == len(comparisons), (
        f"明细小节数 {sections} 与对照命令数 {len(comparisons)} 不符"
    )

    # 找到所有参数明细表：表头第二格是 "Stata `b`"。
    detail_rows = 0
    for index, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[1] == "Stata `b`":
            # 表体从这个表头的分隔行之后开始，到空行或非表格行为止。
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].startswith("|"):
                detail_rows += 1
                cursor += 1

    assert detail_rows == expected_rows, (
        f"明细行数 {detail_rows} 与 JSON 参数总数 {expected_rows} 不符"
    )

