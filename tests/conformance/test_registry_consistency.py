"""证据等级必须与条目自己的叙述一致（R7.1）。

`tests/conformance/test_registry.py` 校验的是**结构**：字段存在、等级合法、容差为正。
它不校验**内容**，于是下面这类矛盾能长期存活：

* `teffects nnmatch` 的 `level` 是 `E1`，`notes` 结尾却写着「Left at E2」；
* `permute` 的 `level` 是 `E1`，`notes` 却写着「still E2」。

两处都是把等级从 E2 降到 E1 时忘了改叙述。读者看到的是互相打架的两句话，
而任何自动检查都不会报错——这正是「文档即契约」最容易失效的地方。

这里加的不是"更严格的 schema"，而是三条内容一致性断言：
1. `notes` 里自述的等级必须等于 `level` 字段；
2. `E1` 条目必须在 `notes` 里说明它属于 (a) 还是没有 Stata 数值，或 (b) 有数值但
   差异未被解释——因为 `E1` 同时覆盖这两种情况，不写清楚就无法判断；
3. `E3` 条目必须给出可比对的容差依据（有 `tolerance`），且不允许在 `notes` 里
   承认"未对拍"——那是 `E1` 的语义。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REGISTRY = Path(__file__).resolve().parent / "registry.yml"

#: `notes` 里形如 "still E2" / "Left at E1" 的自述等级。
SELF_DECLARED = re.compile(r"(?:still|Left at|left at|remains)\s+(E[123])")


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def test_notes_never_contradict_the_level_field(entries: list[dict]) -> None:
    """`notes` 里自述的等级必须与 `level` 一致。

    这是实际发生过的缺陷：降级时只改了 `level`，`notes` 还留着旧话。
    读者无从判断哪个才对，而结构校验完全看不见。
    """

    contradictions = []
    for entry in entries:
        notes = str(entry.get("notes") or "")
        for match in SELF_DECLARED.finditer(notes):
            declared = match.group(1)
            if declared != entry["level"]:
                contradictions.append(
                    f"{entry['command']}: level={entry['level']} but notes say {declared}"
                )

    assert not contradictions, (
        "notes contradict the level field:\n" + "\n".join(contradictions)
    )


def test_every_e1_entry_explains_which_kind_of_e1_it_is(entries: list[dict]) -> None:
    """`E1` 有两种原因，条目必须写明是哪一种。

    (a) 根本没有 Stata 数值——公式抄自 `.ado`/手册，用独立实现做单测；
    (b) 有 Stata 数值、差异也记录了，但差异未被归因到可以称为"已验证"的程度。

    不写清楚，(a) 和 (b) 在表里长得一模一样，而它们的可信度不同。
    """

    unexplained = []
    for entry in entries:
        if entry["level"] != "E1":
            continue
        notes = str(entry.get("notes") or "")
        # 说明 (a)：明确说没对拍过 / 没有 fixture
        says_absent = bool(
            re.search(
                r"NOT been contrasted|no fixture|never compared|not verified against Stata"
                r"|no Stata number|transcribed from",
                notes,
                re.IGNORECASE,
            )
        )
        # 说明 (b)：指出差异存在且未被解释
        says_gap = bool(
            re.search(
                r"differs|below Stata|residual|off by|not attributed|compiled Mata"
                r"|E1\(b\)|E1\(a\)",
                notes,
                re.IGNORECASE,
            )
        )
        if not (says_absent or says_gap):
            unexplained.append(entry["command"])

    assert not unexplained, (
        "E1 entries that do not say whether they have a Stata reference:\n"
        + "\n".join(unexplained)
    )


def test_every_e3_entry_declares_a_tolerance(entries: list[dict]) -> None:
    """`E3` 是可复现对拍，必须写明容差；没有容差就无法判定"通过"。"""

    missing = [
        entry["command"]
        for entry in entries
        if entry["level"] == "E3" and entry.get("tolerance") is None
    ]
    assert not missing, f"E3 entries without a tolerance: {missing}"


def test_e3_notes_do_not_admit_no_stata_comparison(entries: list[dict]) -> None:
    """`E3` 的语义是"可复现地对拍过"，`notes` 里不能同时说没对拍过。"""

    contradictory = []
    for entry in entries:
        if entry["level"] != "E3":
            continue
        notes = str(entry.get("notes") or "")
        # 注意 `never compared` 太宽: `xtnbreg, fe` 写的是「对数似然 never compared
        # with a REAL one」——那是说两个似然不可比,不是没和 Stata 对拍。所以这里
        # 要求短语指向 STATA 或"没有 fixture",而不是任何 `never compared`。
        if re.search(
            r"NOT been contrasted(?:\s+with\s+Stata)?"
            r"|no fixture was captured"
            r"|never compared (?:with|against|to) Stata",
            notes,
            re.IGNORECASE,
        ):
            contradictory.append(entry["command"])

    assert not contradictory, (
        f"E3 entries whose notes admit no Stata comparison: {contradictory}"
    )


def test_sources_types_are_not_a_level_proxy(entries: list[dict]) -> None:
    """`r_crosscheck` 可以出现在任何等级——有这个来源不等于对上了。

    这条测试本身不做断言，它记录的是一个**反直觉的事实**，防止后来者把
    "有 r_crosscheck 就是 E3" 当成规则写进代码：E1 里也有 8 条带它。
    如果将来有人想按 source 类型推断等级，这条会先失败并指向这里。
    """

    by_level: dict[str, set[str]] = {}
    for entry in entries:
        types = {source.get("type") for source in (entry.get("sources") or [])}
        by_level.setdefault(entry["level"], set()).update(types)

    # 每个等级都同时出现过"有 Stata 参考"和"没有 Stata 参考"的来源类型，
    # 所以任何单一 source 类型都不足以判定等级。
    assert "r_crosscheck" in by_level["E1"], (
        "this assertion documents reality: E1 entries DO carry r_crosscheck sources, "
        "so source type cannot decide the level"
    )
    assert "r_crosscheck" in by_level["E3"]
