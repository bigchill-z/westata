"""The alignment summary must carry the round-101-150 continuation.

Round 150 found the summary frozen at round 100: 4398 tests, 115 commits, and a "known gaps"
section still listing areg's absorbed F test, rcm and the RD point estimate as missing -- all of
which had since been fixed and verified. A reader starting from that document would re-investigate
finished work, which is the same failure the P1 recheck guards address for GAP_REGISTER.md.

What was added is a CONTINUATION, not a rewrite: the original round-1-100 numbers were correct
when written and stay in place as history. Section nine gives the current baseline; section fifteen
explicitly supersedes the old "known gaps" list.

One claim was corrected before committing: the draft said "vec 主表 已实现并验证", but the register
still records vec's drift term as NOT compared against Stata. The continuation must not over-claim.
"""

from __future__ import annotations

from pathlib import Path

SUMMARY = Path(__file__).resolve().parents[1] / "docs" / "STATA_ALIGNMENT_SUMMARY.md"


def test_the_continuation_exists() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    assert "续篇：第 101–150 轮" in text
    assert "第 1–100 轮" in text  # original history retained, not overwritten


def test_the_current_baseline_is_recorded() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    section = text.split("## 九、当前基线", 1)[1].split("## 十、", 1)[0]
    assert "4801 passed" in section
    assert "570" in section


def test_the_four_p0_items_have_their_final_states() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    section = text.split("## 十、四个 P0", 1)[1].split("## 十一、", 1)[0]
    for item in ("P0-1", "P0-2", "P0-3", "P0-4"):
        assert item in section, item
    # dfgls is the one that stays open; the three others are fixed.
    assert "未修复" in section
    assert "二进制 Mata" in section or "⬜" in section


def test_the_methodology_and_terminal_states_are_present() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    assert "## 十二、方法论" in text
    assert "凡参照必须来自捕获，不来自记忆" in text
    assert "## 十三、剩余缺口的三类终结状态" in text
    assert "A 类" in text and "B 类" in text and "C 类" in text
    assert "评分栏不能当工单" in text


def test_the_old_gaps_section_is_superseded_without_overclaiming() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    section = text.split("## 十五、", 1)[1]
    assert "上文第八节第 3 条" in section
    # The vec drift term is honestly left open (the phrase wraps across two lines).
    assert "漂移项" in section and "对拍" in section
    assert "记录而非声称" in section
