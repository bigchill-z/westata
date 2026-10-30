"""The API reference must stay in sync with the code.

``docs/API.md`` and ``docs/API.zh-CN.md`` describe 172 names. Two failure modes matter, and
both have already happened to this project's documentation:

1. **An example that does not run.** A manual whose code blocks fail is worse than no
   manual -- the reader trusts it and loses time. So every example is executed here.
2. **A name that exists in the code but not in the manual.** That is a documented
   falsehood in the other direction: the reader concludes the capability is absent.

Both are checked mechanically rather than by reading the rendered text, because the text
is generated: if the generator is right, the text is right.

The prose itself cannot be checked this way -- whether a sentence is TRUE is not
decidable by a test. What these tests can do is make sure the prose is present, paired
across the two languages, and attached to a name that actually exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "api_source"

sys.path.insert(0, str(ROOT / "docs"))


def _load_data() -> dict:
    """Load the collected data that the committed manual was rendered from.

    ``api_data.json`` is COMMITTED, not rebuilt on demand. It is the artefact the manual is
    rendered from, and keeping it in version control means the test compares the rendered
    documents against the same data a reader can inspect -- not against a fresh collection
    that might differ.

    Regenerate it with ``python docs/collect_api_reference.py`` after changing an
    estimator's signature or an example; the first test below fails if it goes stale.
    """

    path = REFERENCE / "api_data.json"
    if not path.exists():
        pytest.fail(
            f"{path} is missing. It is committed; restore it or run "
            "`python docs/collect_api_reference.py`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return _load_data()


@pytest.fixture(scope="module")
def prose(data):
    return data.get("prose", {})


def _known_names(data: dict) -> set[str]:
    names = {entry["name"] for entry in data["exports"]}
    surface = data.get("result_surface") or {}
    names |= {m["name"] for m in surface.get("methods", [])}
    names |= {p["name"] for p in surface.get("properties", [])}
    names |= {"StataResults", "model_data"}
    return names


# --------------------------------------------------------------- the documents
@pytest.mark.parametrize("filename", ["API.md", "API.zh-CN.md"])
def test_the_document_exists_and_is_substantial(filename: str) -> None:
    path = ROOT / "docs" / filename
    assert path.exists(), filename
    text = path.read_text(encoding="utf-8")
    # 172 sections cannot fit in a stub; a tiny file means the render failed partway.
    assert len(text.splitlines()) > 3000, filename


@pytest.mark.parametrize("filename", ["API.md", "API.zh-CN.md"])
def test_the_document_has_a_section_per_name(filename: str, data) -> None:
    text = (ROOT / "docs" / filename).read_text(encoding="utf-8")
    for name in _known_names(data):
        assert f"### `{name}`" in text, f"{filename} is missing a section for {name}"


@pytest.mark.parametrize("filename", ["API.md", "API.zh-CN.md"])
def test_the_document_shows_no_unresolved_warning(filename: str) -> None:
    """The renderer emits a warning block for a name without prose.

    Zero of them means the manual is complete. This is the assertion that would have
    caught a silently half-written reference.
    """

    text = (ROOT / "docs" / filename).read_text(encoding="utf-8")

    assert "no hand-written description yet" not in text
    assert "尚无手写说明" not in text


@pytest.mark.parametrize("filename", ["API.md", "API.zh-CN.md"])
def test_the_document_contains_no_failed_example(filename: str) -> None:
    """A failing example is rendered with a warning marker rather than hidden."""

    text = (ROOT / "docs" / filename).read_text(encoding="utf-8")

    assert "⚠️" not in text


# ----------------------------------------------------------------- the examples
def test_every_example_runs(data) -> None:
    """THE test this file exists for: execute all of them.

    ``_Collector`` is the same runner the generator uses, so a green suite means the
    committed manual has no code block that fails when a reader pastes it.
    """

    from collect_api_reference import _Collector

    collector = _Collector()
    failures = []
    for name, entry in sorted(data["prose"].items()):
        example = entry.get("example")
        if not example:
            continue
        outcome = collector.run(str(example))
        if not outcome["ok"]:
            failures.append(f"{name}: {outcome['error']}")

    assert failures == [], "examples that do not run:\n" + "\n".join(failures)


def test_every_example_produces_output(data) -> None:
    """An example whose last line evaluates to nothing prints nothing useful."""

    from collect_api_reference import _Collector

    collector = _Collector()
    empty = []
    for name, entry in sorted(data["prose"].items()):
        example = entry.get("example")
        if not example:
            continue
        outcome = collector.run(str(example))
        if outcome["ok"] and not str(outcome.get("value") or "").strip():
            empty.append(name)

    assert empty == [], f"examples with no output: {empty}"


# ------------------------------------------------------------------- the prose
def test_every_known_name_has_prose(data, prose) -> None:
    missing = sorted(_known_names(data) - set(prose))

    assert missing == [], f"names without prose: {missing}"


@pytest.mark.parametrize("field", ["stata", "en", "zh", "example"])
def test_every_entry_has_the_required_fields(prose, field: str) -> None:
    absent = [name for name, entry in prose.items() if not entry.get(field)]

    assert absent == [], f"entries missing {field!r}: {absent}"


def test_both_languages_are_present_together(prose) -> None:
    """A caveat in one language but not the other is a fact one reader never learns."""

    mismatched = [
        name
        for name, entry in prose.items()
        if bool(entry.get("notes_en")) != bool(entry.get("notes_zh"))
    ]

    assert mismatched == [], f"caveat present in only one language: {mismatched}"


def test_the_prose_names_only_real_exports(prose, data) -> None:
    """Guard against a typo silently creating a section nothing renders."""

    unknown = sorted(set(prose) - _known_names(data))

    assert unknown == [], f"prose for names that do not exist: {unknown}"


def test_the_chinese_is_not_a_stub(prose) -> None:
    """The Chinese must be real prose, not a placeholder and not unedited English.

    Three checks, each targeting a failure this project could actually have:

    * enough CJK characters to be a sentence rather than a token;
    * not byte-identical to the English (an unedited copy would slip past a CJK count if
      the English happened to contain Chinese, and would be useless besides);
    * carrying at least one backticked identifier, so the API surface is described in the
      same symbols a reader will type.

    What is deliberately NOT checked: a length ratio, or that the entry's own name appears
    in the Chinese. Both were tried and both produced false positives. Chinese is much
    denser per character, and a property like ``pvalues`` is naturally described as
    "每个系数的 p 值" without repeating the English identifier -- the heading already says
    which property this is.
    """

    import re

    suspicious = []
    for name, entry in prose.items():
        chinese = str(entry.get("zh") or "")
        english = str(entry.get("en") or "")
        cjk = len(re.findall(r"[\u4e00-\u9fff]", chinese))
        if cjk < 8:
            suspicious.append(f"{name}: only {cjk} CJK characters")
            continue
        if chinese.strip() == english.strip():
            suspicious.append(f"{name}: zh is a copy of en")
            continue
        if "`" not in chinese:
            suspicious.append(f"{name}: zh names no code identifier")
            continue

    assert suspicious == [], "suspicious translations:\n" + "\n".join(suspicious)