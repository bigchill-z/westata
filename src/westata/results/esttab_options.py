"""esttab 屏幕输出类选项的完整实现。

与 `estimates table` 的根本差别（实测）：

    esttab 的 cells 只出 TWO 行：系数行 + 一个"第二行"（se / t / p / ci 之一）。
    `esttab m1 m2, b t p` 不产生三行，而是"最后一个胜出" -> 只有 p。
    `esttab m1 m2, b se p` 直接报错 `only one allowed of ...`。

    esttab 的默认星号是 5%/1%/0.1%（* p<0.05, ** p<0.01, *** p<0.001），
    而 estimates table 的 star(0.1 0.05 0.01) 是 10%/5%/1%。

    esttab 默认在 (1)(2) 之下打印**依赖变量名**，`mtitle(...)` 才换成模型名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: esttab's own default star thresholds -- NOT the 0.1/0.05/0.01 that `estimates table` takes.
ESTTAB_DEFAULT_STARS: tuple[float, ...] = (0.05, 0.01, 0.001)

#: The "second row" cells esttab accepts. Only ONE may be active; the last one wins.
SECOND_ROW_CELLS = ("se", "t", "p", "ci")


@dataclass
class EsttabOptions:
    """Everything `esttab` accepts that affects the SCREEN table.

    File export (`using`, `style`, `append`/`replace`, `title`, `prehead`/`postfoot`) is out of
    scope, as is LaTeX/HTML/RTF output.
    """

    # -- cells -----------------------------------------------------------------------------
    #: Which second-row cell to print; ``None`` prints the coefficient alone. The last of
    #: ``se``/``t``/``p``/``ci`` in the user's option string wins, matching `esttab`.
    second_row: str | None = "se"
    #: ``ci`` prints ``[lo,hi]`` and widens the columns.
    # -- row selection ---------------------------------------------------------------------
    keep: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    order: tuple[str, ...] = ()
    # -- labels ----------------------------------------------------------------------------
    rename: dict[str, str] = field(default_factory=dict)
    varlabels: dict[str, str] = field(default_factory=dict)
    #: ``mtitle(...)`` replaces the dependent-variable row.
    model_titles: tuple[str, ...] = ()
    #: ``collabels(...)`` ADDS a row below the dependent-variable row.
    column_labels: tuple[str, ...] = ()
    #: ``label`` uses variable labels and widens the label column.
    use_labels: bool = False
    #: ``noabbrev`` keeps long column names intact.
    no_abbrev: bool = False
    # -- formatting ------------------------------------------------------------------------
    #: ``starlevels(...)`` overrides `ESTTAB_DEFAULT_STARS`.
    star_levels: tuple[float, ...] = ESTTAB_DEFAULT_STARS
    #: ``star`` / ``nostar`` gates the star run entirely.
    stars: bool = True
    #: ``level(90)`` widens or narrows the ``ci`` interval.
    level: float = 95.0
    #: ``brac`` / ``paren`` change the second row's delimiters.
    second_row_brackets: tuple[str, str] = ("(", ")")
    #: ``nomargin`` drops the leading blank line. Stata's own default here is NO margin, so the
    #: `nomargin` drops the leading blank line; Stata prints it by default, so this is False.
    no_margin: bool = False
    #: ``notype`` is NOT reachable here. esttab.ado:409 adds it only when a ``using`` file is
    #: written:
    #:
    #:     if "`type'"=="" & `"`macval(using)'"'!="" local notype notype
    #:
    #: and calling ``esttab ..., se notype`` without ``using`` exits with r(7) -- verified. The
    #: dependent-variable row therefore always prints on screen. Kept as a field so the option is
    #: recognised rather than silently ignored, but it never suppresses anything.
    no_type: bool = False
    #: ``numbers`` / ``nonumbers``.
    numbers: bool = True
    #: ``nocons`` drops the constant row. Measured: `esttab m1 m2, se nocons` prints 18 lines
    #: against the default 21, i.e. exactly the `_cons` block (two lines plus its blank line).
    no_constant: bool = False
    #: ``stats(...)`` footer rows.
    stats: tuple[str, ...] = ()

    def refine(self, token: str) -> None:
        """Apply ONE option token, in the order Stata would read them.

        `esttab` resolves the cells by "last one wins", so the tokens must be applied in the order
        given rather than merged as a set.
        """

        head = token.split("(")[0].strip().lower()
        if head in SECOND_ROW_CELLS:
            self.second_row = head
        elif head == "b":
            # `esttab ..., b` ALONE still prints the t statistic beneath the coefficient --
            # measured, not assumed:
            #     's                   0.107***        0.107***'
            #     '                  (16.74)         (17.09)   '
            self.second_row = "t"
        elif head == "star":
            self.stars = True
        elif head == "nostar":
            self.stars = False
        elif head == "brac":
            self.second_row_brackets = ("[", "]")
        elif head == "paren":
            self.second_row_brackets = ("(", ")")
        elif head == "nomargin":
            # MEASURED: `esttab m1 m2, nomargin` and `..., margin` both print the leading blank
            # line, so the option has no effect on the screen table.
            self.no_margin = False
        elif head == "notype":
            self.no_type = True
        elif head == "label":
            self.use_labels = True
        elif head == "noabbrev":
            self.no_abbrev = True
        elif head == "nonumbers":
            self.numbers = False
        elif head == "numbers":
            self.numbers = True
        elif head == "nocons":
            self.no_constant = True
