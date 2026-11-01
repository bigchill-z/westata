"""The single source of truth for what gets replicated in Card & Krueger (1994).

Both sides of the replication read their specifications from this module: the Stata
driver in ``run_stata.py`` and the ``westata`` driver in ``run_westata.py``.  Keeping one
description stops the two samples and control sets from drifting apart, which is the
failure mode that makes a "difference" meaningless.

What is replicated
------------------
**Table 3, rows 4-5**: the raw difference-in-differences in FTE employment, which is
where the paper's headline number lives -- the relative gain in New Jersey is
**2.76 FTE employees, or 13 percent, t = 2.03**.

**Table 4, columns (i)-(v)**: the reduced-form regressions of the change in FTE
employment.  Columns (i)-(ii) use a New Jersey dummy; (iii)-(v) use ``GAP``, the
proportional wage increase needed to reach the new minimum, which is zero in
Pennsylvania and therefore also a difference-in-differences.

Why not Table 5 or the later tables
-----------------------------------
Table 5 splits stores by wage level; its estimates need the ``$4.25`` indicator built
from a wave-1 wage that is missing for part of the sample, and the paper reports the
split only for New Jersey.  The tables after that move to other data sources (BLS,
CPS).  Tables 3 and 4 are the paper's core and are entirely within the public file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    """One estimated model.

    ``stata_command`` and ``formula`` are both generated from the same field list, so a
    control added to one side cannot be missing from the other.
    """

    id: str
    table: str
    column: str
    depvar: str
    regressors: tuple[str, ...]
    label: str
    #: ``True`` when the sample is the 357 stores of Table 4; ``False`` uses Table 3's
    #: balanced sample (which additionally requires nothing beyond known employment).
    table4_sample: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def formula(self) -> str:
        return f"{self.depvar} ~ " + " + ".join(self.regressors)

    def stata_command(self) -> str:
        return f"regress {self.depvar} " + " ".join(self.regressors)


#: The control sets, spelled out.  "Chain dummies" is three of four chains with Burger
#: King omitted; writing them explicitly keeps the omitted category visible.
CHAIN = ("chain2", "chain3", "chain4")
OWNERSHIP = ("co_owned",)
CHAIN_AND_OWNERSHIP = CHAIN + OWNERSHIP
#: Region dummies: two regions of New Jersey and two of eastern Pennsylvania.  The other
#: location cells (shore, the omitted New Jersey region) are the reference group.
REGIONS = ("southj", "centralj", "pa1", "pa2")

SPECS: tuple[Spec, ...] = (
    Spec(
        id="T4_i",
        table="Table 4",
        column="(i)",
        depvar="d_fte",
        regressors=("nj",),
        label="New Jersey dummy, no controls",
    ),
    Spec(
        id="T4_ii",
        table="Table 4",
        column="(ii)",
        depvar="d_fte",
        regressors=("nj",) + CHAIN_AND_OWNERSHIP,
        label="New Jersey dummy + chain and ownership dummies",
    ),
    Spec(
        id="T4_iii",
        table="Table 4",
        column="(iii)",
        depvar="d_fte",
        regressors=("gap",),
        label="initial wage gap, no controls",
    ),
    Spec(
        id="T4_iv",
        table="Table 4",
        column="(iv)",
        depvar="d_fte",
        regressors=("gap",) + CHAIN_AND_OWNERSHIP,
        label="initial wage gap + chain and ownership dummies",
    ),
    Spec(
        id="T4_v",
        table="Table 4",
        column="(v)",
        depvar="d_fte",
        regressors=("gap",) + CHAIN_AND_OWNERSHIP + REGIONS,
        label="initial wage gap + controls + region dummies",
    ),
)

#: What the paper prints.  ``b``/``se`` are the two-decimal values in Table 4.
PAPER: dict[str, dict[str, float]] = {
    "T4_i": {"term": "nj", "b": 2.33, "se": 1.19, "N": 357.0},
    "T4_ii": {"term": "nj", "b": 2.30, "se": 1.20, "N": 357.0},
    "T4_iii": {"term": "gap", "b": 15.65, "se": 6.08, "N": 357.0},
    "T4_iv": {"term": "gap", "b": 14.92, "se": 6.21, "N": 357.0},
    "T4_v": {"term": "gap", "b": 11.91, "se": 7.39, "N": 357.0},
}

#: Table 3's difference-in-differences, quoted in the text rather than a regression.
PAPER_TABLE3 = {
    "did_fte": 2.76,
    "percent": 13.0,
    "t_stat": 2.03,
    "note": "paper text, Section III.A: 'The relative gain ... is 2.76 FTE employees "
            "(or 13 percent), with a t statistic of 2.03'",
}

#: Sample statistics the paper prints in the Table 4 notes, used as a check on the
#: sample construction before any coefficient is compared.
PAPER_SAMPLE = {"n": 357, "d_fte_mean": -0.237, "d_fte_sd": 8.825}

SOURCE = "paper PDF (AER 84(4), September 1994), Tables 3 and 4, pp. 779-781"
