"""Specifications for the Acemoglu-Cantoni-Johnson-Robinson (2001) replication.

The paper
---------
"The Consequences of Radical Reform: The French Revolution", *American Economic Review*
92(5): 1231-1252.  It asks whether the institutional reforms imposed during the French
Revolutionary period raised long-run economic growth in the German states that experienced
them.

The design
----------
``urbrate`` (the urbanisation rate) is regressed on the years of French presence
(``fpresence``) interacted with century dummies, with state fixed effects and year dummies,
clustered on state.  The identifying variation is within a state over time: states occupied
for longer should urbanise faster after the reforms.

**What is and is not replicated here** — read this before the numbers
--------------------------------------------------------------------
The replication package (``20100816_replication10``) contains the do-file, two data files
and a variable README.  It contains **no log of the authors' output**, and the AER article is
behind a paywall, so there is **no published coefficient this replication can be checked
against**.

That changes what the exercise can claim:

* it **cannot** say "the paper's Table 3 is reproduced";
* it **can** say "westata reproduces Stata exactly on the paper's own specifications, on the
  authors' own data".

The baseline is therefore **StataNow running the authors' do-file commands**, captured here
at full precision.  This is a self-consistency replication, and the report says so in its
first line rather than implying a table match it cannot demonstrate.

Tables covered
--------------
Table 3 (columns 1-4): the baseline fixed-effects estimates under four sample/weight
variants.  Table 4 adds time-interacted covariates; its first column is the
outlier-excluded variant.  Both are ``xtreg, fe`` with ``[aweight]`` and clustering on
``id`` — the estimation surface this replication exercises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The five French-presence terms, one per post-reform century.
FPRESENCE = ("fpresence1750", "fpresence1800", "fpresence1850", "fpresence1875", "fpresence1900")

#: Year dummies, the sample years being 1700-1900 by fifties.
YEAR_DUMMIES = ("yr1750", "yr1800", "yr1850", "yr1875", "yr1900")

#: Weight variable the authors use (population in 1750).
WEIGHT = "totalpop1750"


@dataclass(frozen=True)
class Spec:
    id: str
    table: str
    column: str
    label: str
    #: ``True`` for the western (west of the Elbe) sample the paper's baseline uses.
    west_only: bool = True
    weighted: bool = True
    #: States excluded from the sample, as in the paper's Table 4 column 1.
    drop_ids: tuple[int, ...] = ()
    controls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def regressors(self) -> tuple[str, ...]:
        return FPRESENCE + self.controls + YEAR_DUMMIES

    def stata_command(self) -> str:
        """The command as the authors' do-file writes it."""

        parts = ["xtreg", "urbrate", *self.regressors]
        if self.west_only:
            parts.append("if westelbe==1")
        if self.drop_ids:
            ids = " ".join(str(i) for i in self.drop_ids)
            parts.append(("& " if self.west_only else "if ") + f"!inlist(id, {ids})")
        if self.weighted:
            parts.append(f"[aweight={WEIGHT}]")
        parts.append(", fe cluster(id)")
        return " ".join(parts)


SPECS: tuple[Spec, ...] = (
    Spec("T3_c1", "Table 3", "(1)", "western states, weighted"),
    Spec("T3_c2", "Table 3", "(2)", "western states, unweighted", weighted=False),
    Spec("T3_c3", "Table 3", "(3)", "all states, weighted", west_only=False),
    Spec("T3_c4", "Table 3", "(4)", "all states, unweighted", west_only=False, weighted=False),
    Spec("T4_c1", "Table 4", "(1)", "western states, weighted, dropping state 2",
         drop_ids=(2,)),
)

#: Filled in by ``run_stata`` from a live StataNow run: this replication's reference is
#: Stata itself, not a published table, because neither the authors' log nor the article's
#: tables were available.  See this module's docstring.
SOURCE = (
    "StataNow 19.5 running the authors' do-file commands on the delivered data "
    "(self-consistency baseline: no author log and no accessible article tables)"
)

#: Why there is no ``PAPER`` dictionary here, unlike the other replications.
NO_PUBLISHED_BASELINE = (
    "The replication package ships no log, and the AER article is paywalled, so no published "
    "coefficient was available to check against. The baseline is Stata's own output."
)
