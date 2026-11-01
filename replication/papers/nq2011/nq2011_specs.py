"""The single source of truth for what gets replicated in Nunn & Qian (2011).

Both drivers read their specifications here.

What is replicated
------------------
**Table 2, panel "Total Population", columns (1)-(3)**: the flexible specification that
traces the potato's effect on population over eleven centuries, with country fixed
effects and year fixed effects.

**Table 2, panel "City Population Share", columns (4)-(6)**: the same for urbanisation.

Why these six
-------------
They are the paper's baseline and they use ``areg ... i.year, a(isocode) cluster(isocode)``
-- a **panel** fixed-effects estimator.  The previous three replications in this repository
covered cross-sectional OLS and IV only (see ``../ck1994`` and ``../adh2013``), so this is
the first exercise of westata's ``areg`` / ``xtreg`` surface.

The areg-versus-xtreg finding
-----------------------------
On this data the point estimate is identical across every path, but the standard error is
not::

    areg      se = 0.0171579      <- what the authors ran, and what they report
    xtreg fe  se = 0.0164182      <- 4.5 percent smaller
    reghdfe   se = 0.0164182

``areg`` and ``xtreg`` differ in how many parameters their small-sample cluster adjustment
counts: ``areg`` counts the absorbed country effects towards ``K``, ``xtreg`` does not.
Both are Stata.  ``westata``'s ``FEOLS`` and ``PanelOLS`` both agree with ``xtreg``.

**That means the replication does not "fail to match the paper" -- it matches Stata's
``xtreg`` exactly and differs from Stata's ``areg`` by a documented convention.**  The
report separates those two claims, because conflating them would either hide a real
convention difference or invent a defect in westata.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The potato variables, indexed by the eleven post-1000 census years.
POTATO_YEARS = (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1750, 1800, 1850, 1900)


@dataclass(frozen=True)
class Spec:
    id: str
    table: str
    column: int
    panel: str
    depvar: str
    label: str
    controls: tuple[str, ...] = ()

    @property
    def potato_terms(self) -> tuple[str, ...]:
        return tuple(f"ln_wpot_{year}" for year in POTATO_YEARS)

    @property
    def all_regressors(self) -> tuple[str, ...]:
        return self.potato_terms + self.controls

    def stata_command(self) -> str:
        """The command as the authors' do-file writes it.

        ``xi: areg ... i.year`` -- the modern equivalent is ``areg ... i.year``, which
        produces the same design matrix and the same output, so the ``xi:`` prefix is
        dropped rather than emulated.
        """

        parts = ["areg", self.depvar]
        parts.extend(self.all_regressors)
        parts.append("i.year")
        parts.append(", a(isocode) cluster(isocode)")
        return " ".join(parts)


def _scaled(prefix: str, years: tuple[int, ...] = POTATO_YEARS) -> tuple[str, ...]:
    return tuple(f"{prefix}_{year}" for year in years)


#: ``ln_oworld_<year>``: the rest-of-world potato stock, the main robustness control.
OWORLD = _scaled("ln_oworld")
#: Geography interactions: elevation, tropical share, ruggedness, all interacted by year.
GEOGRAPHY = _scaled("ln_elevation") + _scaled("ln_tropical") + _scaled("ln_rugged")

SPECS: tuple[Spec, ...] = (
    # ---- panel A: total population -------------------------------------------
    Spec("T2_c1", "Table 2", 1, "Total population", "ln_population",
         "potato only, country and year FE"),
    Spec("T2_c2", "Table 2", 2, "Total population", "ln_population",
         "adds rest-of-world potato stock", controls=OWORLD),
    Spec("T2_c3", "Table 2", 3, "Total population", "ln_population",
         "adds geography interactions", controls=OWORLD + GEOGRAPHY),
    # ---- panel B: city population share -------------------------------------
    Spec("T2_c4", "Table 2", 4, "City population share", "city_pop_share",
         "potato only, country and year FE"),
    Spec("T2_c5", "Table 2", 5, "City population share", "city_pop_share",
         "adds rest-of-world potato stock", controls=OWORLD),
    Spec("T2_c6", "Table 2", 6, "City population share", "city_pop_share",
         "adds geography interactions", controls=OWORLD + GEOGRAPHY),
)

#: The authors' logged estimates, transcribed from ``Replication_country_level_results.log``.
#: Two potato coefficients per column: the 1800 and 1900 interactions are the ones the paper
#: highlights, so both are recorded.
PAPER: dict[str, dict[str, float]] = {
    "T2_c1": {"ln_wpot_1800": (0.0725947, 0.0138663), "ln_wpot_1900": (0.1213388, 0.0171579)},
    "T2_c2": {"ln_wpot_1800": (0.0478328, 0.0154607), "ln_wpot_1900": (0.0923494, 0.0208822)},
    "T2_c3": {"ln_wpot_1800": (0.0408335, 0.0222412), "ln_wpot_1900": (0.0795030, 0.0240815)},
    "T2_c4": {"ln_wpot_1800": (0.0020031, 0.0008949), "ln_wpot_1900": (0.0118203, 0.0022693)},
    "T2_c5": {"ln_wpot_1800": (0.0016294, 0.0012571), "ln_wpot_1900": (0.0123354, 0.0023503)},
    "T2_c6": {"ln_wpot_1800": (0.0017523, 0.0016902), "ln_wpot_1900": (0.0100279, 0.0031933)},
}

SOURCE = "authors' Replication_country_level_results.log, deposited with the data"
