"""Specifications for the Autor-Dorn-Hanson-Majlesi (2020) replication.

The paper
---------
"The Fall of the Labor Share and the Rise of Superstar Firms" is a different paper; this one
is **ADHM (2020), "Importing Political Polarization? The Electoral Consequences of Rising
Trade Exposure"**, published in the *American Economic Review* 110(10): 3139-3183.

It asks whether rising Chinese import competition polarised US electoral districts between
2002 and 2010: districts hit harder by import competition moved toward the ideological
extremes in their campaign contributions, turnout and vote shares.

Why this paper for this replication
-----------------------------------
The preceding four replications all used one-dimensional clustering:

===========  ====================================
ADH 2013     ``cluster(statefip)``
CK 1994      no clustering (plain OLS)
AD 2013      ``cluster(statefip)``
NQ 2011      ``cluster(isocode)``
===========  ====================================

ADHM cluster on **two dimensions at once**: ``cluster(czone congressionaldistrict)``.  A
congressional district nests inside a commuting zone, so the two-way correction matters --
observations in the same district are correlated, and districts in the same commuting zone
are correlated through local labour markets.

That makes this the first replication here that exercises two-way clustering, and it is where
``westata`` has a **documented boundary**: ``IV2SLS`` raises ``UnsupportedError`` for two-way
clusters.  The replication therefore separates two claims:

* the **coefficient** is reproducible exactly (15.83783 against the authors' 15.83783);
* the **two-way clustered standard error** is not, and the replication reports what the
  single-way options give instead (8.32 and 8.43 against the authors' 9.24), so the size of
  the gap is measured rather than asserted.

Data
----
``house_2002_2016.dta``: 3,772 congressional districts, with the authors' own log recording
every estimate at full precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The 2002-2010 change in total campaign contributions, the paper's Table 3 left-hand side.
DEPVAR = "dhs2_tot_cont_2002_2010"

#: Import exposure: the endogenous regressor, instrumented by the lagged other-country share.
ENDOG = "d_imp_usch_pd"
INSTRUMENT = "d_imp_otch_lag_pd"

#: 2002 district population share, the aweight the authors use.
WEIGHT = "sh_district_2002"

#: The two clustering dimensions, in the order the authors' ivreg2 call lists them.
CLUSTERS = ("czone", "congressionaldistrict")

#: Base controls: manufacturing share, routine-task share, offshorability.
BASE_CONTROLS = ("l_shind_manuf_cbp", "l_sh_routine33", "l_task_outsource")

#: Census-division dummies, as ``reg*`` in the authors' do-file.
REGION_DUMMIES = (
    "reg_midatl", "reg_encen", "reg_wncen", "reg_satl",
    "reg_escen", "reg_wscen", "reg_mount", "reg_pacif",
)

#: Prior presidential vote shares, included only in the ``full_ctrl`` block (column 5).
PRES_VOTE = ("shnr_pres2000", "shnr_pres1996")

#: The demographic control block used from column (4) onward.
DEMOGRAPHICS = (
    "l_sh_pop_f", "l_sh_pop_edu_c", "l_sh_fborn",
    "l_sh_pop_age_1019", "l_sh_pop_age_2029", "l_sh_pop_age_3039",
    "l_sh_pop_age_4049", "l_sh_pop_age_5059", "l_sh_pop_age_6069",
    "l_sh_pop_age_7079", "l_sh_pop_age_8000",
    "l_sh_pop_white", "l_sh_pop_black", "l_sh_pop_asian", "l_sh_pop_hispanic",
)


@dataclass(frozen=True)
class Spec:
    id: str
    column: str
    label: str
    controls: tuple[str, ...] = ()

    #: Every column of Table 3 has the same dependent variable; it lives on the Spec so a
    #: driver never has to hard-code it.
    depvar: str = DEPVAR

    @property
    def regressors(self) -> tuple[str, ...]:
        """The complete exogenous control list -- already includes BASE_CONTROLS where used.

        An earlier version added BASE_CONTROLS here, which silently gave column 1 three
        controls the authors did not include.  The field is now the whole truth so the
        column-by-column regressor counts match the do-file.
        """

        return self.controls

    def cell(self) -> str:
        """The coefficient cell the authors' log reports for this column."""

        return f"({self.column})"


SPECS: tuple[Spec, ...] = (
    Spec("T3_A1", "1", "no controls"),
    Spec("T3_A2", "2", "manufacturing share, routine share, offshorability", BASE_CONTROLS),
    Spec("T3_A3", "3", "adds Census-division dummies", BASE_CONTROLS + REGION_DUMMIES),
    Spec("T3_A4", "4", "demographics + regions", BASE_CONTROLS + DEMOGRAPHICS + REGION_DUMMIES),
    Spec(
        "T3_A5", "5", "full controls: as column 4 plus prior presidential vote shares",
        BASE_CONTROLS + DEMOGRAPHICS + REGION_DUMMIES + PRES_VOTE,
    ),
)

#: The authors' logged estimates for Table 3 panel A (total contributions, 2002-2010).
#: Transcribed from ``log/house_2002_2016_analysis.log``; seven significant digits.
PAPER: dict[str, dict[str, float]] = {
    "T3_A1": {"b": 15.83783, "se": 9.236939, "N": 3772},
    "T3_A2": {"b": 28.69324, "se": 21.29582, "N": 3772},
    "T3_A3": {"b": 33.30375, "se": 21.12325, "N": 3772},
    "T3_A4": {"b": 38.96939, "se": 21.22885, "N": 3772},
    "T3_A5": {"b": 37.23199, "se": 21.04598, "N": 3772},
}

SOURCE = "authors' log/house_2002_2016_analysis.log, ivreg2 with cluster(czone congressionaldistrict)"
