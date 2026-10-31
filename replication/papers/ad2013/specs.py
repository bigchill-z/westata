"""The single source of truth for what gets replicated in Autor & Dorn (2013).

Both drivers read their specifications here, so the Stata side and the ``westata`` side
cannot drift apart in sample restrictions or control sets.

What is replicated
------------------
**Table 3** (six rows): the relationship between a commuting zone's routine-task
specialisation and its change in routine-task intensity, i.e. the paper's evidence that
routine jobs declined where they started out concentrated.  Row 4-6 use the stacked
differences sample (1980-2005); rows 1-3 are the two long differences and their stack.

**Table 4** (five rows): the growth of low-skill service employment against the same
routine-task measure, by decade.  This is the "polarization" half of the argument.

Why these two tables
--------------------
Every specification is ``regress`` with ``[aw=timepwt48]`` and ``cluster(statefip)`` and a
full set of state dummies -- squarely inside the command set ``westata`` claims machine
precision for.  The remaining tables in the do-file either reuse these specifications
with a different left-hand side (mechanical) or move to microdata the kit does not
redistribute.

The baseline values are taken from the authors' own ``log/czone_analysis.log``, shipped
in the archive, rather than from a typeset table.  That is a stronger reference: it was
produced by the authors on their own machine with their own Stata, at full precision,
which is exactly what these drivers capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    id: str
    table: str
    row: str
    depvar: str
    term: str
    condition: str | None
    #: Period dummies and other non-state regressors, in the do-file's written order.
    extra_regressors: tuple[str, ...] = ()
    label: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def stata_command(self) -> str:
        """The command as the authors' do-file writes it, with ``i.statefip`` spelled out.

        The do-file uses ``xi i.statefip`` to create ``_Istatefip_*`` columns and then
        lists ``_Istatefip*``.  ``i.statefip`` is the modern equivalent and leaves the
        same design matrix; using it keeps this file readable while producing an
        identical regression.
        """

        parts = ["regress", self.depvar, self.term, *self.extra_regressors, "i.statefip"]
        if self.condition:
            parts.append(f"if {self.condition}")
        parts.append("[aw=timepwt48]")
        parts.append(", cluster(statefip)")
        return " ".join(parts)


#: Both tables' regressions share the same weighting, clustering and state-dummy set.
COMMON = dict(extra_regressors=())

SPECS: tuple[Spec, ...] = (
    # ---- Table 3: routine-task intensity -----------------------------------------
    Spec("T3_1", "Table 3", "(1)", "d_rpc", "l_sh_routine33a", "t1==1",
         label="long difference, 1990-2000"),
    Spec("T3_2", "Table 3", "(2)", "d_rpc", "l_sh_routine33a", "t2==1",
         label="long difference, 2000-2005"),
    Spec("T3_3", "Table 3", "(3)", "d_rpc", "l_sh_routine33a", "t1==1 | t2==1",
         extra_regressors=("t2",), label="stacked long differences"),
    Spec("T3_4", "Table 3", "(4)", "d_sh_routine33a", "l_sh_routine33a", "yr>=1980",
         extra_regressors=("t2", "t3"), label="stacked differences, all routine employment"),
    Spec("T3_5", "Table 3", "(5)", "d_sh_routine33a_c", "l_sh_routine33a", "yr>=1980",
         extra_regressors=("t2", "t3"), label="column (4), college workers"),
    Spec("T3_6", "Table 3", "(6)", "d_sh_routine33a_nc", "l_sh_routine33a", "yr>=1980",
         extra_regressors=("t2", "t3"), label="column (4), non-college workers"),
    # ---- Table 4: low-skill service employment ----------------------------------
    Spec("T4_1", "Table 4", "(1)", "d_shocc1_service_nc", "l_sh_routine33a", "t50==1",
         label="1950-1970"),
    Spec("T4_2", "Table 4", "(2)", "d_shocc1_service_nc", "l_sh_routine33a", "t70==1",
         label="1970-1980"),
    Spec("T4_3", "Table 4", "(3)", "d_shocc1_service_nc", "l_sh_routine33a", "t1==1",
         label="1990-2000"),
    Spec("T4_4", "Table 4", "(4)", "d_shocc1_service_nc", "l_sh_routine33a", "t2==1",
         label="2000-2005"),
    Spec("T4_5", "Table 4", "(5)", "d_shocc1_service_nc", "l_sh_routine33a", "t3==1",
         label="1980-2005 (ten-year equivalent)"),
)

#: The authors' own logged estimates, transcribed from ``log/czone_analysis.log``.
#: ``b``/``se`` are the full-precision values their Stata printed, not a rounded table.
PAPER: dict[str, dict[str, float]] = {
    "T3_1": {"b": 0.6950717, "se": 0.0614371, "N": 675},
    "T3_2": {"b": 0.4899328, "se": 0.0760501, "N": 660},
    "T3_3": {"b": 0.6185140, "se": 0.0441480, "N": 1335},
    "T3_4": {"b": -0.2540723, "se": 0.0225534, "N": 2166},
    "T3_5": {"b": -0.1526445, "se": 0.0244188, "N": 2166},
    "T3_6": {"b": -0.2945519, "se": 0.0181593, "N": 2166},
    "T4_1": {"b": -0.1333048, "se": 0.0203481, "N": 722},
    "T4_2": {"b": 0.0415499, "se": 0.0315114, "N": 722},
    "T4_3": {"b": 0.0825400, "se": 0.0279456, "N": 722},
    "T4_4": {"b": 0.0843262, "se": 0.0454320, "N": 722},
    "T4_5": {"b": 0.3540265, "se": 0.1100271, "N": 722},
}

SOURCE = "authors' log/czone_analysis.log, shipped inside the replication archive"
