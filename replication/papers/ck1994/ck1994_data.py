"""Read the Card & Krueger (1994) public-use data file.

Why this module exists
----------------------
The paper's data is a 1997-era fixed-width ASCII file (``public.dat``) with a separate
codebook giving column positions.  There is no ``.dta`` and no Stata do-file in the
archive, only a SAS check program.  Building the reader here -- with the column map
transcribed from the codebook and asserted against the counts the codebook itself
states -- is what turns "the data is downloadable" into "the data is usable".

Two facts about the file that matter and are easy to get wrong:

* **Missing values are ``.``**, not blanks.  A blank field would parse to ``NaN`` too,
  but the file uses ``.`` for a store that was asked and did not answer, and blanks do
  not occur.  Treating ``.`` as text and then to-numeric with ``errors="coerce"`` gets
  both right.
* **The ``STATUS2`` codes drive the employment definition.**  Six stores closed
  permanently and four closed temporarily; the paper counts the first as zero employment
  and the second as missing.  That single rule is the difference between a
  difference-in-differences of 2.33 and one of 2.75, so it lives here rather than in a
  notebook.

The column map is the codebook's, in order, so a reader can check any field against the
PDF that ships with the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Path to the raw fixed-width file inside the downloaded archive.
RAW = DATA / "ck1994" / "public.dat"

#: ``(start, end, name)`` in 1-based inclusive columns, transcribed from ``codebook``.
#: The order is the codebook's, which also makes the transcription checkable line by line.
COLUMNS: tuple[tuple[int, int, str], ...] = (
    (1, 3, "sheet"),
    (5, 5, "chain"),
    (7, 7, "co_owned"),
    (9, 9, "state"),
    (11, 11, "southj"),
    (13, 13, "centralj"),
    (15, 15, "northj"),
    (17, 17, "pa1"),
    (19, 19, "pa2"),
    (21, 21, "shore"),
    (23, 24, "ncalls"),
    (26, 30, "empft"),
    (32, 36, "emppt"),
    (38, 42, "nmgrs"),
    (44, 48, "wage_st"),
    (50, 54, "inctime"),
    (56, 60, "firstinc"),
    (62, 62, "bonus"),
    (64, 68, "pctaff"),
    (70, 70, "meals"),
    (72, 76, "open"),
    (78, 82, "hrsopen"),
    (84, 88, "psoda"),
    (90, 94, "pfry"),
    (96, 100, "pentree"),
    (102, 103, "nregs"),
    (105, 106, "nregs11"),
    (108, 108, "type2"),
    (110, 110, "status2"),
    (112, 117, "date2"),
    (119, 120, "ncalls2"),
    (122, 126, "empft2"),
    (128, 132, "emppt2"),
    (134, 138, "nmgrs2"),
    (140, 144, "wage_st2"),
    (146, 150, "inctime2"),
    (152, 156, "firstin2"),
    (158, 158, "special2"),
    (160, 160, "meals2"),
    (162, 166, "open2r"),
    (168, 172, "hrsopen2"),
    (174, 178, "psoda2"),
    (180, 184, "pfry2"),
    (186, 190, "pentree2"),
    (192, 193, "nregs2"),
    (195, 196, "nregs112"),
)

#: Expected counts the codebook states outright; used as an integrity check, because a
#: mis-set column boundary usually shows up as a wrong ``STATUS2`` distribution rather
#: than as a parse error.
CODEBOOK_COUNTS = {"observations": 410, "state_new_jersey": 331, "state_pennsylvania": 79}
STATUS2_COUNTS = {1: 399, 2: 2, 3: 6, 4: 1, 5: 1, 0: 1}

#: New Jersey minimum wage after the April 1992 increase, and the variable `GAP` is
#: defined as the proportional increase in a store's starting wage needed to reach it.
NEW_JERSEY_MINIMUM = 5.05


def read_raw(path: Path | None = None) -> pd.DataFrame:
    """Parse ``public.dat`` into a typed frame, one row per store."""

    path = path or RAW
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run acquire() to download and unpack the archive"
        )

    records = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(
                {name: line[start - 1 : end].strip() for start, end, name in COLUMNS}
            )

    frame = pd.DataFrame(records)
    for name in frame.columns:
        frame[name] = pd.to_numeric(frame[name].replace(".", np.nan), errors="coerce")
    return frame


def fte(frame: pd.DataFrame, wave: str = "") -> pd.Series:
    """Full-time-equivalent employment: full-time + half of part-time + managers.

    The paper's unit throughout.  ``wave`` is ``""`` for the first interview and
    ``"2"`` for the second.
    """

    return (
        frame[f"empft{wave}"]
        + 0.5 * frame[f"emppt{wave}"]
        + frame[f"nmgrs{wave}"]
    )


def build() -> pd.DataFrame:
    """Return the analysis frame with FTE, changes and GAP, exactly as the paper defines them.

    The four rules implemented here are the ones a reader cannot infer from the codebook
    alone and must take from the paper's table notes:

    1. ``FTE = EMPFT + 0.5*EMPPT + NMGRS`` in each wave.
    2. **Permanently closed stores** (``STATUS2 == 3``, six of them) get wave-2
       employment **0** -- they are real job losses and dropping them would bias the
       estimate upward.
    3. **Temporarily closed stores** (``STATUS2`` in 2, 4, 5 -- four of them) get wave-2
       employment **missing** in the baseline specification, because their employment is
       unknown rather than zero.  Table 3 row 5 reports the variant that sets them to 0.
    4. ``GAP`` is the proportional wage increase needed to reach the new New Jersey
       minimum; it is **0 for every Pennsylvania store**, which is what makes the
       coefficient a difference-in-differences.
    """

    frame = read_raw()

    counts = frame["state"].value_counts()
    if len(frame) != CODEBOOK_COUNTS["observations"]:
        raise ValueError(
            f"parsed {len(frame)} rows, codebook says {CODEBOOK_COUNTS['observations']}; "
            "a column boundary is probably off by one"
        )
    if int(counts.get(1.0, 0)) != CODEBOOK_COUNTS["state_new_jersey"]:
        raise ValueError(
            f"New Jersey rows = {int(counts.get(1.0, 0))}, codebook says "
            f"{CODEBOOK_COUNTS['state_new_jersey']}"
        )
    observed_status = {int(k): int(v) for k, v in frame["status2"].value_counts().items()}
    if observed_status != STATUS2_COUNTS:
        raise ValueError(
            f"STATUS2 distribution {observed_status} does not match the codebook's "
            f"{STATUS2_COUNTS}"
        )

    frame["fte"] = fte(frame)
    frame["fte2_raw"] = fte(frame, "2")

    permanently_closed = frame["status2"] == 3
    temporarily_closed = frame["status2"].isin([2, 4, 5])

    # Baseline: permanent closures are zero employment, temporary closures are missing.
    frame["fte2"] = frame["fte2_raw"]
    frame.loc[temporarily_closed, "fte2"] = np.nan
    frame.loc[permanently_closed, "fte2"] = 0.0

    # Table 3 row 5 variant: temporary closures also set to zero.
    frame["fte2_zero_temp"] = frame["fte2"]
    frame.loc[temporarily_closed, "fte2_zero_temp"] = 0.0

    frame["d_fte"] = frame["fte2"] - frame["fte"]
    frame["d_fte_zero_temp"] = frame["fte2_zero_temp"] - frame["fte"]

    frame["nj"] = frame["state"]
    frame["gap"] = np.where(
        frame["nj"] == 1,
        np.maximum(NEW_JERSEY_MINIMUM - frame["wage_st"], 0.0) / frame["wage_st"],
        0.0,
    )

    # Chain dummies: the paper says "dummies for three of the chains", i.e. Burger King
    # is the omitted category.  Building them explicitly (rather than via C(chain))
    # makes the omitted chain visible instead of implicit.
    for level in (2, 3, 4):
        frame[f"chain{level}"] = (frame["chain"] == level).astype(float)

    return frame


def table4_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """The 357 stores that Table 4 runs on.

    The rule, recovered by matching the paper's reported N: employment and a starting
    wage in wave 1, wave-2 employment known, **and** a wave-2 starting wage -- except
    that the six permanently closed stores qualify without one, because their wave-2
    employment is defined as zero rather than observed.

    That exception is not cosmetic: requiring a wave-2 wage for everyone drops the six
    closures and gives N = 351 and a New Jersey coefficient of 2.28 instead of the
    published 2.33.  See ``REPORT.md`` for the arithmetic.
    """

    has_wage2_or_closed = frame["wage_st2"].notna() | (frame["status2"] == 3)
    mask = (
        frame["fte"].notna()
        & frame["fte2"].notna()
        & frame["wage_st"].notna()
        & has_wage2_or_closed
    )
    return frame.loc[mask].reset_index(drop=True)


def main() -> int:
    frame = build()
    sample = table4_sample(frame)
    print(f"parsed {len(frame)} stores; Table 4 sample {len(sample)}")
    print(f"d_fte in Table 4 sample: mean {sample.d_fte.mean():.3f} "
          f"sd {sample.d_fte.std():.3f}  (paper: -0.237, 8.825)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
