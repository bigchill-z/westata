"""Build the datasets the documentation's comparison tables are generated from.

Three datasets, chosen so that the comparison spans the library rather than showing one happy path:

    auto.dta     Stata's own bundled dataset, 74 observations. Used for every command that needs
                 real-world data -- regression, IV, quantile, robust, systems, discrete choice.
                 Committed at tests/data/auto.dta, so the numbers replay without a licence.

    panel        A 200-firm x 8-year panel with an unobserved effect, for xtreg/xtlogit/xtpoisson
                 and friends. Synthetic so the generating process is known and the document can
                 state the true parameter.

    sim          A 600-row cross-section with correlated errors, used where a command needs a
                 structure auto does not have: endogeneity, treatment effects, selection, count
                 overdispersion.

Every generator is seeded and the seeds are recorded, so a reader can reproduce any table.
The data is written to docs/data/ as CSV as well, so the document does not depend on this script
having been run -- but the script is committed so it can be.

The point of separating auto from the synthetic ones: `auto` is REAL data, and a comparison on real
data is worth more than one on a generator the library's author also wrote.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(r"D:\myCodeProject\statapy")
OUT = REPO / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

AUTO = REPO / "tests" / "data" / "auto.dta"


def build_panel(seed: int = 20261024, n_firm: int = 200, n_year: int = 8) -> pd.DataFrame:
    """200 firms x 8 years with a firm effect correlated with the regressor.

    The correlation is deliberate: it is what makes fixed effects differ from random effects, so the
    document can show both and explain why the estimates move.
    """

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firm), n_year)
    year = np.tile(np.arange(1, n_year + 1), n_firm)

    # Firm effect and regressor share a component -> RE is inconsistent, FE is not.
    alpha = rng.normal(size=n_firm) * 1.0
    shared = rng.normal(size=n_firm) * 0.8
    x1 = shared[firm] + rng.normal(size=firm.size) * 0.6
    x2 = rng.normal(size=firm.size)
    year_effect = 0.2 * year

    y = 1.0 + 1.5 * x1 - 0.8 * x2 + alpha[firm] + year_effect + rng.normal(size=firm.size) * 0.7

    # A binary and a count outcome on the same index, for xtlogit / xtpoisson.
    latent = 0.5 + 1.0 * x1 - 0.6 * x2 + alpha[firm] * 0.5
    binary = (latent + rng.logistic(size=firm.size) > 0).astype(float)
    counts = rng.poisson(np.exp(0.3 + 0.5 * x1 - 0.3 * x2 + alpha[firm] * 0.3))

    return pd.DataFrame(
        {
            "firm": firm,
            "year": year,
            "y": y,
            "x1": x1,
            "x2": x2,
            "binary": binary,
            "count": counts.astype(float),
        }
    )


def build_sim(seed: int = 7, n: int = 600) -> pd.DataFrame:
    """Cross-section with endogeneity, a treatment effect and a selection mechanism.

    Three separate structures, because three separate command families need them:

        endog      z1, z2 instruments; x_endog correlated with the error -> OLS biased, 2SLS not.
        treat      binary treatment assigned by a probit -> etregress / teffects.
        select     outcome observed only when a latent index is positive -> heckman.
        count      overdispersed counts -> nbreg.
    """

    rng = np.random.default_rng(seed)

    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    w = rng.normal(size=n)

    e = rng.normal(size=n)
    # Endogenous regressor: built from the same shock as the outcome error.
    x_endog = 0.7 * z1 + 0.3 * z2 + 0.6 * e + rng.normal(size=n) * 0.5
    y_iv = 1.0 + 2.0 * x_endog + 0.5 * w + e

    # Treatment: assigned by a probit on z1, z2; outcome effect 1.5.
    u = rng.normal(size=n)
    treat_index = 0.4 + 0.8 * z1 - 0.5 * z2 + u
    treated = (treat_index > 0).astype(float)
    y_treat = 1.0 + 0.9 * w + 1.5 * treated + u * 0.6 + rng.normal(size=n)

    # Selection: outcome exists only when the index is positive; correlates with the error.
    v = rng.normal(size=n)
    sel_index = 0.5 + 0.8 * z1 - 0.5 * z2 + v
    observed = sel_index > 0
    y_sel = 1.0 + 1.2 * w + 0.7 * v + rng.normal(size=n)
    y_selected = np.where(observed, y_sel, np.nan)

    # Counts: negative binomial (overdispersed), not Poisson.
    mu = np.exp(0.5 + 0.6 * w)
    shape = 1.8
    counts = rng.negative_binomial(shape, shape / (shape + mu))

    return pd.DataFrame(
        {
            "y_iv": y_iv,
            "x_endog": x_endog,
            "w": w,
            "z1": z1,
            "z2": z2,
            "y_treat": y_treat,
            "treated": treated,
            "treat_index": treat_index,
            "y_sel": y_selected,
            "y_sel_full": y_sel,
            "observed": observed.astype(float),
            "sel_index": sel_index,
            "count": counts.astype(float),
        }
    )


def main() -> None:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    auto.to_csv(OUT / "auto.csv", index=False)
    print(f"auto      : {len(auto)} rows, {len(auto.columns)} cols -> docs/data/auto.csv")

    panel = build_panel()
    panel.to_csv(OUT / "panel.csv", index=False)
    print(f"panel     : {len(panel)} rows ({panel.firm.nunique()} firms x {panel.year.nunique()} years) -> docs/data/panel.csv")

    sim = build_sim()
    sim.to_csv(OUT / "sim.csv", index=False)
    print(f"sim       : {len(sim)} rows -> docs/data/sim.csv")

    print()
    print("seeds: panel=20261024, sim=7")
    print()
    print("auto columns :", list(auto.columns))
    print("panel columns:", list(panel.columns))
    print("sim columns  :", list(sim.columns))


if __name__ == "__main__":
    main()
