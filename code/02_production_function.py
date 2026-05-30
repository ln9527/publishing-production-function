"""The prestige production function: standardized OLS coefficients.

Outcome: prestige placement (1 if the paper appeared in a Top-5 or a leading
field journal, else 0).

Two specifications, both with team size, field fixed effects, issue-year fixed
effects, and standard errors clustered by author:

  Compact index specification
      prestige ~ idea_quality + offshelf_llm_score + execution_quality
               + connection_index + author_ability_index
      The connection and author-ability constructs each enter as a single
      composite index. Continuous inputs are standardized (per +1 SD); the
      coefficients are therefore directly comparable in size.

  Multi-indicator benchmark
      The same three text scores, but the connection and author-ability
      composites are replaced by their underlying indicators (coauthor-network
      position, institutional prestige, editor ties; prior output and
      experience). This shows the compact result is not an artifact of
      collapsing many indicators into one index.

Output:
  tables/production_function_coefficients.csv
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from common import (AUTHOR_ABILITY_BLOCK, BINARY_COLS, CONNECTION_BLOCK,
                    FIVE_INPUTS, INPUT_LABELS, TABLES, collapse_rare_fields,
                    fixed_effect_dummies, load_analysis_sample,
                    significance_stars, standardize)


def cluster_ols(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    """OLS with cluster-robust standard errors."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y.reset_index(drop=True).astype(float),
                  Xc.astype(float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters.reset_index(drop=True)})


def fit_specification(df: pd.DataFrame, predictors: list[str]):
    """prestige ~ predictors + team_size + field FE + issue-year FE."""
    y = df["prestige"].astype(float)
    fe = fixed_effect_dummies(df).reset_index(drop=True)
    X = pd.concat([
        df[predictors + ["team_size"]].astype(float).reset_index(drop=True),
        fe,
    ], axis=1)
    return cluster_ols(y, X, df["author_cluster_id"])


def coefficient_cell(res, var: str) -> str:
    b, se, p = res.params[var], res.bse[var], res.pvalues[var]
    return f"{b:+.4f}{significance_stars(p)} ({se:.4f})"


def main() -> None:
    df = load_analysis_sample()
    df = collapse_rare_fields(df)

    # Standardize all continuous inputs; leave binary indicators at 0/1.
    # team_size enters raw (it is a control, not a standardized input).
    std_cols = [c for c in (FIVE_INPUTS + CONNECTION_BLOCK
                            + AUTHOR_ABILITY_BLOCK)
                if c not in BINARY_COLS]
    df = standardize(df, std_cols)

    TABLES.mkdir(parents=True, exist_ok=True)

    compact = fit_specification(df, FIVE_INPUTS)
    benchmark = fit_specification(
        df, ["idea_quality", "offshelf_llm_score", "execution_quality"]
        + CONNECTION_BLOCK + AUTHOR_ABILITY_BLOCK)

    # Build the display table.
    rows = []
    for var in ["execution_quality", "idea_quality", "offshelf_llm_score",
                "connection_index", "author_ability_index"]:
        rows.append({
            "predictor": INPUT_LABELS[var],
            "compact_index_specification": coefficient_cell(compact, var),
            "multi_indicator_benchmark":
                coefficient_cell(benchmark, var)
                if var in benchmark.params.index else "(see indicators)",
        })
    # one informative connection indicator from the benchmark
    rows.append({
        "predictor": "Acknowledged editors (benchmark indicator)",
        "compact_index_specification": "(in connection index)",
        "multi_indicator_benchmark":
            coefficient_cell(benchmark, "acknowledged_editors"),
    })
    rows.append({
        "predictor": "Connection / ability indicators",
        "compact_index_specification": "single index each",
        "multi_indicator_benchmark": "all indicators included",
    })
    rows.append({
        "predictor": "Team size, field FE, issue-year FE",
        "compact_index_specification": "included",
        "multi_indicator_benchmark": "included",
    })
    rows.append({
        "predictor": "N",
        "compact_index_specification": str(int(compact.nobs)),
        "multi_indicator_benchmark": str(int(benchmark.nobs)),
    })
    rows.append({
        "predictor": "R-squared",
        "compact_index_specification": f"{compact.rsquared:.4f}",
        "multi_indicator_benchmark": f"{benchmark.rsquared:.4f}",
    })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "production_function_coefficients.csv", index=False)

    print(f"prestige production function (N = {int(compact.nobs):,}, "
          f"{df['author_cluster_id'].nunique():,} author clusters)")
    print(table.to_string(index=False))
    print("\nCompact-index standardized coefficients (per +1 SD):")
    for var in FIVE_INPUTS:
        print(f"  {INPUT_LABELS[var]:<36s} {compact.params[var]:+.4f}")


if __name__ == "__main__":
    main()
