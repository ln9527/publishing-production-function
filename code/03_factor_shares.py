"""Factor shares: how the explained variance splits across the five inputs.

For each of five placement margins, we ask what share of the explained
variance (R-squared) each input block contributes. We use the Shapley value, so
the share is order-invariant: it does not depend on the order in which blocks
are entered into the regression. Each block's Shapley value is its average
marginal contribution to R-squared over all possible subsets of the other four
blocks.

The five blocks are:
  Execution quality      execution_quality
  Connections            coauthor-network position, institutional prestige,
                         and ties to editors (the multi-indicator block)
  Idea quality           idea_quality (discipline-trained text score)
  Author ability         prior output and experience (the multi-indicator block)
  LLM text score         offshelf_llm_score

The five placement margins (each a 0/1 outcome), from the broadest to the
narrowest:
  Any publication        placement at any journal
  Mid or better          placement at a mid-tier journal or higher
  Top-field              placement at a leading field journal (or above)
  Prestige               placement at a Top-5 OR a leading field journal
  Top-5                  placement at a Top-5 journal

Continuous blocks are standardized; the four 0/1 connection indicators are left
at 0/1. No fixed effects enter the decomposition, so the shares are a clean
read on the inputs' explanatory content.

Output:
  tables/factor_shares_by_margin.csv
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (AUTHOR_ABILITY_BLOCK, BINARY_COLS, CONNECTION_BLOCK,
                    FIVE_INPUTS, TABLES, collapse_rare_fields,
                    load_analysis_sample, standardize)

# Block name -> the columns it contains.
BLOCKS = {
    "Execution": ["execution_quality"],
    "Connections": CONNECTION_BLOCK,
    "Idea quality": ["idea_quality"],
    "Author ability": AUTHOR_ABILITY_BLOCK,
    "LLM text score": ["offshelf_llm_score"],
}

# Margins, from broadest to narrowest. Each maps to a 0/1 outcome series.
MARGINS = ["Any publication", "Mid or better", "Top-field",
           "Prestige", "Top-5"]


def margin_outcome(df: pd.DataFrame, margin: str) -> pd.Series:
    rung = df["placement_rung"].astype(int)
    return {
        "Any publication": (rung >= 1),
        "Mid or better": (rung >= 2),
        "Top-field": (rung == 3),
        "Prestige": (rung >= 3),
        "Top-5": (rung == 4),
    }[margin].astype(float).reset_index(drop=True)


def r_squared(y: pd.Series, df: pd.DataFrame, block_names: tuple[str, ...]) -> float:
    """R-squared of an OLS of y on the union of the named blocks' columns."""
    if not block_names:
        return 0.0
    cols: list[str] = []
    for b in block_names:
        cols.extend(BLOCKS[b])
    X = df[cols].astype(float).reset_index(drop=True)
    Xc = sm.add_constant(X, has_constant="add")
    return float(sm.OLS(y, Xc.astype(float)).fit().rsquared)


def shapley_shares(y: pd.Series, df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Order-invariant Shapley value of each block's contribution to R-squared."""
    names = list(BLOCKS)
    n = len(names)
    subsets = [s for k in range(n + 1) for s in combinations(names, k)]
    cache = {tuple(sorted(s)): r_squared(y, df, s) for s in subsets}
    full = cache[tuple(sorted(names))]

    values: dict[str, float] = {}
    for b in names:
        contribution = 0.0
        for s in subsets:
            if b in s:
                continue
            with_b = tuple(sorted(list(s) + [b]))
            without_b = tuple(sorted(s))
            weight = (factorial(len(s)) * factorial(n - len(s) - 1)
                      ) / factorial(n)
            contribution += weight * (cache[with_b] - cache[without_b])
        values[b] = contribution
    return values, full


def main() -> None:
    df = load_analysis_sample()
    df = collapse_rare_fields(df)

    all_block_cols = [c for cols in BLOCKS.values() for c in cols]
    std_cols = [c for c in all_block_cols if c not in BINARY_COLS]
    df = standardize(df, std_cols)

    TABLES.mkdir(parents=True, exist_ok=True)

    rows = []
    for margin in MARGINS:
        y = margin_outcome(df, margin)
        values, full = shapley_shares(y, df)
        for block in BLOCKS:
            share = values[block] / full if full else float("nan")
            rows.append({
                "placement_margin": margin,
                "input_block": block,
                "shapley_r_squared": round(values[block], 4),
                "share_of_explained_variance": round(100 * share, 1),
                "block_r_squared": round(full, 4),
            })

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "factor_shares_by_margin.csv", index=False)

    # Pretty wide view for the console: shares (%) by block x margin.
    wide = table.pivot(index="input_block", columns="placement_margin",
                       values="share_of_explained_variance")
    wide = wide.reindex(index=list(BLOCKS), columns=MARGINS)
    print(f"factor shares (% of explained variance), N = {len(df):,} papers")
    print(wide.to_string())
    conn = wide.loc["Connections"]
    print(f"\nConnection share rises from {conn['Any publication']:.1f}% at "
          f"any publication to {conn['Top-5']:.1f}% at Top-5.")


if __name__ == "__main__":
    main()
