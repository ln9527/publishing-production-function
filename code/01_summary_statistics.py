"""Summary statistics and the correlation matrix among the five inputs.

Outputs:
  tables/summary_statistics.csv     means, SDs, and ranges for the five inputs,
                                    team size, and the placement outcome.
  tables/input_correlation_matrix.csv  pairwise correlations among the five
                                    inputs (in raw units).

The five inputs are: idea quality (a discipline-trained text score), an
off-the-shelf large-language-model text score, execution quality, a connection
index, and an author-ability index. Run on the five-input analytical sample.
"""
from __future__ import annotations

import pandas as pd

from common import FIVE_INPUTS, INPUT_LABELS, TABLES, load_analysis_sample


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / SD / min / median / max for the five inputs plus team size and
    the placement rung."""
    rows = []
    variables = {**INPUT_LABELS,
                 "team_size": "Team size",
                 "placement_rung": "Placement rung (0-4)"}
    for col, label in variables.items():
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": label,
            "n": int(s.notna().sum()),
            "mean": round(s.mean(), 3),
            "sd": round(s.std(), 3),
            "min": round(s.min(), 3),
            "median": round(s.median(), 3),
            "max": round(s.max(), 3),
        })
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[FIVE_INPUTS].apply(pd.to_numeric, errors="coerce")
    sub.columns = [INPUT_LABELS[c] for c in FIVE_INPUTS]
    corr = sub.corr().round(3)
    corr.insert(0, "input", corr.index)
    return corr.reset_index(drop=True)


def main() -> None:
    df = load_analysis_sample()
    TABLES.mkdir(parents=True, exist_ok=True)

    summ = summary_table(df)
    corr = correlation_table(df)

    summ.to_csv(TABLES / "summary_statistics.csv", index=False)
    corr.to_csv(TABLES / "input_correlation_matrix.csv", index=False)

    print(f"summary statistics (N = {len(df):,} papers, "
          f"{df['author_cluster_id'].nunique():,} author clusters)")
    print(summ.to_string(index=False))
    print("\ncorrelation matrix among the five inputs")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
