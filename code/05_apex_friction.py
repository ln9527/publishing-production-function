"""Apex friction: a strong idea is not enough to clear the top of the market.

We rank every scored working paper by its idea-quality score and, for nested
top-percentile groups, tabulate where those papers ended up. Two numbers tell
the story:

  the ceiling   the share that reached a Top-5 journal rises steeply with the
                idea score (a roughly tenfold spread from the bottom half to
                the top 1%); and
  the floor     the share that went unmatched (never placed in any journal)
                falls only gently (a roughly twofold spread).

So idea quality lifts the ceiling far more than it lifts the floor: even the
very best ideas go unmatched a sizeable fraction of the time. Among the top 1%
of papers by the idea score, the Top-5 rate is about 57% but the unmatched rate
is still about 18%.

This runs on the full scored cohort (every paper with an idea score), so the
percentile cuts are taken over the entire scored set, not the smaller
five-input regression sample. The idea score used for ranking is the ensemble
score (the geometric mean of two independently trained discipline models).

Output:
  tables/placement_by_idea_percentile.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import TABLES, load_calibration_sample

# (label, lower-percentile-cut). A None means "below the median".
CUTS = [
    ("top 1%", 99),
    ("top 5%", 95),
    ("top 10%", 90),
    ("top 25%", 75),
    ("top 50%", 50),
    ("bottom 50%", None),
]


def percentile_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["idea_quality_ensemble", "placement_rung"]).copy()
    score = d["idea_quality_ensemble"].to_numpy()
    rung = d["placement_rung"].astype(int)
    median = np.percentile(score, 50)

    rows = []
    for label, cut in CUTS:
        if cut is None:
            sub = d[d["idea_quality_ensemble"] < median]
        else:
            threshold = np.percentile(score, cut)
            sub = d[d["idea_quality_ensemble"] >= threshold]
        r = sub["placement_rung"].astype(int)
        n = len(sub)
        rows.append({
            "idea_percentile_group": label,
            "n": n,
            "top5_pct": round(100 * (r == 4).mean(), 1),
            "top_field_pct": round(100 * (r == 3).mean(), 1),
            "mid_pct": round(100 * (r == 2).mean(), 1),
            "lower_pct": round(100 * (r == 1).mean(), 1),
            "unmatched_pct": round(100 * (r == 0).mean(), 1),
            "mean_placement_rung": round(r.mean(), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_calibration_sample()
    TABLES.mkdir(parents=True, exist_ok=True)

    table = percentile_table(df)
    table.to_csv(TABLES / "placement_by_idea_percentile.csv", index=False)

    print(f"placement by idea-quality percentile "
          f"(N = {len(df):,} scored papers)")
    print(table.to_string(index=False))

    top1 = table[table["idea_percentile_group"] == "top 1%"].iloc[0]
    bottom = table[table["idea_percentile_group"] == "bottom 50%"].iloc[0]
    print(f"\nThe top 1% by the idea score reach Top-5 {top1['top5_pct']}% of "
          f"the time, yet still go unmatched {top1['unmatched_pct']}% of the "
          f"time.")
    print(f"Top-5 spread top-1% vs bottom-half: "
          f"{top1['top5_pct'] / bottom['top5_pct']:.1f}x (ceiling). "
          f"Unmatched spread: {bottom['unmatched_pct'] / top1['unmatched_pct']:.1f}x "
          f"(floor).")


if __name__ == "__main__":
    main()
