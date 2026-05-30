"""Does the idea score line up with where papers actually land?

This is the measurement-validation exhibit. For each realized placement tier, we
report how the pre-publication idea evaluator rated those papers: the share
called exceptional, the share called strong or better, the share called limited,
and the mean idea score. If the score is meaningful, higher placement tiers
should carry higher idea ratings, monotonically.

They do. Papers that reached a Top-5 journal were rated exceptional 53.3% of the
time and strong-or-exceptional 94.1% of the time, with a mean idea score of
3.361; unmatched papers were rated exceptional only 23.6% of the time. The
gradient is monotonic in the score and steep at the top.

This runs on the full scored cohort (every paper that received an idea score),
since calibration is about the measurement, not the regression sample.

Output:
  tables/measurement_calibration.csv
"""
from __future__ import annotations

import pandas as pd

from common import TABLES, load_calibration_sample

# Placement tiers from lowest to highest, with display labels.
TIER_ORDER = [
    ("unmatched", "Unmatched"),
    ("lower", "Lower-tier journal"),
    ("mid", "Mid-tier journal"),
    ("top_field", "Leading field journal"),
    ("top5", "Top-5 journal"),
]


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in TIER_ORDER:
        g = df[df["placement_label"] == key]
        tier = g["ensemble_tier"]
        rows.append({
            "placement": label,
            "n": len(g),
            "rated_exceptional_pct":
                round(100 * (tier == "exceptional").mean(), 1),
            "rated_strong_or_exceptional_pct":
                round(100 * tier.isin(["exceptional", "strong"]).mean(), 1),
            "rated_limited_pct":
                round(100 * (tier == "limited").mean(), 1),
            "mean_idea_score": round(g["idea_quality_ensemble"].mean(), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_calibration_sample()
    TABLES.mkdir(parents=True, exist_ok=True)

    table = calibration_table(df)
    table.to_csv(TABLES / "measurement_calibration.csv", index=False)

    print(f"measurement calibration (N = {len(df):,} scored papers)")
    print(table.to_string(index=False))

    top5 = table[table["placement"] == "Top-5 journal"].iloc[0]
    print(f"\nTop-5 papers were rated exceptional {top5['rated_exceptional_pct']}% "
          f"and strong-or-exceptional {top5['rated_strong_or_exceptional_pct']}% "
          f"of the time (mean idea score {top5['mean_idea_score']}).")


if __name__ == "__main__":
    main()
