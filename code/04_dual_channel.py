"""The two channels through which connections operate: capture and favoritism.

Connections can raise a paper's placement two ways:

  Capture     better-connected papers are also genuinely higher quality, so
              part of their placement advantage is earned. We measure this as
              the difference in mean input scores across connection tiers.

  Favoritism  even holding quality fixed, better-connected papers place
              higher. We measure this by tabulating the Top-5 placement rate
              within idea tiers, across connection tiers.

We then test whether the two channels are additive (no interaction) by
regressing Top-5 placement on the connection index, the idea score, and their
product.

Connection tiers. A paper has no connection tier ("none") if neither author
has ever been an editor and no editor is acknowledged. Among the connected
papers, we split the connection index into within-sample terciles
(low / mid / high).

Idea tiers. The discipline-trained evaluator assigns each paper to one of four
tiers (exceptional / strong / fair / limited). We use the more conservative
(more limited) of two independently trained models, so a paper is only called
high-quality if both models agree it is strong.

Robustness with the untrained score. Because the discipline-trained score is
calibrated on placement, we repeat the favoritism table holding the
off-the-shelf large-language-model score fixed (within-sample terciles). That
score never saw the placement outcome, so it provides a circularity-free check.

Outputs:
  tables/capture.csv                       mean inputs by connection tier
  tables/favoritism_by_idea_tier.csv       Top-5 rate, idea tier x connection
  tables/favoritism_by_offshelf_tier.csv   Top-5 rate, off-the-shelf x connection
  tables/interaction_test.csv              connection x idea interaction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (TABLES, fixed_effect_dummies, load_analysis_sample,
                    zscore)

CONN_TIERS = ["none", "low", "mid", "high"]
IDEA_TIERS = ["exceptional", "strong", "fair", "limited"]
OFFSHELF_TIERS = ["low", "mid", "high"]


def offshelf_terciles(df: pd.DataFrame) -> pd.Series:
    """Within-sample terciles of the off-the-shelf LLM score."""
    s = df["offshelf_llm_score"].astype(float)
    lo, hi = np.percentile(s, [100 / 3, 200 / 3])
    return pd.Series(np.where(s <= lo, "low",
                     np.where(s <= hi, "mid", "high")), index=df.index)


def capture_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean input scores and outcome rates by connection tier."""
    rows = []
    for tier in CONN_TIERS:
        g = df[df["connection_tier"] == tier]
        rows.append({
            "connection_tier": tier,
            "n": len(g),
            "mean_idea_quality": round(g["idea_quality"].mean(), 3),
            "mean_offshelf_llm_score": round(g["offshelf_llm_score"].mean(), 3),
            "mean_execution_quality": round(g["execution_quality"].mean(), 3),
            "top5_rate_pct": round(100 * g["reached_top5"].mean(), 1),
            "prestige_rate_pct": round(100 * g["prestige"].mean(), 1),
        })
    return pd.DataFrame(rows)


def favoritism_table(df: pd.DataFrame, row_col: str, row_levels: list[str]
                     ) -> pd.DataFrame:
    """Within each row level (an idea tier or off-the-shelf tier), the Top-5
    placement rate across connection tiers."""
    rows = []
    for level in row_levels:
        sub = df[df[row_col] == level]
        rec = {row_col: level}
        for ct in CONN_TIERS:
            cell = sub[sub["connection_tier"] == ct]
            rec[f"{ct}_top5_rate_pct"] = (round(100 * cell["reached_top5"].mean(), 1)
                                          if len(cell) else np.nan)
            rec[f"{ct}_n"] = len(cell)
        none = sub[sub["connection_tier"] == "none"]["reached_top5"].mean()
        high = sub[sub["connection_tier"] == "high"]["reached_top5"].mean()
        rec["high_over_none_ratio"] = (round(high / none, 1)
                                       if none and not np.isnan(none) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def interaction_test(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether connections and ideas enter additively or compound.

    For each of two outcomes (prestige, Top-5), we fit the full-input model
    with and without a connection-by-idea interaction:

        outcome ~ idea_quality_ensemble + connection_index
                + execution_quality + offshelf_llm_score [+ interaction]
                + field FE + issue-year FE

    All four continuous inputs are standardized; the interaction is the product
    of the standardized idea and connection scores. Standard errors are
    clustered by author. A null interaction term means the capture and
    favoritism channels are additive: connections add the same placement boost
    regardless of idea quality.
    """
    d = df.copy()
    cont = ["idea_quality_ensemble", "connection_index",
            "execution_quality", "offshelf_llm_score"]
    for v in cont:
        d[f"{v}_z"] = zscore(d[v])
    d["connection_x_idea"] = (d["idea_quality_ensemble_z"]
                              * d["connection_index_z"])
    fe = fixed_effect_dummies(d).reset_index(drop=True)

    main_cols = [f"{v}_z" for v in cont]
    rows = []
    for outcome, outcome_label in [("prestige", "Prestige"),
                                   ("reached_top5", "Top-5")]:
        y = d[outcome].astype(float).reset_index(drop=True)
        for spec, cols in [("main effects only", main_cols),
                           ("with connection x idea",
                            main_cols + ["connection_x_idea"])]:
            X = pd.concat([
                d[cols].astype(float).reset_index(drop=True), fe], axis=1)
            Xc = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, Xc.astype(float)).fit(
                cov_type="cluster",
                cov_kwds={"groups":
                          d["author_cluster_id"].reset_index(drop=True)})
            if "connection_x_idea" not in cols:
                continue
            b = res.params["connection_x_idea"]
            se = res.bse["connection_x_idea"]
            p = res.pvalues["connection_x_idea"]
            rows.append({
                "outcome": outcome_label,
                "term": "connection x idea interaction",
                "coefficient": round(b, 4),
                "std_error": round(se, 4),
                "p_value": round(p, 4),
                "significant_at_5pct": "yes" if p < 0.05 else "no (additive)",
                "n": int(res.nobs),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_analysis_sample()
    df["offshelf_tier"] = offshelf_terciles(df)

    TABLES.mkdir(parents=True, exist_ok=True)

    capture = capture_table(df)
    fav_idea = favoritism_table(df, "idea_tier", IDEA_TIERS)
    fav_offshelf = favoritism_table(df, "offshelf_tier", OFFSHELF_TIERS)
    interaction = interaction_test(df)

    capture.to_csv(TABLES / "capture.csv", index=False)
    fav_idea.to_csv(TABLES / "favoritism_by_idea_tier.csv", index=False)
    fav_offshelf.to_csv(TABLES / "favoritism_by_offshelf_tier.csv", index=False)
    interaction.to_csv(TABLES / "interaction_test.csv", index=False)

    print(f"dual-channel decomposition (N = {len(df):,} papers)\n")
    print("CAPTURE: mean inputs rise with connection tier")
    print(capture.to_string(index=False))

    print("\nFAVORITISM: Top-5 rate by idea tier x connection tier")
    print(fav_idea[["idea_tier", "none_top5_rate_pct", "low_top5_rate_pct",
                    "mid_top5_rate_pct", "high_top5_rate_pct",
                    "high_over_none_ratio"]].to_string(index=False))
    fair = fav_idea[fav_idea["idea_tier"] == "fair"].iloc[0]
    print(f"  Within fair-rated papers: Top-5 rate rises "
          f"{fair['none_top5_rate_pct']}% -> {fair['high_top5_rate_pct']}% "
          f"({fair['high_over_none_ratio']}x).")

    print("\nFAVORITISM (untrained off-the-shelf score held fixed)")
    print(fav_offshelf[["offshelf_tier", "none_top5_rate_pct",
                        "low_top5_rate_pct", "mid_top5_rate_pct",
                        "high_top5_rate_pct"]].to_string(index=False))
    low = fav_offshelf[fav_offshelf["offshelf_tier"] == "low"].iloc[0]
    print(f"  Within the low off-the-shelf tier: Top-5 rate rises "
          f"{low['none_top5_rate_pct']}% -> {low['high_top5_rate_pct']}%.")

    print("\nINTERACTION TEST: is the connection boost the same at every "
          "idea level?")
    print(interaction.to_string(index=False))
    print("  A non-significant interaction means the channels are additive.")


if __name__ == "__main__":
    main()
