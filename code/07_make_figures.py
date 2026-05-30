"""The four main figures, drawn from the package's own tables.

Run scripts 01-05 first; this script reads the tables they produce (and the
analysis sample for the coefficient figure) and writes four clean PNGs:

  figures/figure1_production_function.png   standardized coefficients with 95%
                                            confidence intervals
  figures/figure2_factor_shares.png         share of explained variance by
                                            input across placement margins
  figures/figure3_dual_channel.png          Top-5 rate by connection tier,
                                            within idea tiers (a) and within
                                            off-the-shelf tiers (b)
  figures/figure4_apex_friction.png         ceiling (Top-5) vs floor
                                            (unmatched) by idea-quality
                                            percentile

The visual style is deliberately plain: a colourblind-safe palette, thin axes,
and small type, with the connection variable accented because it carries the
paper's central finding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common import (FIGURES, FIVE_INPUTS, INPUT_LABELS, TABLES,
                    collapse_rare_fields, fixed_effect_dummies,
                    load_analysis_sample, standardize)

# Colourblind-safe palette (Okabe-Ito); connections are accented (vermillion).
BLUE, SKY, GREEN, VERMILLION, GREY, BLACK = (
    "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#999999", "#222222")
ACCENT = VERMILLION

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def figure1_production_function() -> None:
    """Refit the compact specification to recover coefficients and standard
    errors, then plot them with 95% confidence intervals."""
    df = collapse_rare_fields(load_analysis_sample())
    df = standardize(df, FIVE_INPUTS)
    y = df["prestige"].astype(float)
    fe = fixed_effect_dummies(df).reset_index(drop=True)
    X = pd.concat([
        df[FIVE_INPUTS + ["team_size"]].astype(float).reset_index(drop=True),
        fe], axis=1)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y.reset_index(drop=True), Xc.astype(float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["author_cluster_id"].reset_index(drop=True)})

    colour = {"execution_quality": BLUE, "idea_quality": SKY,
              "offshelf_llm_score": GREY, "connection_index": ACCENT,
              "author_ability_index": GREEN}
    order = ["execution_quality", "idea_quality", "offshelf_llm_score",
             "connection_index", "author_ability_index"][::-1]

    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for i, var in enumerate(order):
        b, se = res.params[var], res.bse[var]
        ax.errorbar(b, i, xerr=1.96 * se, fmt="o", ms=5, color=colour[var],
                    ecolor=colour[var], elinewidth=1.3, capsize=2.5, zorder=3)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([INPUT_LABELS[v] for v in order])
    ax.set_xlabel("Association with prestige placement\n(per +1 SD; 95% CI)")
    ax.set_xlim(-0.03, 0.18)
    ax.margins(y=0.18)
    fig.savefig(FIGURES / "figure1_production_function.png")
    plt.close(fig)
    print("figure1_production_function.png")


def figure2_factor_shares() -> None:
    table = pd.read_csv(TABLES / "factor_shares_by_margin.csv")
    margins = ["Any publication", "Mid or better", "Top-field",
               "Prestige", "Top-5"]
    order = ["Execution", "Connections", "Idea quality",
             "Author ability", "LLM text score"]
    colour = {"Execution": BLUE, "Connections": ACCENT, "Idea quality": SKY,
              "Author ability": GREEN, "LLM text score": GREY}

    def share(margin: str, block: str) -> float:
        row = table[(table["placement_margin"] == margin)
                    & (table["input_block"] == block)]
        return float(row["share_of_explained_variance"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ypos = np.arange(len(margins))
    for yi, margin in zip(ypos, margins):
        left = 0.0
        for block in order:
            w = share(margin, block)
            ax.barh(yi, w, left=left, color=colour[block], edgecolor="white",
                    linewidth=0.6, height=0.66)
            if block == "Connections":
                ax.text(left + w / 2, yi, f"{w:.0f}", ha="center",
                        va="center", color="white", fontsize=7,
                        fontweight="bold")
            left += w
    ax.set_yticks(ypos)
    ax.set_yticklabels(margins)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of explained variance (%)")
    handles = [mpatches.Patch(color=colour[b], label=b) for b in order]
    ax.legend(handles=handles, ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.1)
    fig.savefig(FIGURES / "figure2_factor_shares.png")
    plt.close(fig)
    print("figure2_factor_shares.png")


def figure3_dual_channel() -> None:
    fav_idea = pd.read_csv(TABLES / "favoritism_by_idea_tier.csv")
    fav_off = pd.read_csv(TABLES / "favoritism_by_offshelf_tier.csv")
    conn_tiers = ["none", "low", "mid", "high"]
    conn_labels = ["None", "Low", "Mid", "High"]
    rate_cols = [f"{t}_top5_rate_pct" for t in conn_tiers]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(conn_tiers))

    # Panel a: within idea tiers (exceptional / strong / fair)
    idea_colour = {"exceptional": BLACK, "strong": BLUE, "fair": ACCENT}
    for tier, label in [("exceptional", "Exceptional"), ("strong", "Strong"),
                        ("fair", "Fair")]:
        row = fav_idea[fav_idea["idea_tier"] == tier].iloc[0]
        vals = [row[c] for c in rate_cols]
        axes[0].plot(x, vals, "-o", ms=4, lw=1.4, color=idea_colour[tier])
        axes[0].text(x[-1] + 0.08, vals[-1], label, color=idea_colour[tier],
                     fontsize=7, va="center")
    axes[0].set_title("Holding idea quality fixed\n(discipline-trained tier)",
                      fontsize=8)

    # Panel b: within off-the-shelf score tiers (high / mid / low)
    off_colour = {"high": BLACK, "mid": BLUE, "low": ACCENT}
    for tier, label in [("high", "High"), ("mid", "Mid"), ("low", "Low")]:
        row = fav_off[fav_off["offshelf_tier"] == tier].iloc[0]
        vals = [row[c] for c in rate_cols]
        axes[1].plot(x, vals, "-o", ms=4, lw=1.4, color=off_colour[tier])
        axes[1].text(x[-1] + 0.08, vals[-1], label, color=off_colour[tier],
                     fontsize=7, va="center")
    axes[1].set_title("Holding the off-the-shelf score fixed", fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(conn_labels)
        ax.set_xlabel("Connection tier")
        ax.set_xlim(-0.2, len(conn_tiers) + 0.5)
    axes[0].set_ylabel("Top-5 placement rate (%)")
    axes[0].text(-0.18, 1.04, "a", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold")
    axes[1].text(-0.06, 1.04, "b", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold")
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(FIGURES / "figure3_dual_channel.png")
    plt.close(fig)
    print("figure3_dual_channel.png")


def figure4_apex_friction() -> None:
    table = pd.read_csv(TABLES / "placement_by_idea_percentile.csv")
    # order from bottom half up to the top 1%
    order = ["bottom 50%", "top 50%", "top 25%", "top 10%", "top 5%", "top 1%"]
    table = table.set_index("idea_percentile_group").reindex(order)
    labels = ["Bottom\n50%", "Top\n50%", "Top\n25%", "Top\n10%",
              "Top\n5%", "Top\n1%"]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(x, table["top5_pct"].values, "-o", ms=4.5, lw=1.6, color=BLUE,
            label="Reached Top-5 (ceiling)")
    ax.plot(x, table["unmatched_pct"].values, "-s", ms=4.5, lw=1.6,
            color=ACCENT, label="Went unmatched (floor)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Idea-quality percentile (cumulative)")
    ax.set_ylabel("Placement rate (%)")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)
    fig.savefig(FIGURES / "figure4_apex_friction.png")
    plt.close(fig)
    print("figure4_apex_friction.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    missing = [t for t in ["factor_shares_by_margin.csv",
                           "favoritism_by_idea_tier.csv",
                           "favoritism_by_offshelf_tier.csv",
                           "placement_by_idea_percentile.csv"]
               if not (TABLES / t).exists()]
    if missing:
        raise SystemExit(
            f"missing tables {missing}; run scripts 03-05 first.")
    figure1_production_function()
    figure2_factor_shares()
    figure3_dual_channel()
    figure4_apex_friction()
    print(f"\nfour figures written to {FIGURES}")


if __name__ == "__main__":
    main()
