"""Shared helpers for the reproduction scripts.

Every script reads ONLY from the package's own data/ directory and writes to
tables/ or figures/. This module centralises the data paths, the five-input
definitions, the input blocks used by the variance decomposition, and a few
small statistical utilities so the headline numbers are computed identically
everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- paths -----------------------------------------------------------------
PKG = Path(__file__).resolve().parent.parent
DATA = PKG / "data"
TABLES = PKG / "tables"
FIGURES = PKG / "figures"
ANALYSIS_SAMPLE = DATA / "analysis_sample.csv"
CALIBRATION_SAMPLE = DATA / "calibration_sample.csv"


# --- the five compact inputs (one scalar per construct) --------------------
FIVE_INPUTS = [
    "idea_quality",          # discipline-trained text score
    "offshelf_llm_score",    # off-the-shelf large language model text score
    "execution_quality",     # execution / methods quality
    "connection_index",      # composite index of ties to the field's gatekeepers
    "author_ability_index",  # composite index of author track record
]

INPUT_LABELS = {
    "idea_quality": "Idea quality (discipline-trained)",
    "offshelf_llm_score": "Off-the-shelf LLM text score",
    "execution_quality": "Execution quality",
    "connection_index": "Connection index",
    "author_ability_index": "Author ability index",
}


# --- multi-indicator blocks (for the variance decomposition + benchmark) ---
# Connections are NOT the author's own productivity: author productivity
# (prior output, experience) forms the ability block; the connection block is
# coauthor-network position, institutional prestige, and ties to editors.
AUTHOR_ABILITY_BLOCK = [
    "team_max_prior_publications", "team_mean_prior_publications",
    "team_max_prior_top5_publications", "team_sum_prior_top5_publications",
    "team_max_years_since_first_publication",
    "team_mean_years_since_first_publication",
    "author_prior_publications_total", "author_prior_publications_recent_5yr",
    "author_prior_top5_publications", "author_years_since_first_publication",
]
CONNECTION_BLOCK = [
    "coauthor_network_degree_max", "coauthor_network_degree_mean",
    "coauthor_network_centrality_max", "has_star_coauthor",
    "institution_prestige_best_tier", "institution_prestige_mean_tier",
    "n_authors_moved_to_better_institution",
    "author_ever_editor", "author_ever_top5_editor",
    "editor_tenure_years_max", "editor_tenure_years_sum",
    "has_editor_coauthor", "n_editor_coauthors",
    "n_advisors_who_are_top5_editors", "acknowledged_editors",
]

# Indicators left at 0/1 (never z-scored), so a coefficient reads as a level
# contrast and the decomposition treats them as raw binaries.
BINARY_COLS = {
    "has_star_coauthor", "author_ever_editor", "author_ever_top5_editor",
    "has_editor_coauthor",
}


def load_analysis_sample() -> pd.DataFrame:
    """The five-input analytical sample (complete cases on the five inputs)."""
    if not ANALYSIS_SAMPLE.exists():
        raise SystemExit(f"missing data/{ANALYSIS_SAMPLE.name} (run from the repo; check the shipped data files)")
    return pd.read_csv(ANALYSIS_SAMPLE)


def load_calibration_sample() -> pd.DataFrame:
    """The full scored cohort used for calibration and the percentile cuts."""
    if not CALIBRATION_SAMPLE.exists():
        raise SystemExit(f"missing data/{CALIBRATION_SAMPLE.name} (run from the repo; check the shipped data files)")
    return pd.read_csv(CALIBRATION_SAMPLE)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s - s.mean()


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy with the continuous `cols` z-scored; binary indicators
    are coerced to numeric but left at their 0/1 values."""
    out = df.copy()
    for c in cols:
        if c in BINARY_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = zscore(out[c])
    return out


def collapse_rare_fields(df: pd.DataFrame, col: str = "field_jel",
                         min_n: int = 30, other: str = "other") -> pd.DataFrame:
    """Fold field categories with fewer than `min_n` papers into one 'other'
    level, so rare fixed-effect categories do not cause separation."""
    out = df.copy()
    counts = out[col].value_counts()
    rare = set(counts[counts < min_n].index)
    if rare:
        out[col] = out[col].where(~out[col].isin(rare), other)
    return out


def fixed_effect_dummies(df: pd.DataFrame,
                         cols: tuple[str, ...] = ("field_jel", "issue_year")
                         ) -> pd.DataFrame:
    """One-hot the fixed-effect columns, dropping one level each (the
    intercept absorbs it)."""
    parts = [pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True,
                            dtype=float) for c in cols]
    return pd.concat(parts, axis=1)


def significance_stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
