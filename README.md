# Merit or networks? What decides where research is published

Replication package for the paper of the same title (Ning Li).

This package reproduces every table and figure in the paper from de-identified
data. A reader can clone the repository, install a handful of standard Python
packages, run one command, and regenerate the headline results.

## What the paper does

Economics working papers are eventually published, and where they land in the
journal hierarchy shapes careers. Two stories compete: papers rise on the merit
of the research, or they rise on the authors' connections. Telling them apart
has always been hard, because no one had a measure of a paper's quality that was
fixed *before* publication and independent of who wrote it.

This paper uses a name-blind, pre-publication evaluator that scores a paper's
idea from its text alone. With that score in hand, it estimates a five-input
production function for journal placement on 5,848 papers with complete inputs,
drawn from a scored cohort of 6,208 NBER economics working papers (2010-2015). The five inputs are:

1. **Idea quality** — a discipline-trained text score.
2. **Off-the-shelf LLM text score** — the same idea, rated by a general
   language model that never saw the publication outcome (a circularity-free
   check).
3. **Execution quality** — how well the research is carried out.
4. **Connection index** — the authors' ties to the field's gatekeepers.
5. **Author ability index** — the authors' prior track record.

### Headline findings

- **The idea score tracks where papers land.** Papers that reached a Top-5
  journal were rated exceptional 53.3% of the time and strong-or-exceptional
  94.1% of the time, versus 23.6% exceptional among unmatched papers — a
  monotonic gradient that validates the measure.
- **Execution is the largest input.** It accounts for 37.3% of the explained
  variance in prestige placement and sets a meritocratic floor.
- **Connections concentrate at the very top.** Their share of explained variance
  rises from 15.3% at the any-publication margin to 38.8% at the Top-5 margin.
- **Connections act through two additive channels** — capture (better-connected
  papers are also genuinely better) and favoritism (even at equal quality, the
  connected place higher). Within equally-rated "fair" papers, the Top-5 rate
  rises from 1.9% with no connections to 14.1% at the top connection tier
  (about 7x), and the same gradient appears when the untrained off-the-shelf
  score is held fixed.
- **A strong idea is not enough at the apex.** Among the top 1% of papers by the
  idea score, 57.1% reach a Top-5 journal, yet 17.5% go unmatched entirely.

## Repository layout

```
data/
  analysis_sample.csv          5,848 papers, complete on the five inputs
  calibration_sample.csv       6,208 scored papers (full cohort)
  DATA_DICTIONARY.md           every column described in plain language
code/
  common.py                    shared definitions and helpers
  01_summary_statistics.py     summary stats + input correlation matrix
  02_production_function.py    standardized OLS coefficients
  03_factor_shares.py          variance decomposition across placement margins
  04_dual_channel.py           capture + favoritism + the additivity test
  05_apex_friction.py          placement by idea-quality percentile
  06_measurement_calibration.py idea score vs realized placement
  07_make_figures.py            the four main figures
  run_all.py                   runs every step in order
tables/                        generated CSV tables
figures/                       generated PNG figures
requirements.txt               Python package versions
```

## How to reproduce

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
cd code
python3 run_all.py
```

(Use `python3`, or `python` if that name points to your Python 3.10+ install.)

`run_all.py` runs the seven scripts in order and stops if any step fails. Scripts
01-06 each run on their own (for example `python3 02_production_function.py`);
`07_make_figures.py` reads the tables the earlier steps produce, so run those first
(or just use `run_all.py`). Every script reads only from `data/` and writes to
`tables/` or `figures/`.

## What each script produces

| Script | Output |
|---|---|
| `01_summary_statistics.py` | `tables/summary_statistics.csv`, `tables/input_correlation_matrix.csv` |
| `02_production_function.py` | `tables/production_function_coefficients.csv` |
| `03_factor_shares.py` | `tables/factor_shares_by_margin.csv` |
| `04_dual_channel.py` | `tables/capture.csv`, `tables/favoritism_by_idea_tier.csv`, `tables/favoritism_by_offshelf_tier.csv`, `tables/interaction_test.csv` |
| `05_apex_friction.py` | `tables/placement_by_idea_percentile.csv` |
| `06_measurement_calibration.py` | `tables/measurement_calibration.csv` |
| `07_make_figures.py` | `figures/figure1_production_function.png` ... `figure4_apex_friction.png` |

## Methods, briefly

The headline outcome is **prestige placement**: 1 if the working paper was
eventually published in a Top-5 or a leading field journal, else 0.

- **Production function.** Prestige is regressed on the five inputs, plus team
  size, field fixed effects, and issue-year fixed effects. Continuous inputs are
  standardized, so coefficients are comparable per one-standard-deviation
  change. Standard errors are clustered by author. We report a compact
  specification (one index per construct) and a multi-indicator benchmark that
  replaces the connection and ability indices with their underlying indicators.
- **Factor shares.** Each input's contribution to the explained variance is the
  order-invariant Shapley value, computed over all subsets of the other inputs,
  for each of five placement margins.
- **The two channels.** Capture is read off the mean input scores across
  connection tiers; favoritism is the Top-5 rate within idea tiers across
  connection tiers. Adding a connection-by-idea interaction to the full model
  leaves it statistically insignificant, so the channels are additive.
- **Apex friction.** Papers are ranked by their idea score over the full scored
  cohort, and placement outcomes are tabulated for nested top-percentile groups.

A note on language. The trained evaluator is calibrated on the journal tier,
so its score reflects what the literature treats as high quality. Read the idea
measure as "text-legible quality," and read all estimates as conditional
associations among the observed inputs, not structural causal effects.

## Data and its provenance

Every row in both data files is keyed by the public NBER working-paper id (for
example `w15630`); see `data/DATA_DICTIONARY.md` for the full column list.

The sample is NBER economics working papers issued 2010-2015. The derived
measures were built from public sources, at a high level, as follows.

- **Idea quality** and **off-the-shelf LLM score** are text scores: each working
  paper's text is read by an evaluator that never sees who wrote it or where it
  was published. The idea-quality score comes from a discipline-trained
  evaluator; the off-the-shelf score comes from a general language model and is
  included as a circularity-free check because it was never exposed to the
  placement outcome.
- **Execution quality** is scored from the same text (identification, methods,
  robustness, data, writing).
- The **connection index** and **author ability index** are composites of
  numeric features describing each team's coauthor-network position,
  institutional prestige, editorial ties, and prior publishing record. Every
  such feature uses only information dated *before* the working paper's issue
  date, so no post-issue information leaks into a pre-publication predictor.
- The **placement outcome** records where each working paper was eventually
  published in the journal hierarchy, observed years after issue.

The public upstream sources behind these measures are the National Bureau of
Economic Research (working-paper texts and issue dates), OpenAlex and Crossref
(publication records, citations, and authorship metadata), Semantic Scholar
(citation and reference data), and the published editorial boards of economics
journals (editorial ties). Each retains its own terms of use.

Author and editor identities are withheld for privacy: the public files carry no
author names, author identifiers, paper titles, editor names, or journal
identities. Because every row is keyed by the public NBER working-paper id and
all upstream sources above are public, a reader can rebuild the identifying
layer independently from those sources and re-derive the features.

## Privacy

The public data contains no author names, author identifiers, paper titles,
editor names, or journal identities. Each row is keyed by the public NBER
working-paper id. The primary author appears only as an integer
`author_cluster_id` with no link to any name or external identifier.

## Citation

If you use this package, please cite the paper and the evaluator it builds on.
Both are working papers; venue and year are provisional.

```bibtex
@unpublished{li_merit_networks,
  author = {Li, Ning},
  title  = {Merit or networks? What decides where research is published},
  note   = {Working paper},
  year   = {2026}
}

@unpublished{idea_quality_evaluator,
  author = {Gong, Zhen and Li, Ning and Zhou, Haoran},
  title  = {A discipline-trained evaluator of economics research ideas},
  note   = {Working paper},
  year   = {2026}
}
```

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The data are provided for replication; the public upstream sources from which
they are derived retain their own terms of use. A short data-availability
statement is in `DATA_AVAILABILITY.md`.

## Contact

Questions about the package or the paper may be directed to the corresponding
author.
