# Data dictionary

Two data files ship with this package. Both are de-identified: every row is
keyed by the public NBER working-paper identifier only. No author names, author
identifiers, paper titles, editor names, or journal identities appear anywhere.
The only person-level field is `author_cluster_id`, a dense integer code with no
link to any name or external identifier, provided so that standard errors can be
clustered by author.

The sample is NBER economics working papers issued 2010-2015. Working papers
are scored by a name-blind, pre-publication evaluator that reads only the
paper's text, then the realized journal placement is observed years later.

---

## `analysis_sample.csv`

One row per working paper in the five-input analytical sample: 5,848 papers
(3,067 author clusters) with complete data on all five inputs. This is the
sample used for the production function, the variance decomposition, and the
two-channel analysis.

### Identifiers and design fields

| column | type | description |
|---|---|---|
| `paper_id` | text | Public NBER working-paper id (for example `w15630`). |
| `issue_year` | integer | Calendar year the working paper was issued (2010-2015). |
| `field_jel` | text | Primary field, given by the first letter of the paper's JEL classification (for example `D`, `E`, `G`). Used as a field fixed effect. |
| `team_size` | integer | Number of authors on the paper. |
| `author_cluster_id` | integer | De-identified code for the paper's primary author. Papers by the same primary author share a code. Used to cluster standard errors. No link to any name or external id. |

### The five inputs (one scalar per construct)

| column | type | units | description |
|---|---|---|---|
| `idea_quality` | numeric | score, ~1-4 | Predicted idea quality from a discipline-trained text evaluator. Higher is better. |
| `offshelf_llm_score` | numeric | score, ~1-4 | Idea-quality rating from a general off-the-shelf large language model reading the same text. Provided because it never saw the placement outcome, so it serves as a circularity-free check on the trained score. |
| `execution_quality` | numeric | score, 1-5 | Quality of the paper's execution (identification, methods, robustness, data, writing), scored from the text. Higher is better. |
| `connection_index` | numeric | standardized index | Composite index of the authors' ties to the field's gatekeepers (coauthor-network position, institutional prestige, and ties to editors). Standardized to mean 0. Higher means more connected. |
| `author_ability_index` | numeric | standardized index | Composite index of the authors' prior track record (prior output and experience). Standardized to mean 0. Higher means a stronger track record. |
| `idea_quality_ensemble` | numeric | score, ~1-4 | Ensemble idea-quality score (geometric mean of two independently trained discipline models). Used as the idea regressor in the connection-by-idea interaction test, and to form the percentile cuts in the apex-friction analysis. |

### Underlying indicators (the multi-indicator blocks)

These numeric features are the raw indicators that feed the connection index and
the author-ability index. They let the package reproduce the variance
decomposition and the multi-indicator benchmark regression. All are derived
counts, tier codes, network statistics, or 0/1 flags; none identifies a person.
Every feature uses only information dated before the working paper's issue date.

Author-ability block (prior output and experience):

| column | description |
|---|---|
| `team_max_prior_publications` | Most prior publications held by any author on the team. |
| `team_mean_prior_publications` | Mean prior publications across the team. |
| `team_max_prior_top5_publications` | Most prior Top-5-journal publications by any author. |
| `team_sum_prior_top5_publications` | Total prior Top-5 publications across the team. |
| `team_max_years_since_first_publication` | Longest publishing career on the team. |
| `team_mean_years_since_first_publication` | Mean career length across the team. |
| `author_prior_publications_total` | Primary author's total prior publications. |
| `author_prior_publications_recent_5yr` | Primary author's prior publications in the five years before issue. |
| `author_prior_top5_publications` | Primary author's prior Top-5 publications. |
| `author_years_since_first_publication` | Primary author's career length at issue. |

Connection block (network position, prestige, editor ties):

| column | description |
|---|---|
| `coauthor_network_degree_max` | Largest coauthor-network degree on the team. |
| `coauthor_network_degree_mean` | Mean coauthor-network degree across the team. |
| `coauthor_network_centrality_max` | Largest coauthor-network eigenvector centrality. |
| `has_star_coauthor` | 1 if any author has a highly central ("star") coauthor. |
| `institution_prestige_best_tier` | Best (lowest-numbered) institution-prestige tier on the team; lower is more prestigious. |
| `institution_prestige_mean_tier` | Mean institution-prestige tier across the team. |
| `n_authors_moved_to_better_institution` | Count of authors who moved to a more prestigious institution before issue. |
| `author_ever_editor` | 1 if any author had ever served as a journal editor by issue. |
| `author_ever_top5_editor` | 1 if any author had ever served as a Top-5-journal editor. |
| `editor_tenure_years_max` | Longest prior editorial tenure on the team, in years. |
| `editor_tenure_years_sum` | Total prior editorial tenure across the team, in years. |
| `has_editor_coauthor` | 1 if any author has a coauthor who has been an editor. |
| `n_editor_coauthors` | Count of the team's coauthors who have been editors. |
| `n_advisors_who_are_top5_editors` | Count of the authors' PhD advisors who are Top-5-journal editors. |
| `acknowledged_editors` | Count of journal editors named in the paper's acknowledgments. |

### Outcomes

| column | type | description |
|---|---|---|
| `placement_rung` | integer 0-4 | Where the working paper was eventually published: 0 unmatched (never placed), 1 lower-tier journal, 2 mid-tier journal, 3 leading field journal, 4 Top-5 journal. |
| `placement_label` | text | Text label for `placement_rung` (`unmatched`, `lower`, `mid`, `top_field`, `top5`). |
| `reached_top5` | 0/1 | 1 if `placement_rung` == 4. |
| `prestige` | 0/1 | 1 if `placement_rung` >= 3 (a Top-5 or a leading field journal). The headline outcome. |
| `log_citations` | numeric | log(1 + citations to the published paper); 0 for papers with no recorded citations. Provided for context; the headline results do not use it. |

### Tier labels (derived; provided for convenience)

| column | type | description |
|---|---|---|
| `idea_tier` | text | Discipline-trained idea tier (`exceptional`, `strong`, `fair`, `limited`). The more conservative (more limited) of two independently trained model tiers, so a paper is only called high-quality if both models agree. |
| `offshelf_llm_tier` | text | Off-the-shelf model's idea tier, same four labels. |
| `has_editor_tie` | 0/1 | 1 if any author has ever been an editor or any editor is acknowledged. |
| `connection_tier` | text | `none` if `has_editor_tie` is 0; otherwise the paper's standing among connected papers, split into within-sample terciles of `connection_index` (`low`, `mid`, `high`). |

---

## `calibration_sample.csv`

One row per scored working paper in the full cohort: 6,208 papers. This is the
full set of papers that received an idea score, used for the measurement
calibration and for the apex-friction percentile cuts (which are taken over the
entire scored cohort, not the smaller regression sample).

| column | type | description |
|---|---|---|
| `paper_id` | text | Public NBER working-paper id. |
| `issue_year` | integer | Year the working paper was issued. |
| `idea_quality` | numeric | Discipline-trained idea-quality score (single primary model). |
| `idea_quality_ensemble` | numeric | Ensemble idea-quality score (geometric mean of two trained models). Used for the apex-friction percentile cuts. |
| `ensemble_tier` | text | Idea tier from the ensemble score (`exceptional`, `strong`, `fair`, `limited`). |
| `placement_rung` | integer 0-4 | Realized placement (same coding as above). |
| `placement_label` | text | Text label for the placement rung. |
| `reached_top5` | 0/1 | 1 if placed in a Top-5 journal. |
| `prestige` | 0/1 | 1 if placed in a Top-5 or a leading field journal. |
| `idea_tier` | text | Conservative discipline-trained idea tier (as above). |

---

## How the public files were derived

These two files were produced from the study's source records by keeping only
numeric derived features and scores, keying every row to the public NBER
working-paper id, and replacing the primary author with a de-identified integer
cluster code. Author names, author identifiers, paper titles, editor names, and
journal identities were dropped before the public files were written.
Reproducing the results requires only the two CSVs above and the scripts in
`code/`.
