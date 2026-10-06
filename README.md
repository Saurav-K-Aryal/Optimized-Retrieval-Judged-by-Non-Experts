# Code for "Retrieval Done Right, Judged by Non-Experts"

Analysis code for the study of how non-expert labelers perceive responses from Llama 3.1 8B
alone and from the same model inside an AutoRAG-optimized retrieval pipeline, under a
2 x 2 x 2 repeated-measures design (pipeline x persona prefix x explicit-steps suffix).
The code reproduces every number, table, and figure in the paper from the ratings workbook
and re-parses the AutoRAG trial folders that selected the pipeline.

## Data

The ratings are not distributed with this code. They were collected for the earlier
symposium report that introduced the Project Comprehension instrument:

> Nias, J., Aryal, S. K., Watson, C., Blackstone, J., Smarr, S., Williams, L., and
> Washington, G. (2026). LLM forensic evaluation: Diagnosing actionability, uncertainty,
> and human comprehension in high-stakes outputs. *Proceedings of the AAAI Spring
> Symposium Series (SSS 2026)*, 74-82.

To request the data, contact the authors of that paper. What you will receive is a
de-identified workbook with the following tabs (the KEY tab that links labeler codes to
people is not shared and must not be present when the code is run for release purposes):

| Tab | Content |
|---|---|
| `MAIN` | 960 rows, one per labeler x response submission: `Item ID`, `Timestamp` (date), `Prompt_ID`, `Prompt`, `Response`, `LABELER_ID`, `PU`, `PP`, `CS`, `GT`, `AC`, `FW`, `GN` (general notes), `SU` (source-of-uncertainty checkboxes), `UN` (uncertainty notes) |
| `prompts` | 220 rows of prompt and response text with word counts (text is missing for `Base_Prompts_AR_011` to `_030`) |
| eight form tabs (`PSGM ...`, `PRGM ...`, `ASGM ...`, `BPGM ...`, `PSAR ...`, `BPAR ...`, `PRAR ...`, `ASAR ...`) | the raw form exports, one row per submission, with full timestamps and 30 blocks of 9 question columns; used to rebuild the long table and to separate the two double submissions |

`Prompt_ID` encodes the condition: `_L_` = base model, `_AR_` = AutoRAG pipeline;
`Persona` and `Steps` in the prefix mark the framings; the trailing number is the base prompt (1 to 30).

The AutoRAG trial folders (`<trial>/config.yaml`, `summary.csv`, per-node `summary.csv` and
`*.parquet`) are available from the same authors on request.

## Requirements

Python 3.11 or later. Install with

    pip install -r requirements.txt

`bambi` and `pymc` are needed only for the Bayesian cumulative-link models (step 4); everything
else runs without them when that step is skipped.

## Running the ratings analysis

    cd ratings_analysis
    python run_all.py "/path/to/COMBINED LABEL DATA.xlsx"              # full run, about 12 min on 2 cores
    python run_all.py "/path/to/COMBINED LABEL DATA.xlsx" --skip-ordinal   # skips step 4, about 5 min

`run_all.py` sets `RATINGS_XLSX` and runs the steps below in order, writing everything to
`out/` together with `out/MANIFEST.txt` (workbook checksum, package versions, and a checksum of
every output file). Each step can also be run on its own after setting
`RATINGS_XLSX=/path/to/workbook.xlsx`.

| Step | What it does | Main outputs |
|---|---|---|
| `step00_forms.py` | rebuilds the long rating table from the eight raw form tabs (full timestamps) and cross-checks it against `MAIN` | `form_long.parquet` |
| `step0_prep.py` | keeps the earlier of each double submission (900 unique ratings), extracts text features (word count, headers, numbered items, hedges, authority vocabulary, directive citations, refusal openings), parses the source-of-uncertainty tags | `ratings.parquet`, `rerate.parquet`, `responses.parquet`, `out/00_data_summary.txt` |
| `step1_descriptives.py` | descriptives, cell and marginal means with cluster-bootstrap CIs over base prompts, paired contrasts, correlations, PCA, Cronbach alpha | `out/01_*.csv`, `out/01_descriptives.txt`, `fig1_interaction.*`, `fig2_distributions.*` |
| `step2_irr.py` | ordinal Krippendorff alpha with bootstrap CIs (all labelers, core five, by pipeline, by condition), ICC(2,1)/ICC(2,k) on complete blocks, pairwise weighted kappa, test-retest for the two double submissions, labeler severity | `out/02_*.csv`, `out/02_irr.txt` |
| `step3_models.py` | primary rating-level linear mixed models with crossed random intercepts (labeler, base prompt, response), response-level model, robustness refits (labeler fixed, core five, refusals excluded two ways, length-adjusted, ML), variance decomposition, leave-one-labeler-out, simple effects, Bonferroni check | `out/03_*.csv`, `out/03_models.txt` |
| `step4_ordinal.py` | Bayesian cumulative-link mixed models (bambi/PyMC, 4 chains x 1000 draws, seed 11) | `out/04_ordinal_coefficients.csv`, `out/04_ordinal.txt` |
| `step5_secondary.py` | source-of-uncertainty tags (descriptives and linear probability mixed models), surface quality vs grounding, per-prompt retrieval effects, text-feature model, refusal analysis, keyword coding of free-text notes | `out/05_*.csv`, `out/05_secondary.txt`, `fig3_rag_by_prompt.*` |
| `step6_report.py` | LaTeX tables and the coefficient and variance figures | `out/tables.tex`, `fig4_coefficients.*`, `fig5_variance.*` |

All random seeds are fixed; a second run on the same workbook reproduces every output exactly
(the ordinal step is deterministic given the seed and library versions in `MANIFEST.txt`).

### Generating tables and figures come from

| Paper | File |
|---|---|
| Table: instrument wording | written by hand from the form tabs |
| Table: labeler allocation and refusals | `out/tables.tex` (`rater_allocation`), counts in `out/00_data_summary.txt` |
| Table: descriptives and reliability | `out/tables.tex` (`descriptives`); `out/01_descriptives.csv`, `out/02_krippendorff.csv`, `out/02_icc_by_condition.csv`, `out/03_variance_decomposition.csv` |
| Table: primary mixed models | `out/tables.tex` (`primary_model`); `out/03_model_coefficients.csv` (model `A_rating_crossed`) |
| Table: robustness | `out/tables.tex` (`robustness`); `out/03_model_coefficients.csv`, `out/04_ordinal_coefficients.csv` |
| Supplement: cell means, test-retest, text features, source-of-uncertainty tags | `out/tables.tex` (`cell_means`, `test_retest`, `text_features`, `su_tags`) |
| Figure 1 (means by condition) | `out/fig1_interaction.pdf` |
| Figure 2 (coefficient plot) | `out/fig4_coefficients.pdf` |
| Figure 3 (retrieval effect per prompt) | `out/fig3_rag_by_prompt.pdf` |
| Text-feature model, refusal statistics, notes coding | `out/05_secondary.txt`, `out/05_text_features_model.csv`, `out/05_notes_coded.csv` |
| Leave-one-labeler-out, simple effects, Bonferroni check | `out/03_models.txt`, `out/03_leave_one_rater_out.csv`, `out/03_simple_effects_*.csv` |

Labelers appear as `L01` to `L09` (no `L04`) in the data and the outputs; the paper relabels them
`R1` to `R8` in that order.

## Running the AutoRAG re-parse

    cd autorag_analysis
    python autorag_parse.py /path/to/experiment_results --out analysis

The script walks every trial folder, skips duplicated copies, groups trials into comparable
families (same prompt templates and metrics), and writes `trials.csv`, per-node configuration
tables, `generator_by_model.csv`, `model_pairwise_wins.csv`, `variance_decomposition.csv`,
`optimal_pipeline.yaml` and `.json`, and `report.md`. The selection rules (which metric breaks
which tie) are stated at the top of the script. The report's caveats section records what the
trial outputs show about the effective search space (three evaluation queries, identical
retrieval across the declared embedding models, rerankers tied on every metric, prompt templates
that hard-code a question, and `context_length` being the generator's context window rather
than a chunk size). 

## Notes

* Nothing in this repository contains labeler names or contact details. If you receive a
  workbook that still has a `KEY` tab, delete that tab before running or sharing anything.
* `step0_prep.py` recovers the missing prompt text for `Base_Prompts_AR_011` to `_030` from the
  base-model twin of the same base prompt; the response text for those 20 items is not
  recoverable and the text-feature analyses use the 220 items that have text.
* Thresholds that are judgment calls are explicit in the code and reported as sensitivity
  analyses in the paper: a hard refusal is a response that opens with a refusal and has fewer
  than 50 words (`hard_refusal`), a refusal opening of any length is `refusal`; the core-labeler
  set is the five labelers who rated at least four forms.
