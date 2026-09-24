# Spindle → Genome Biology: Experiment Plan

This is the single tracking document for what's left before submission. Target: 4–6 weeks.
Keep this file current as the source of truth — don't create parallel plan files.

## Context

`paper/main_genome_biology.tex` received a pre-submission review. Its main problems: numeric
claims (speedup/build-time/recall) don't match each other across sections, the dataset count
is inconsistent (6 vs. 7 mentions), the Methods "Algorithm" section describes a single-niche
routing step the code no longer has, and a niche-assignment-accuracy claim is no longer true.
This plan fixes those and adds the ablations/baselines/robustness checks the review asked for.

## Current state of the method (what every task below should treat as ground truth)

- **Search**: every query searches **every niche's DAG** (no single-niche routing) — Stage-1
  candidates from all niches are merged, then Stage-2 exact-re-ranks, capped at `top_c=400`
  candidates per niche.
- **Ground truth for accuracy evaluation is the block-diagonalized Frobenius log-Euclidean
  distance** (`benchmarks/holdout_validation.py::compute_ground_truth`) — not a whole-matrix
  comparison. Rationale: the index only ever claims *approximate* block-diagonal structure
  (manuscript §4.1 Assumption 2, §4.2's own "Approximate block diagonalization" heading), and
  Proposition 1 proves the LE distance decomposes *exactly* under that assumption — so
  block-diagonalized distance is what the method's correctness guarantee is stated in terms
  of, not a lossy stand-in for "the real answer". A whole-matrix ground truth was also
  computed (`compute_ground_truth_whole_matrix`) and is empirically far stricter — recall
  against it plateaus well below 1.0 (near 0 for `lymph_node_5k`) even at very high search
  budget. Keep it as Discussion/supplementary evidence for *why* block-diagonalized is the
  right comparison; never as a primary Results number.
- **Metrics reported**: `recall_at_eps_{0.1,0.5}` and `overlap_at_eps_{0.5,1.0}` (a
  distance-tolerance band scaled by each niche's own epsilon — see
  `EPSILON_TOLERANCE_FRACTIONS` in `benchmarks/holdout_validation.py`). Do not report
  `recall_at_1` (a near-miss scores a full failure — too punishing to be meaningful).
- **Production `budget_multiplier = 1.0`**, all 8 datasets — at this value,
  `recall_at_eps_0.1` min=0.96/mean=0.99 and `mean_speedup` min=24×/mean=37×, with essentially
  no tradeoff against lower values in the swept range (0.3–2.0 gives the same speedup).
  Source: `results/budget_sweep_holdout/sweep_summary_block.csv`.
- **Preprocessing drops low-density tiles by default**
  (`src/spindle_dev/preprocessing.py::build_quadtree_tiles(filter_low_density=True)`) — avoids
  near-empty background/debris tiles collapsing to identical covariances and distorting
  clustering/search.
- **Holdout methodology**: fixed-count holdout (100 tiles per dataset, not a percentage) via
  `--seed`/`--n-holdout` on `index_datasets.py` and `budget_sweep_holdout.py`. The current
  sweep used `seed=73`. **E5 (multi-seed reruns) uses a different, percentage-based holdout**
  (`--train-test-ratio 0.10`, i.e. 10% of tiles) instead, per explicit user decision for that
  task — its 5 seeds (`{0,1,2,3,4}`) are not directly comparable in raw holdout tile *count* to
  E1's fixed-100 sweep, though both target the same `budget_multiplier=1.0` production point.
- **All 8 datasets, used consistently everywhere**: breast, kidney, lung, lymph_node,
  lymph_node_5k, pancreatic, skin, brain. Paths:
  `/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium/xenium_human_{breast_cancer,
  kidney_nondiseased,lung_cancer,lymph_node,lymph_node_5k,pancreatic_cancer,brain_cancer,
  skin_melanoma}.h5ad`.
- **Cross-modal search (E12)** runs on a separate, matched Xenium/Visium breast-cancer pair
  (`dataset/cross_modal/{xenium,visium}_rotated.h5ad`, gitignored, copied from the original
  machine; 307 shared genes, 165 Xenium / 156 Visium tiles). It follows the same conventions as
  above (all-niche search, block-diagonalized ground truth, eps metrics, `budget_multiplier=1.0`)
  plus one **global** modality bias correction: a single whole-matrix tangent-space shift
  (`L_corr = (L_q − mean_q)·min(1, sd_t/sd_q) + mean_t`) applied before any niche layout. The
  correction is batch-level: the query-side statistics come from a set of query-platform tiles,
  so cross-modal search requires a **reference dataset representative of the query tile's
  distribution** (documented as a Discussion/limitation point in `paper/results_tracking.md`).

## Decisions confirmed with user

- Baseline comparator: FAISS/HNSW over flattened covariance features (not an attempt to run
  SOAR/STomicsDB/geneSCOPE themselves) — once that lands, soften the Background's critique of
  those tools to a qualitative capability contrast rather than a benchmark comparison.
- Cross-dataset biological ground truth: obtain external cell-type/cluster annotations for
  kidney, lung, pancreas, skin, lymph (breast already validated) and proceed with ARI/NMI.
- Local vs. server: assign each experiment to *one* venue as a whole — no splitting a single
  experiment's datasets across local/server. Default to server for anything long/heavy.
- Report mean±sd across seeds wherever error bars are shown (E5), not point estimates.
- E12 cross-modal bias correction: **global** whole-matrix shift (not per-niche). The
  per-(niche, block) all-query variant collapsed the true nearest neighbour onto one niche
  (v2x recall@eps0.1 0.24 vs 0.98 on seed 0) — see `results/cross_modal_search/correction_ablation_seed0/`.
- E12 `--seed` controls both the random query subsample (50 per direction) and the index build's
  PCA/UMAP/Leiden `random_state`.

## Tracking-as-we-go and results hygiene

- Keep a running `paper/results_tracking.md` (plain markdown, not a parallel manuscript): one
  short section appended per stage below as it locks in — final numbers, the exact CSV/column
  each came from, and the figure/table it will become. This file is the numbers staging area;
  this file's "Progress Tracking" table stays the single status source; `main_genome_biology.tex`
  itself only gets touched once, at the end (E0).
- After each stage's numbers are locked in and sanity-checked, delete any superseded/wrong/debug
  CSVs for that stage's output directory before moving to the next stage — don't leave stale
  pre-fix files sitting next to the final ones. Per-seed outputs that feed a mean±s.d. aggregate
  are final results, not scratch — keep them.
- `results/holdout_validation_indexed/` (index+covariance pickles, ~24G) and
  `results/ground_truth_cache/` (~34G) are reusable build artifacts, not stage outputs — leave
  them alone during per-stage cleanup. **Done for E5** (2026-09-23): its 40 seed-suffixed
  index/covariance pickles (116G) and ground-truth caches (162G) were dropped post-lock-in,
  reproducible from `--seed`/`--train-test-ratio` alone; the 8 original unsuffixed pairs (24G/34G)
  were kept — still needed by not-yet-run E7 (E12 builds its own Visium/Xenium indices and never used them). Also dropped in the same pass:
  `results/holdout_validation/` (stale pre-rewrite data, see `paper/results_tracking.md`'s
  "Session cleanup" note — kept only `index_scalability_summary.csv`),
  `results/multiseed_holdout/_dataset_symlinks/`, and `logs/*.out`. **Done for E12**
  (2026-09-24): deleted the pre-fix `results/cross_modal_search/{benchmark_summary,x2v_query_metrics,
  v2x_query_metrics}.csv` + `tile_overlay.png`, the smoke-test output, and this stage's
  `logs/*cross_modal*.out`/`logs/cm_diag_*.out`. Kept the per-seed dirs, `summary*.csv`,
  the figure-input CSVs, and `correction_ablation_seed0/`. The ~200 `logs/spindle_multiseed_partial_panel_*.out`
  files left over from E11 (2026-09-23) are still there, superseded by `results/run_logs/*.json`,
  and can be deleted.
- Every new/updated benchmark script (E5, E11, E12, E2–E4, E6, E7, E9, E10) wraps its per-dataset/
  per-run body in `benchmarks/run_logging.py::RunLogger` — E8 isn't a separate step, it's a
  requirement baked into every task prompt below.

## Existing infrastructure (reuse, don't rebuild)

- `benchmarks/index_datasets.py` — builds/pickles the index per dataset
  (`results/holdout_validation_indexed/`); `--seed`/`--n-holdout`/`--train-test-ratio` exposed.
- `benchmarks/holdout_validation.py` — core recall/speedup benchmark: searches all niches,
  computes/caches both ground truths (`compute_ground_truth` block,
  `compute_ground_truth_whole_matrix` whole, `load_or_compute_ground_truth` handles caching to
  `results/ground_truth_cache/`), `evaluate_against_ground_truth` produces
  `recall_at_eps_*`/`overlap_at_eps_*`/`spindle_time_ms`/`bf_time_ms`/`speedup` (the latter two
  always measured against the whole-matrix brute-force cost — see its docstring for why).
- `benchmarks/budget_sweep_holdout.py` — sweeps `budget_multiplier` (28-point grid, 0.05–16.0)
  against both ground truths per dataset; plateau early-stop never fires before
  `budget_multiplier=1.0`. Produces `results/budget_sweep_holdout/{name}/*_{block,whole}.csv`
  and merged `sweep_summary_{block,whole}.csv`.
- `benchmarks/run_logging.py` — `RunLogger` context manager, writes per-dataset/per-stage wall
  time, peak memory, and device specs to `results/run_logs/*.json`.
- `scripts/generate_budget_sweep_figure.py` — `figures/fig_budget_sweep_{block,whole}.{pdf,png}`.
- `scripts/generate_epsilon_scale_figure.py` — `figures/fig_epsilon_scale.{pdf,png}`
  (distribution of within-niche pairwise distances vs. ε — supports the block-only argument).
- `benchmarks/partial_panel_search.py` — **fixed under E11** (query construction rewritten around
  `build_binned_queries`, `RunLogger`/`--seed` added, all 8 datasets; see `paper/results_tracking.md`).
  `benchmarks/cross_modal_search.py` — **fixed under E12** (all-niche search, holdout GT/metrics,
  global bias correction, `RunLogger`/`--seed`, data at `dataset/cross_modal/`; see
  `paper/results_tracking.md`). `benchmarks/gene_signature_search.py` (pathway enrichment) still
  calls `search.assign_clusters_to_new_spds(...)` (single-niche routing), has no `RunLogger`, and
  hardcodes `np.random.seed(42)` -- its fix is scoped inside E9.
- E12 infrastructure: `benchmarks/multiseed_cross_modal_search.py` (aggregate-only; `--variant
  single_niche_baseline` for the routing diagnostic), `slurm_jobs/run_cross_modal_search.sbatch`
  (`<seed> [extra args]`) + `submit_cross_modal_search.sh` (seeds 0–4; seed 0 also runs
  `--single-niche-baseline --write-overlay`). ~2.5 min/job, ~1.2 GB peak RSS.
- Shared-code changes made during E12 (additive, defaults unchanged):
  `holdout_validation.compute_ground_truth(..., query_blocks_log_override=None)`;
  `index_datasets.run_index(..., random_state=0)`; and `ProcessedData.cluster_spds` now actually
  forwards `random_state` to `leiden_clustering_latent`. Before, Leiden always ran with seed 0
  whatever was passed, so any future seed-varying clustering (e.g. E3) gets real seed variation
  only from this change onward. Earlier results are unaffected (all used 0).
- `scripts/organize_panel_data.py` → `scripts/generate_result_figure.py` — curates benchmark
  CSVs into `results/panel_data/panel_{A-J}.csv` and renders `figures/fig_main_result.pdf`.
- `src/spindle_dev/metrics.py::add_spd_noise()` — noise-injection primitive, used by E7.
- `src/spindle_dev/index.py`: `block_diagonalize: bool` and Leiden `resolution` are real,
  exposed parameters (used by E2/E3).
- `slurm_jobs/`: `run_single_index_build.sbatch` + `submit_reindex_all.sh` (parallel index
  builds), `run_single_budget_sweep.sbatch` + `submit_all_budget_sweep.sh` (parallel sweeps,
  dependency-chained on their index jobs), `run_generate_figures.sbatch` (cheap, figure-only).
- No FAISS/HNSW/Annoy installed yet — must be added to `requirements.txt` for E4.
- Conda env: `spindle_env` (`source ~/miniforge3/bin/activate base && conda activate spindle_env`).
- **Keep all large/long-running experiments on the server via `sbatch`** — never run a full
  index build or benchmark sweep directly on the login node.

## Experiment Inventory

| ID | Experiment | Venue | Status | Output → Figure/Table |
|----|---|---|---|---|
| E1 | Recall-vs-budget tradeoff curve | Server | **Done** | `figures/fig_budget_sweep_block.{pdf,png}` (main text), `fig_budget_sweep_whole.{pdf,png}` + `fig_epsilon_scale.{pdf,png}` (Discussion/supplementary) |
| E5 | Multi-seed holdout-validation reruns for error bars (5 seeds × 8 datasets, `budget_multiplier=1.0` locked) | Server | **Done** | `results/multiseed_holdout/summary.csv`; mean±s.d. error bars on every headline number |
| E11 | Fix `partial_panel_search.py`'s query construction (drop `assign_clusters_to_new_spds`, length-targeted binned draws), add `RunLogger`/`--seed`, multi-seed rerun | Server | **Done** | `results/multiseed_partial_panel_search/summary.csv`, `figures/fig_partial_panel_search_by_length.{pdf,png}`, `figures/fig_partial_panel_search_by_case.{pdf,png}` |
| E12 | Fix `cross_modal_search.py` to search all niches, fix hardcoded `/home/asus/...` paths, add `RunLogger`/`--seed`, multi-seed rerun | Server | **Done** | `results/cross_modal_search/summary.csv`, `figures/fig_cross_modal.{pdf,png}`, main-figure panel H |
| E2 | Ablation: block-diagonalization on/off (breast, lung, pancreas) | Local/Server | Not started | New ablation table/figure |
| E3 | Ablation: Leiden resolution sweep — niche granularity's effect on Stage-1 cost/compression (recall should be roughly flat now that every niche is searched — confirm, don't assume) | Server | Not started | New ablation figure: speedup/niche-count/index-size vs. resolution |
| E4 | Baseline: FAISS/HNSW over flattened covariance features, all 8 datasets | Server | Not started | New comparison table/figure: Spindle vs. generic ANN |
| E6 | Scalability sweep across 2–3 orders of magnitude | Server | **Done** | `figures/fig_scalability_sweep.{pdf,png}`, `results/scalability_sweep/*_synthetic_scaling.csv` + `*_scaling_exponents.json` |
| E9 | Systematic enrichment quantification across all 6 tissue datasets + stats test | Local | Not started | New summary table + effect-size column |
| E10 | ARI/NMI vs. external annotations, all datasets | Local — blocked on annotation files | Not started | New table: ARI/NMI per dataset |
| E7 | Noise robustness → end-to-end Recall@ε vs. noise level | Local/Server | Not started | Replacement figure for the removed niche-assignment claim |
| E0 | **Numeric reconciliation**: rewrite every numeric claim in `main_genome_biology.tex` from all locked-in results (dataset count, `tab:xenium_summary`, §3.1/§3.2 prose, ablation/baseline/enrichment/ARI-NMI/noise tables); rewrite the Methods Algorithm section; remove the stale niche-assignment claim; add a Discussion paragraph on block-only reporting | Local | **Last — not started** | Full manuscript numeric pass |
| E8 | Environment logging | Cross-cutting (`RunLogger` in every script above), zero separate compute | Mostly done | "Computational environment" paragraph, written alongside E0 |

Not in this plan (per user decision): directly benchmarking SOAR/STomicsDB/geneSCOPE.

## Sequencing

Each stage's numbers must be locked in (sanity-checked, final CSV committed under `results/`,
`paper/results_tracking.md` updated) before starting the next stage's code changes.

1. **Stage A — E5** (done): `budget_multiplier=1.0` is already the confirmed production value (see
   "Current state of the method" above) — no new decision needed. Write
   `benchmarks/multiseed_holdout.py`, seeds 0–4, all 8 datasets, `RunLogger`-wrapped.
2. **Stage B — E11** (done): fix `partial_panel_search.py`'s single-niche routing, add
   `RunLogger`/`--seed`, rerun it for all 8 datasets across seeds.
3. **Stage C — E12** (done): same treatment for `cross_modal_search.py`, plus fixing its hardcoded
   dataset paths.
4. **Stage D — E2, E3, E4, E6, in parallel**: none of these depend on Stages B/C's code, only
   on the finalized `budget_multiplier=1.0` convention from Stage A.
5. **Stage E — E9, E10, in parallel**: independent of Stage D; may run concurrently with it if
   resourcing allows, listed after per explicit ordering.
6. **Stage F — E7**: last experiment; depends only on Stage A's index builds.
7. **Stage G — E0 (+ E8's writeup)**: once all of A–F are locked in, do the full manuscript
   numeric pass in one clean sweep.

## Verification

- E1: done — sanity-checked (`recall_at_eps_1.0 ≥ recall_at_eps_0.5 ≥ recall_at_eps_0.1` by
  construction; plateau never triggers before `budget_multiplier=1.0` by construction).
- E5: reruns across seeds produce different-but-close numbers (sanity check) before reporting
  mean±s.d.
- E11: done — the "recall should go up" check from this plan's original premise didn't apply
  (search was already all-niches; no single-niche-routing bug existed at that level). Actual
  verification: `rank` ≈ 1 across niches/bins (confirms search correctness independent of the
  overlap metric's near-set-size sensitivity); every length bin populated with exactly
  `--num-queries` draws (confirms the query-construction fix).
- E12: done. The all-niche search is at least as good as kNN-routed single-niche search, scored
  against the same ground truth on seed 0 (x2v 1.00 vs 0.00, v2x 0.98 vs 0.68;
  `results/cross_modal_search/summary_single_niche_baseline.csv`). Recall is monotone in eps for
  every seed and direction, there are no empty candidate sets, and `spindle_best_rank == 1` for
  92–100% of queries.
- E2/E4: new comparison tables reproduce Spindle's already-known numbers as the baseline row
  before trusting new rows.
- E3/E6/E7: new figure PDFs render and are correctly referenced in the manuscript; underlying
  CSVs saved under `results/<experiment_name>/`.
- E9/E10: spot-check one dataset's enrichment/ARI number by hand.
- E0: every number in the manuscript traces to a specific CSV/JSON cell (via
  `paper/results_tracking.md`); dataset count reads "eight" everywhere; PDF recompiles with no
  figure-reference or `% TODO` markers left.
- Final: recompile `main_genome_biology.tex` end-to-end — PDF builds, all figure references
  resolve, no `% TODO` markers remain anywhere.

---

## Standalone task prompts

Each block below is self-contained — paste it into a fresh chat (no prior context needed) to
execute that task independently.

### E1 — Recall-vs-budget tradeoff curve (done — for reference only)

```
Done. benchmarks/budget_sweep_holdout.py sweeps budget_multiplier (28-point
grid, 0.05-16.0) against both a block-diagonalized and a whole-matrix ground
truth, per dataset, with a plateau early-stop keyed on the whole-matrix
ground truth's recall_at_eps_0.25 that never fires before
budget_multiplier=1.0. Figures: figures/fig_budget_sweep_block.{pdf,png} (put
in the manuscript), figures/fig_budget_sweep_whole.{pdf,png} +
figures/fig_epsilon_scale.{pdf,png} (Discussion/supplementary only -- support
the block-only-reporting argument, never a primary Results number).
Re-run only if the methodology changes again, via
slurm_jobs/submit_reindex_all.sh + slurm_jobs/submit_all_budget_sweep.sh,
then scripts/generate_budget_sweep_figure.py.
```

### E5 — Multi-seed holdout-validation reruns (done — for reference only)

```
Done. Used benchmarks/holdout_validation.py (single budget point, fixed at
its own default budget_mult=1.0), not budget_sweep_holdout.py -- the latter
is E1's 28-point grid tool, unnecessary overkill for one production-value
rerun. benchmarks/multiseed_holdout.py wraps index_datasets.py ->
holdout_validation.py per dataset x seed (unit mode), pointing both at a
seed-suffixed symlink of the real dataset file (both scripts derive
dataset_name from Path(path).stem, so this gives each seed its own
index/covariance/run-log/ground-truth-cache artifacts with zero changes to
either script), plus an --aggregate mode that computes mean+-s.d. across
seeds into results/multiseed_holdout/summary.csv. Per user decision, the
holdout split used --train-test-ratio 0.10 (10% of tiles) instead of E1's
fixed --n-holdout 100, for all 5 seeds (0-4), all 8 datasets -- 40
independent index-build + single-search-pass runs total, submitted via
slurm_jobs/submit_multiseed_holdout.sh.

holdout_validation.py itself needed a fix first: its bf_time_ms/speedup
were computed against the block-diagonalized ground truth's own brute-force
timing instead of the whole-matrix one (the documented invariant
budget_sweep_holdout.py already enforced) -- fixed by having it also
compute the whole-matrix ground truth via load_or_compute_ground_truth,
matching budget_sweep_holdout.py's pattern; also added RunLogger wrapping
and --seed/--n-holdout/--train-test-ratio args (informational, for
logging/cache-key metadata only) it was missing. The already-run units were
patched in place (not re-run) via benchmarks/patch_whole_matrix_speedup.py,
which computes only the missing whole-matrix ground truth and rewrites
bf_time_ms/speedup, via slurm_jobs/run_patch_whole_matrix_speedup.sbatch +
submit_patch_whole_matrix_speedup.sh.

Final mean±s.d. table and source columns in paper/results_tracking.md.
Existing single-seed results/budget_sweep_holdout outputs were kept (not
deleted) -- E1 still needs the full sweep curve; only the headline
recall/speedup citation moved to the multi-seed summary.
```

### E11 — Fix partial-panel search query construction, then multi-seed (done — for reference only)

```
Done. Investigation found this task's original premise was partly wrong:
data_helpers.search_all_clusters_spindle (the actual search function used by
this benchmark) already searched every niche and every block exhaustively --
no single-niche-routing bug existed at the search level, unlike the pattern
holdout_validation.py had. The real bugs were in query construction and
metrics: (1) search.assign_clusters_to_new_spds predicted one niche per
held-out query, used only to pick which niche's gene-block layout defined
the synthetic partial-panel test query (an unnecessary guess, not a search
gate); (2) run_benchmark_suite hardcoded block_index=0, discarding the
caller's intended block round-robin; (3) no RunLogger/--seed, only 6/8
datasets, and old rank-based metrics (recall_at_1, overlap_at_N) banned by
"Current state of the method". Fixed by: dropping assign_clusters_to_new_spds
entirely and interleaving every (niche, block_index) pair round-robin across
niches first (a first version that flat-sorted pairs by niche then block was
caught mid-sweep -- it let niche 0 alone dominate every query whenever
--num-queries was smaller than niche 0's own block count, true for every
dataset at the default of 5 -- re-run after the interleaving fix); switching
metrics to recall_at_eps_{0.05,0.1}/overlap_at_eps_{0.1,0.25} (smaller
fractions than holdout_validation.py's, because partial-gene-interval
distances are far less discriminative than full-block distances -- verified
empirically, see paper/results_tracking.md); and rescaling the epsilon
tolerance band per query by sqrt(valid_len/block_size) (the block epsilon
was calibrated against full-block distances, not sub-range interval
distances). Added RunLogger, --seed/--n-holdout/--train-test-ratio, and the
2 missing datasets (lymph_node_5k, brain_cancer). New
benchmarks/multiseed_partial_panel_search.py mirrors multiseed_holdout.py's
seed-symlink pattern, seeds 0-4, --train-test-ratio 0.10, all 8 datasets,
submitted via slurm_jobs/submit_multiseed_partial_panel_search.sh.

Initial full sweep used the script's original default --num-queries=5 and
showed recall ~1.0 everywhere -- looked suspicious given how thin a sample
that is (50 rows/seed across 4 length bins). Per explicit user request to
push it harder, reran at --num-queries=50 (10x). This changed the picture
materially, not just the error bars: recall is no longer uniformly 1.0
(small genuine imperfections show up, most on lymph_node_5k), and
overlap-vs-query-length is an inverted U (rises then falls) rather than the
monotonic rise the n=5 pilot suggested -- that apparent monotonic trend was
an undersampling artifact.

Query construction itself went through several redesigns before landing.
Round-robin (niche, block) sampling -- even after fixing a niche-interleaving
bug that let niche 0 dominate every query (caught via lymph_node_5k's
per-seed CSVs showing exactly 1 distinct Niche value) -- left long-query
bins completely empty for most datasets, since large blocks are rare (~5-6%
of all blocks everywhere) and pure round-robin never sampled deep enough
into any one niche's block list to reach them. Fixed by rewriting query
construction around explicit length control
(partial_panel_search.py::build_binned_queries): for each length bin, find
every (niche, block) pair large enough to hold a target length drawn from
that bin, pick one at random, and carve out a contiguous/non-contiguous
gene range of exactly that length -- guaranteeing every bin gets exactly
--num-queries draws regardless of how rare qualifying blocks are. This was
first built with 8 narrower bins to fully close the empty-bin problem
(verified working), then reverted back to the original 4 bins
(<=6/7-12/13-16/>16) per explicit user decision -- keeping the
guaranteed-per-bin-draw mechanism, which works identically well at 4 bins.

The overlap metric itself was also revisited multiple times after the
brain_cancer/lymph_node_5k investigation surfaced a near-set-size ceiling
effect (a large true-near-set structurally bounds
intersection/|true_near| at top_k/|true_near| regardless of search quality).
Three alternatives were implemented, empirically verified, and explicitly
reverted: a min-based overlap coefficient (fixes the ceiling but collapses
to an implicit precision measure, losing signal), uncapped search
(top_k=None -- backfired badly: returned literally the entire 1,944-tile
dataset for one degenerate query, defeating the metric rather than fixing
it), and a fixed-K top-K set-overlap metric (overlap_at_20/50 -- worked
cleanly, sidesteps the ceiling entirely, but was reverted to keep the
manuscript's overlap definition consistent with holdout_validation.py's
eps-based convention). Final metrics: recall_at_eps_{0.05,0.1} and
overlap_at_eps_{0.1,0.25}, with the near-set-size caveat now documented in
prose rather than as an extra reported column. Full history in
paper/results_tracking.md's "Methodology dead ends" (n=50, 4-bin,
eps-only numbers are what's locked in and reported).

Two figures: figures/fig_partial_panel_search_by_length.{pdf,png}
(scripts/generate_partial_panel_search_figure.py, now a 4x2 grid) --
recall/overlap vs. query length, one panel per dataset; and NEW
figures/fig_partial_panel_search_by_case.{pdf,png}
(scripts/generate_partial_panel_search_by_case_figure.py) -- grouped bars
per dataset comparing contiguous vs. non-contiguous query patterns,
collapsing length. Finding: the two patterns perform statistically
indistinguishably everywhere. Superseded pre-fix data (stale, from
2026-09-07/08, predating this session) and intermediate per-seed run dirs
from every sweep this session (initial fix-verification, niche-interleaving
fix, 8-bin trial, final 4-bin n=50 push) were deleted per the plan's hygiene
rule -- only the final configuration's per-seed CSVs were kept.
```

### E12 — Fix cross-modal search to search all niches, then multi-seed (done — for reference only)

```
Done. benchmarks/cross_modal_search.py rewritten around holdout_validation.py:
every niche's DAG searched per query (single-niche kNN routing removed),
hv.compute_ground_truth (block-diagonalized, all niches) + hv.evaluate_against_
ground_truth (recall/overlap_at_eps, whole-matrix-bf speedup, top_c=400),
production index pipeline (index_datasets.run_index + configure_and_build_dag),
budget_mult 1.0, RunLogger, --seed (query subsample + PCA/UMAP/Leiden
random_state), --visium-path/--xenium-path (default dataset/cross_modal/).
Modality bias correction is now one GLOBAL whole-matrix tangent-space shift
(--correction global); the per-(niche,block) all-query variant was tried first
and collapsed the true NN onto one niche (v2x recall_at_eps_0.1 0.24 vs 0.98 on
seed 0) -- ablation kept in results/cross_modal_search/correction_ablation_seed0/.
Seeds 0-4 via slurm_jobs/submit_cross_modal_search.sh, aggregated by
benchmarks/multiseed_cross_modal_search.py -> results/cross_modal_search/
summary.csv. --single-niche-baseline (seed 0) confirms all-niche >= routed.
Figures: fig_cross_modal.{pdf,png} + main-figure panel H. Numbers and caveats
in paper/results_tracking.md.
```

### E2 — Ablation: block-diagonalization on/off

```
Repo: /home/NAShome/ghoss18/spindle_dev (server). src/spindle_dev/index.py's
ProcessedData.cluster_spds() takes a block_diagonalize: bool = True parameter
that is never toggled anywhere in benchmarks/. Build a small harness that,
for 3 datasets (breast, lung, pancreas), runs the full index_datasets.py ->
budget_sweep_holdout.py pipeline twice per dataset (seed=1, n_holdout=100,
budget_multiplier=1.0): once with block_diagonalize=True (current
default/baseline) and once with block_diagonalize=False. Save outputs under
results/ablation_block_diag/<dataset>/{on,off}/. Produce a comparison
table/figure (recall_at_eps_0.1, mean speedup, index size, build time, with
vs. without block-diagonalization) as results/ablation_block_diag/summary.csv
and a simple figure. Report whether block-diagonalization meaningfully helps
recall/speed/size, with numbers. Run big steps via SLURM.
```

### E3 — Ablation: Leiden resolution sweep

```
Repo: /home/NAShome/ghoss18/spindle_dev (server). Every niche's DAG is
searched per query now (no single-niche routing), so Leiden resolution
mainly trades off niche count against Stage-1 search cost and index
compression/block-diagonalization quality -- NOT recall, which should be
roughly resolution-independent. For 3 datasets (breast, lung, pancreas),
rebuild the index at a sweep of Leiden resolutions and rerun the budget
sweep (budget_multiplier=1.0) at each. Report speedup, niche count, and
index size vs. resolution -- confirm recall stays roughly flat rather than
assuming it. Save under results/leiden_resolution_sweep/<dataset>/<resolution>/.
Run big steps via SLURM.
```

### E4 — Baseline: FAISS/HNSW over flattened covariance features

```
Repo: /home/NAShome/ghoss18/spindle_dev (server). The only comparator
anywhere in the codebase/manuscript is exact brute-force search. Add
faiss-cpu (preferred) or hnswlib to requirements.txt. Write
benchmarks/faiss_baseline.py that, for all 8 datasets, flattens/vectorizes
each tile's SPD covariance matrix (matching whatever feature representation
src/spindle_dev/index.py uses before its own PCA/clustering step, for a fair
comparison), builds a FAISS (IndexHNSWFlat or similar) or hnswlib index over
those vectors, and runs the same held-out query set used in
benchmarks/holdout_validation.py, computing the same metrics
(recall_at_eps_{0.1,0.5}, overlap_at_eps_{0.5,1.0}, query time, speedup vs.
brute force -- use budget_multiplier=1.0 for Spindle's own numbers). Save
results to results/faiss_baseline/<dataset>_query_metrics.csv and a summary
CSV. Produce a comparison table/figure: Spindle vs. this baseline, per
dataset, on recall/speedup/index size. Report which method wins where, and
by how much. Run big steps via SLURM.
```

### E6 — Scalability sweep across 2-3 orders of magnitude (done — for reference only)

```
Done. benchmarks/scalability_sweep.py subsamples (target <= real cell count)
or bootstrap-resamples-with-replacement + jitters spatial coordinates
(target > real cell count) breast_cancer (159,226 real cells) and
lymph_node (377,985 real cells) to synthetic cell counts [1e3, 5e3, 2e4,
5e4, 1e5, 5e5, 1e6], reruns the real run_index/configure_and_build_dag
pipeline (imported from index_datasets.py) at each size, and records
build_time_s/index_size_mb. Unit mode writes one row per (dataset,
target_cells) to results/scalability_sweep/<dataset>/<dataset>_cells<N>_seed<n>.csv
(avoids concurrent-write races across parallel SLURM jobs); --aggregate
merges into results/scalability_sweep/<dataset>_synthetic_scaling.csv and
fits log-log scaling exponents (scipy.stats.linregress) to
*_scaling_exponents.json. Submitted via slurm_jobs/run_scalability_sweep.sbatch
+ submit_scalability_sweep.sh. Figure: figures/fig_scalability_sweep.{pdf,png}
(scripts/generate_scalability_sweep_figure.py) -- distinct from the existing
real-dataset-only fig_scalability figure, which still needs its own 8-dataset
refresh per E0's task list. Results: sub-linear exponents across a genuine
1000x cell-count range (build_time_s: 0.50/0.54, index_size_mb: 0.37/0.35,
breast/lymph_node respectively) -- a much stronger claim than the previous
~4.3x real-dataset-only range. Full numbers in paper/results_tracking.md,
including a caveat about a visible kink at the subsample/bootstrap method
transition (~1e5-5e5).
```

### E7 — Noise robustness → end-to-end Recall@epsilon

```
Repo: /home/NAShome/ghoss18/spindle_dev (server). src/spindle_dev/metrics.py
has add_spd_noise(spd, noise_level, seed) but no benchmark script uses it.
This replaces the removed niche-assignment-accuracy claim (E0 item 6) with a
meaningful end-to-end robustness check. Write benchmarks/noise_robustness.py
that: for each of the 8 datasets, takes the existing holdout query set,
applies add_spd_noise() to each query SPD at increasing noise_level (e.g.
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), reruns the full search (all niches,
budget_multiplier=1.0) against the unperturbed index, and records
recall_at_eps_{0.1,0.5} at each noise level. Save to
results/noise_robustness/<dataset>_noise_sweep.csv and produce a figure:
recall_at_eps vs. noise_level, one line/panel per dataset. Report at what
noise level recall meaningfully degrades, per dataset. Run big steps via SLURM.
```

### E8 — Environment logging

```
Repo: /home/NAShome/ghoss18/spindle_dev. results/run_logs/*.json (from the
completed budget sweep) already has hostname/cpu_model/cpu_count/
total_ram_gb per dataset run -- read a couple of these to confirm the
hardware is consistent across the run, then also capture: OS/kernel version
(uname -a), Python version, and numpy/scipy/scikit-learn versions actually
installed (pip freeze inside the spindle_env conda env). Confirm no
GPU-dependent code path exists. Write a short "Computational environment"
paragraph (3-5 sentences) for the Methods section or an Additional File.
```

### E9 — Systematic enrichment quantification across all datasets

```
Repo: /home/NAShome/ghoss18/spindle_dev (local or server).
benchmarks/gene_signature_search.py currently only benchmarks breast cancer,
and still uses the old single-niche-routing search pattern -- confirm
whether it needs the same search-all-niches update as
holdout_validation.py before trusting its per-niche top-K retrieval. Extend
to run across all 6 tissue datasets referenced in the manuscript (skin,
kidney, breast, lung, pancreas, lymph node) by: (1) sourcing a curated
pathway/gene-set list per tissue type (document source per tissue), (2) for
every niche discovered by Spindle in each dataset, computing pathway
enrichment score for Spindle's top-K matches vs. background, (3) running a
Mann-Whitney U test (or Welch's t-test) + effect size (Cliff's delta or
Cohen's d), (4) reporting, per dataset, the fraction of niches achieving
FDR-corrected p<0.05 for at least one curated pathway. Save to
results/enrichment_quantification/<dataset>_niche_enrichment_stats.csv and a
summary table.
```

### E10 — ARI/NMI vs. external annotations, all datasets

```
Repo: /home/NAShome/ghoss18/spindle_dev (local). Currently only breast cancer
has been validated against external published spatial cluster/cell-type
annotations. Once external cell-type/cluster annotation files are obtained
for kidney, lung, pancreas, skin, and lymph node, write a new script (e.g.
benchmarks/niche_concordance.py) that, for each dataset, loads Spindle's
assigned niche/cluster labels (ProcessedData.assign_label_to_spots() output
-- locate the existing breast-cancer comparison code first, near the
breast_enrichment.pdf/niche_to_celltype.pdf figures) and the external
annotation labels for the same cells/spots, and computes ARI and NMI between
the two labelings. Save to results/niche_concordance/<dataset>_ari_nmi.csv
and a summary table across all 6 datasets. Report ARI/NMI per dataset and
flag any dataset where concordance is notably weaker than breast cancer's.
```

### E0 — Numeric reconciliation (run last, Stage G)

```
Repo: /home/NAShome/ghoss18/spindle_dev (paper/ has the manuscript). Run
this only after Stages A-F (E5, E11, E12, E2-E4, E6, E9, E10, E7) are all
locked in. Read EXPERIMENT_PLAN.md's "Current state of the method" and
"Decisions confirmed with user" sections first, plus the accumulated
paper/results_tracking.md -- these are the ground truth this task writes
into the manuscript. Do NOT rerun any benchmark -- pull every number from
already-produced CSVs/JSONs, primarily:
  - results/multiseed_holdout/summary.csv (E5) for per-dataset
    recall_at_eps_{0.1,0.25,0.5,1.0}, overlap_at_eps_{0.1,0.25,0.5,1.0},
    mean_speedup, mean_spindle_time_ms, with error bars (mean±s.d.).
  - results/run_logs/*_index_build_run_log.json for wall_time_s/peak_rss_gb,
    plus each *_spindle_index.pkl's own index_size_mb field (pickle.load it).
  - results/multiseed_partial_panel_search/summary.csv (E11),
    results/cross_modal_search/summary.csv (E12), results/ablation_block_diag/summary.csv,
    results/leiden_resolution_sweep/, results/faiss_baseline/,
    results/scalability_sweep/, results/enrichment_quantification/,
    results/niche_concordance/, results/noise_robustness/ (E2-E4,E6,E7,E9,E10)
    for their respective new tables/figures.

In main_genome_biology.tex:
1. Fix every "six"/"seven datasets" occurrence to "eight" (Background, Results
   intro, Figure 1 caption, Results S3.5 closing line).
2. Rewrite Results S3.1 (Scalability): 8 rows/numbers sourced from run_logs +
   index pickles, plus the E6 scalability-sweep figure. Delete the
   commented-out tab:scalability table instead of reconciling it against
   tab:xenium_summary -- one source of truth now.
3. tab:xenium_summary: add lymph_node_5k as an 8th row; refresh all 7
   existing rows from the same fresh source, with mean±s.d. from E5.
4. Rewrite Results S3.2 (Accuracy): replace "exact brute-force"/"exhaustive
   pairwise" ground-truth framing with: ground truth = exact
   block-diagonalized Frobenius distance (cite Proposition 1 -- the LE
   distance decomposes exactly under the block-diagonal assumption, so this
   isn't an approximate substitute for "the real answer"). Report
   recall_at_eps_{0.1,0.5} and overlap_at_eps_{0.5,1.0} at
   budget_multiplier=1.0 with error bars from E5. Drop the Recall@1/2/10
   rank-bucket framing (built on the exhaustive-search premise). Add
   fig_budget_sweep_block.pdf as an in-text figure in S3.1 or S3.2.
5. Methods "Index Searching" section + its Algorithm ("searchIdx"): delete
   the kNN-consensus single-niche-selection step entirely; replace with
   "search every niche's DAG (each with its own permutation/budget), merge
   Stage-1 candidates, then Stage-2 re-rank" (the actual code behavior, and
   now also true of partial-panel and cross-modal search per E11/E12). The
   DFSBranchAndBound algorithm and its cost-bound theorem are written
   per-niche -- add a clarifying sentence (bound applies per searched niche;
   total cost sums/maxes over every niche searched) rather than rewriting
   the proof.
6. Remove or reframe the niche_assignment_performance.pdf figure + caption
   (">90% accuracy at noise=0.5") -- the concept (which single niche a query
   routes to) is moot once every niche is searched. Replace with E7's
   end-to-end recall_at_eps vs. noise figure.
7. Add new sections/tables for E2 (block-diag ablation), E3 (Leiden
   resolution), E4 (FAISS/HNSW baseline), E9 (enrichment quantification),
   E10 (ARI/NMI) -- each per its own task prompt's output above.
8. Add a Discussion paragraph (currently "% TODO") arguing why
   block-diagonalized-only reporting is the right choice -- cite the
   Methods' own block-diagonal assumption + Proposition 1, and
   figures/fig_budget_sweep_whole.png + fig_epsilon_scale.png as empirical
   support (whole-matrix recall plateaus far below 1.0 for a comparison the
   method never claims to satisfy).
9. Add the "Computational environment" paragraph (E8).
9b. Cross-modal subsection (Results ~L296-320, Fig 1E caption, fig:cross_modal caption):
    166/157 tiles -> 165/156; replace Recall@1 100.0%/84.0% and top-5 overlap
    100.0%/80.4% with results/cross_modal_search/summary.csv (recall_at_eps_{0.1,0.5},
    overlap_at_eps_{0.5,1.0}, mean±s.d., 5 seeds); drop the Recall@K/Overlap@K framing;
    delete the commented-out tab:cross_modal. Methods: describe the global
    tangent-space bias correction. Discussion/Limitations: cross-modal search needs a
    reference sample of the query platform to estimate the correction (batch-level,
    not per-query) -- see paper/results_tracking.md E12.
10. Fix any figure caption that describes the wrong tissue/dataset (check
    each \includegraphics caption against what the figure actually shows).

Report every number you changed, old value -> new value, with the source
CSV/JSON and column for each.
```

## Progress Tracking

| ID | Status | Notes |
|----|--------|-------|
| E1 | Done | `figures/fig_budget_sweep_block.{pdf,png}` for the manuscript; `fig_budget_sweep_whole.{pdf,png}` + `fig_epsilon_scale.{pdf,png}` for the Discussion argument. |
| E5 | Done | `results/multiseed_holdout/summary.csv`; mean±s.d. across seeds 0-4, `--train-test-ratio 0.10`, `budget_multiplier=1.0`. See `paper/results_tracking.md`. |
| E11 | Done | Stage B — search was already all-niches; real fixes were query-construction (drop `assign_clusters_to_new_spds`, length-targeted binned draws via `build_binned_queries`), epsilon-band rescaling, and metrics. Pushed from `--num-queries 5` to `50` (user request) — revealed recall is not uniformly 1.0 and overlap-vs-length is an inverted U, not monotonic; n=5 numbers superseded. Final: 4 length bins (`<=6/7-12/13-16/>16`), `recall_at_eps_{0.05,0.1}`/`overlap_at_eps_{0.1,0.25}` (several alternative metrics tried and reverted — see `paper/results_tracking.md`'s "Methodology dead ends"). New `fig_partial_panel_search_by_case.{pdf,png}`: contiguous vs. non-contiguous gene panels perform indistinguishably. Brain-cancer's lower overlap investigated and attributed to genuine near-set-size effects, not a bug (search `rank` ≈ 1 throughout). Multi-block-query accuracy gap documented as a discussion-only limitation (not fixed). |
| E12 | Done | Stage C — all-niche search + holdout GT/metrics, production index pipeline, `--seed`/`RunLogger`, data at `dataset/cross_modal/`. Bias correction switched to one **global** whole-matrix tangent-space shift (user decision: the per-niche all-query variant collapsed true-NN onto one niche, v2x recall 0.24 on seed 0). Seeds 0–4: x2v recall@ε0.1 1.000±0.000, v2x 0.956±0.033; all-niche ≥ single-niche routing on seed 0 (x2v 1.00 vs 0.00, v2x 0.98 vs 0.68). Tiles now 165/156 (low-density filter). Discussion point recorded: the correction is batch-level, so cross-modal search needs a reference dataset representative of the query tile's distribution. See `paper/results_tracking.md`. |
| E2 | Not started | Stage D (parallel with E3, E4, E6). |
| E3 | Not started | Stage D (parallel with E2, E4, E6). |
| E4 | Not started | Stage D (parallel with E2, E3, E6). |
| E6 | Done | Run out of sequence (Stage D, doesn't depend on B/C). See `paper/results_tracking.md` — sub-linear exponents (build_time_s: 0.50/0.54, index_size_mb: 0.37/0.35) across a genuine 1000x cell-count range. |
| E9 | Not started | Stage E (parallel with E10). Check `gene_signature_search.py` against the search-all-niches fix first. |
| E10 | Not started | Stage E (parallel with E9). Blocked on annotation files. |
| E7 | Not started | Stage F — replaces the removed niche-assignment claim. |
| E0 | Not started | Stage G — run last, once A–F are locked in. |
| E8 | Mostly done | Cross-cutting `RunLogger` requirement baked into every stage's task prompt; final writeup happens alongside E0. |
