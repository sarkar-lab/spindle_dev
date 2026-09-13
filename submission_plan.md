# Plan 1: Tasks & Timeline

## Experiment Inventory (each experiment runs entirely on one venue)

| ID | Experiment | Venue | New code needed | Output → Figure/Table | Addresses |
|----|---|---|---|---|---|
| E0 | **Numeric reconciliation**: rerun the canonical pipeline (`index_datasets.py` → `holdout_validation.py` → `organize_panel_data.py` → `generate_result_figure.py`) end-to-end for all 8 datasets; treat `panel_data/*.csv` as single source of truth; remove stale commented tables; update every prose number and `tab:xenium_summary` | **Local** (all 8 datasets already ran locally to produce the current CSVs — known-feasible) | None — orchestration only | Corrected Results prose + Table `tab:xenium_summary` + Fig. 1 panels A/B/D | Speedup discrepancy (8.94–16.97× vs 5.83–9.93× vs actual 1.68–7.82×); build-time/index-size mismatch |
| E1 | **Recall-vs-budget tradeoff curve**: run `budget_sweep_holdout.py` across all 8 datasets; write `scripts/generate_budget_sweep_figure.py` | **Server** (sweep + effort-widening across 8 datasets, including two large ones, is long-running) | New figure script | New Fig. "Recall–speedup tradeoff" | Missing recall-vs-budget/QPS curve |
| E2 | **Ablation: block-diagonalization on/off** (breast, lung, pancreas; single seed each) | **Local** (3 datasets, one extra build per dataset — modest) | Small harness looping `index_datasets.py` with `block_diagonalize={True,False}` + comparison script | New ablation table/figure | No ablations of design choices |
| E3 | **Ablation: Leiden resolution sweep** (0.3/0.5/0.8/1.2/1.6 × 3 datasets = 15 full index rebuilds + benchmarks) | **Server** | New sweep script (index-build-level) | New ablation figure: recall/speedup/#niches vs. resolution | No sweep of niche granularity |
| E4 | **Baseline: FAISS/HNSW over flattened covariance features**, all 8 datasets | **Server** (kept as one experiment across all datasets, including two large/memory-heavy ones) | `benchmarks/faiss_baseline.py` (new); add `faiss-cpu`/`hnswlib` to `requirements.txt` | New comparison table/figure: Spindle vs. generic ANN | "Only comparator is brute force" |
| E5 | **Multi-seed reruns for error bars** (≥5 seeds × 8 datasets = 40 runs) | **Server** | Thin wrapper around `holdout_validation.py` looping `--seed` | Mean±s.d. error bars on every headline number | Single-run point estimates, no error bars |
| E6 | **Scalability sweep across 2–3 orders of magnitude** (synthetic subsample/replicate to ~1K–1M cells) | **Server** (long-running, large synthetic runs) | `benchmarks/scalability_sweep.py` (new) | New/extended scalability figure | Thin scalability claim (<5× range across 6 real datasets) |
| E7 | **Noise robustness → end-to-end Recall@1/10** (perturb query SPDs at increasing noise, no index rebuild needed, all 8 datasets) | **Local** (lighter weight — search-time only, no rebuild) | New script built on `add_spd_noise()` | New/replacement figure for `niche_assignment_performance.pdf`, reframed as recall degradation | Noise-robustness claim currently niche-assignment-only, in appendix |
| E8 | **Environment logging** (CPU/RAM/OS/Python/NumPy versions for the server run that produces final timing numbers) | **Local** capture step, zero compute | None | "Computational environment" paragraph | No hardware/software environment reported |
| E9 | **Systematic enrichment quantification across all 6 tissue datasets** + Mann-Whitney/t-test + effect size on pathway scores | **Local** (cheap compute; needs curated per-tissue gene sets for kidney/lung/pancreas/skin/lymph) | Extend `benchmarks/gene_signature_search.py` | New summary table + effect-size column | One hand-picked module per tissue, no stats test |
| E10 | **ARI/NMI vs. external annotations, all datasets** | **Local** (cheap compute) — blocked on user obtaining annotation files first | New comparison script | New table: ARI/NMI per dataset | Concordance validated for breast only |

Not in this plan (per user decision): directly benchmarking SOAR/STomicsDB/geneSCOPE — instead, once E4 lands, soften the Background critique to a qualitative capability contrast.

## Sequencing

**Phase 1 (Week 1) — Reconcile + kick off long server jobs**
1. E0: full local pipeline rerun, all numbers reconciled, stale commented tables removed.
2. In parallel: add FAISS/HNSW dependency; confirm the `results/`→`figures/` copy step for `fig_main_result.pdf`; user begins gathering external annotation files for E10.
3. Kick off the longest server jobs immediately, since they depend only on E0's index-build code being correct: **E3 (Leiden sweep)**, **E5 (multi-seed reruns)**, **E6 (scalability sweep)**, **E1 (budget sweep)**, **E4 (FAISS baseline)** — all server, can run concurrently if server has capacity, or queued.
4. E8: capture environment info from whichever machine (server) produces the final timing numbers.
5. Locally: E2 (block-diag ablation), E7 (noise robustness) — start immediately, don't depend on server jobs.

**Phase 2 (Weeks 2–3) — Continue local work, monitor server jobs**
6. E9 (enrichment quantification + stats test) — local, start once per-tissue gene sets assembled.
7. E10 (ARI/NMI extension) — local, start once annotation files are in hand.
8. Server jobs (E1, E3, E4, E5, E6) continue/complete.

**Phase 3 (Week 3–4) — Collect + build figures/tables**
9. Pull in all server outputs; build every new figure/table (recall-budget curve, ablation figures, baseline comparison, scalability curve, noise-robustness curve, ARI/NMI table, enrichment table).
10. Cross-check every number that will appear in the manuscript against its source CSV — no hand-typed numbers.

**Phase 4 (Week 4–5) — Writing pass (no new experiments)**
11. Discussion + Conclusions written from actual E2–E10 findings.
12. Fill Declarations (fix placeholder email/affiliation).
13. Rewrite Abstract in Genome Biology's unstructured-paragraph style.
14. Restructure figures (Additional File split, palette/ordering/plot-type unification, scale bars, vectorize DAG Sankey, caption convention).
15. Soften SOAR/STomicsDB Background critique referencing the FAISS baseline framing.
16. Full typo/grammar pass.

**Phase 5 (Week 5–6) — Internal review + submit**
17. Co-author read-through, resolve comments.
18. Final proofread, compile Additional File 1, submit.

## Timeline summary

| Week | Focus |
|---|---|
| 1 | E0 reconciliation (local); kick off E1/E3/E4/E5/E6 on server; E2/E7 local; start E10 data gathering |
| 2–3 | E9/E10 local execution; server jobs continue |
| 3–4 | Collect all results, build final figures/tables, cross-check numbers |
| 4–5 | Discussion/Conclusions/Declarations/Abstract rewrite, figure reorg, Background softening, typo pass |
| 5–6 | Internal review, final compile, submit |
