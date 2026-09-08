# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 40 | 100.0 | 100.0 | 14.5 | 14.2 | 13.2 | 1.0 | 19.03 | 101.82 | 5.35 |
| 7-12 genes | 91 | 97.8 | 97.8 | 28.6 | 27.2 | 24.8 | 1.1 | 29.02 | 116.59 | 4.02 |
| 13-16 genes | 72 | 98.6 | 98.6 | 37.2 | 38.0 | 32.1 | 1.0 | 39.35 | 140.13 | 3.56 |
| >16 genes | 47 | 100.0 | 100.0 | 68.5 | 65.7 | 58.0 | 1.0 | 49.52 | 188.12 | 3.80 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 22 | 100.0 | 100.0 | 6.4 | 8.2 | 6.5 | 1.0 | 19.85 | 102.76 | 5.18 |
| 7-12 genes | 118 | 100.0 | 100.0 | 16.5 | 17.1 | 15.9 | 1.0 | 29.59 | 117.11 | 3.96 |
| 13-16 genes | 67 | 100.0 | 100.0 | 44.8 | 43.4 | 37.8 | 1.0 | 39.86 | 140.67 | 3.53 |
| >16 genes | 43 | 100.0 | 100.0 | 61.4 | 57.6 | 49.3 | 1.0 | 48.53 | 175.35 | 3.61 |

