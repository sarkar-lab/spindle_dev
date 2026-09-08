# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 40 | 100.0 | 100.0 | 14.5 | 14.2 | 13.2 | 1.0 | 18.87 | 100.83 | 5.34 |
| 7-12 genes | 91 | 97.8 | 97.8 | 28.6 | 27.2 | 24.8 | 1.1 | 28.65 | 114.55 | 4.00 |
| 13-16 genes | 72 | 98.6 | 98.6 | 37.2 | 38.0 | 32.1 | 1.0 | 38.99 | 137.08 | 3.52 |
| >16 genes | 47 | 100.0 | 100.0 | 68.5 | 65.7 | 58.0 | 1.0 | 49.03 | 186.05 | 3.80 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 22 | 100.0 | 100.0 | 6.4 | 8.2 | 6.5 | 1.0 | 19.72 | 101.13 | 5.13 |
| 7-12 genes | 118 | 100.0 | 100.0 | 16.5 | 17.1 | 15.9 | 1.0 | 29.40 | 114.91 | 3.91 |
| 13-16 genes | 67 | 100.0 | 100.0 | 44.8 | 43.4 | 37.8 | 1.0 | 39.57 | 137.35 | 3.47 |
| >16 genes | 43 | 100.0 | 100.0 | 61.4 | 57.6 | 49.3 | 1.0 | 48.29 | 172.75 | 3.58 |

