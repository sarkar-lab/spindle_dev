# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 18 | 100.0 | 100.0 | 46.1 | 41.1 | 38.4 | 1.0 | 4.82 | 20.01 | 4.15 |
| 7-12 genes | 78 | 96.2 | 96.2 | 76.8 | 75.4 | 65.8 | 1.2 | 6.84 | 24.65 | 3.60 |
| 13-16 genes | 51 | 100.0 | 100.0 | 92.4 | 89.2 | 71.8 | 1.0 | 9.53 | 33.02 | 3.46 |
| >16 genes | 103 | 100.0 | 100.0 | 95.7 | 93.3 | 70.6 | 1.0 | 12.74 | 54.18 | 4.25 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 19 | 89.5 | 89.5 | 47.4 | 44.7 | 45.5 | 1.1 | 5.21 | 19.77 | 3.79 |
| 7-12 genes | 92 | 96.7 | 96.7 | 85.1 | 82.7 | 70.8 | 1.2 | 7.65 | 25.12 | 3.28 |
| 13-16 genes | 63 | 98.4 | 98.4 | 91.0 | 86.1 | 73.0 | 1.0 | 9.93 | 33.94 | 3.42 |
| >16 genes | 76 | 100.0 | 100.0 | 95.3 | 91.1 | 71.4 | 1.0 | 12.41 | 48.20 | 3.88 |

