# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 80.0 | 78.0 | 1.0 | 5.85 | 18.86 | 3.22 |
| 7-12 genes | 13 | 100.0 | 100.0 | 63.8 | 60.0 | 64.5 | 1.0 | 7.42 | 21.30 | 2.87 |
| 13-16 genes | 7 | 100.0 | 100.0 | 80.0 | 72.1 | 68.0 | 1.0 | 10.59 | 27.95 | 2.64 |
| >16 genes | 4 | 100.0 | 100.0 | 90.0 | 100.0 | 79.5 | 1.0 | 13.14 | 37.05 | 2.82 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 0.0 | 0.0 | 70.0 | 85.0 | 86.0 | 2.0 | 5.62 | 18.19 | 3.24 |
| 7-12 genes | 12 | 100.0 | 100.0 | 71.7 | 69.6 | 60.3 | 1.0 | 8.15 | 22.08 | 2.71 |
| 13-16 genes | 7 | 100.0 | 100.0 | 75.7 | 71.4 | 67.7 | 1.0 | 10.21 | 27.49 | 2.69 |
| >16 genes | 5 | 100.0 | 100.0 | 94.0 | 99.0 | 82.4 | 1.0 | 12.81 | 35.51 | 2.77 |

