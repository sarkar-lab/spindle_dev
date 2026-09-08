# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 60 | 96.7 | 96.7 | 38.8 | 37.5 | 37.9 | 1.0 | 1.82 | 7.46 | 4.09 |
| 7-12 genes | 102 | 99.0 | 99.0 | 71.0 | 69.3 | 57.1 | 1.0 | 2.34 | 8.64 | 3.69 |
| 13-16 genes | 72 | 94.4 | 94.4 | 85.8 | 81.5 | 61.5 | 1.1 | 2.88 | 10.92 | 3.79 |
| >16 genes | 16 | 100.0 | 100.0 | 76.2 | 68.8 | 61.1 | 1.0 | 3.23 | 12.32 | 3.82 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 26 | 84.6 | 84.6 | 68.8 | 70.0 | 66.3 | 1.2 | 1.99 | 7.53 | 3.78 |
| 7-12 genes | 140 | 98.6 | 98.6 | 72.5 | 69.9 | 61.6 | 1.1 | 2.64 | 8.63 | 3.27 |
| 13-16 genes | 77 | 96.1 | 96.1 | 86.0 | 81.2 | 65.2 | 1.1 | 3.09 | 10.60 | 3.43 |
| >16 genes | 7 | 100.0 | 100.0 | 81.4 | 67.9 | 55.1 | 1.0 | 3.25 | 12.13 | 3.73 |

