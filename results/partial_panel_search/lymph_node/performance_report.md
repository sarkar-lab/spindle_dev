# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 5 | 100.0 | 100.0 | 22.0 | 22.0 | 22.0 | 1.0 | 8.14 | 52.19 | 6.41 |
| 7-12 genes | 11 | 100.0 | 100.0 | 70.0 | 70.0 | 61.6 | 1.0 | 13.55 | 69.99 | 5.17 |
| 13-16 genes | 7 | 100.0 | 100.0 | 84.3 | 81.4 | 67.1 | 1.0 | 16.59 | 86.87 | 5.24 |
| >16 genes | 2 | 100.0 | 100.0 | 100.0 | 97.5 | 70.0 | 1.0 | 21.70 | 115.98 | 5.35 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 78.0 | 1.0 | 10.71 | 54.34 | 5.08 |
| 7-12 genes | 11 | 100.0 | 100.0 | 55.5 | 53.6 | 49.1 | 1.0 | 13.61 | 67.39 | 4.95 |
| 13-16 genes | 7 | 100.0 | 100.0 | 84.3 | 85.0 | 70.3 | 1.0 | 15.75 | 82.79 | 5.26 |
| >16 genes | 6 | 100.0 | 100.0 | 98.3 | 94.2 | 69.0 | 1.0 | 19.58 | 119.96 | 6.13 |

