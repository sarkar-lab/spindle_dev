# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 60.0 | 62.5 | 58.0 | 1.0 | 5.38 | 20.23 | 3.76 |
| 7-12 genes | 19 | 100.0 | 100.0 | 77.9 | 75.0 | 77.5 | 1.0 | 6.73 | 25.30 | 3.76 |
| 13-16 genes | 9 | 100.0 | 100.0 | 86.7 | 84.4 | 81.1 | 1.0 | 9.59 | 32.89 | 3.43 |
| >16 genes | 20 | 100.0 | 100.0 | 85.5 | 86.5 | 66.1 | 1.0 | 12.54 | 51.55 | 4.11 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 3 | 100.0 | 100.0 | 76.7 | 75.0 | 60.7 | 1.0 | 5.11 | 20.18 | 3.95 |
| 7-12 genes | 19 | 100.0 | 100.0 | 66.3 | 68.2 | 66.5 | 1.0 | 8.18 | 24.93 | 3.05 |
| 13-16 genes | 5 | 100.0 | 100.0 | 78.0 | 74.0 | 83.2 | 1.0 | 10.12 | 33.44 | 3.31 |
| >16 genes | 23 | 100.0 | 100.0 | 85.7 | 91.3 | 78.8 | 1.0 | 12.52 | 42.43 | 3.39 |

