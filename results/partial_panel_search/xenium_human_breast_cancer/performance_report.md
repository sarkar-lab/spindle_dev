# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 18 | 100.0 | 100.0 | 46.1 | 41.1 | 38.4 | 1.0 | 4.80 | 19.98 | 4.17 |
| 7-12 genes | 78 | 96.2 | 96.2 | 76.8 | 75.4 | 65.8 | 1.2 | 6.82 | 24.41 | 3.58 |
| 13-16 genes | 51 | 100.0 | 100.0 | 92.4 | 89.2 | 71.8 | 1.0 | 9.50 | 32.68 | 3.44 |
| >16 genes | 103 | 100.0 | 100.0 | 95.7 | 93.3 | 70.6 | 1.0 | 12.73 | 53.79 | 4.23 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 19 | 89.5 | 89.5 | 47.4 | 44.7 | 45.5 | 1.1 | 5.20 | 19.79 | 3.81 |
| 7-12 genes | 92 | 96.7 | 96.7 | 85.1 | 82.7 | 70.8 | 1.2 | 7.60 | 24.95 | 3.28 |
| 13-16 genes | 63 | 98.4 | 98.4 | 91.0 | 86.1 | 73.0 | 1.0 | 9.85 | 33.66 | 3.42 |
| >16 genes | 76 | 100.0 | 100.0 | 95.3 | 91.1 | 71.4 | 1.0 | 12.34 | 47.79 | 3.87 |

