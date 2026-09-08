# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 48 | 97.9 | 97.9 | 55.8 | 55.5 | 56.4 | 1.0 | 2.94 | 8.90 | 3.03 |
| 7-12 genes | 104 | 98.1 | 98.1 | 76.3 | 76.6 | 70.3 | 1.0 | 4.26 | 10.85 | 2.55 |
| 13-16 genes | 70 | 100.0 | 100.0 | 87.9 | 84.0 | 77.5 | 1.0 | 5.38 | 13.75 | 2.55 |
| >16 genes | 28 | 100.0 | 100.0 | 84.6 | 79.3 | 76.6 | 1.0 | 6.76 | 17.95 | 2.66 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 24 | 100.0 | 100.0 | 79.2 | 79.2 | 74.7 | 1.0 | 3.35 | 9.15 | 2.73 |
| 7-12 genes | 136 | 99.3 | 99.3 | 83.8 | 81.7 | 73.7 | 1.0 | 4.46 | 10.79 | 2.42 |
| 13-16 genes | 66 | 100.0 | 100.0 | 78.9 | 77.5 | 76.7 | 1.0 | 5.61 | 13.93 | 2.48 |
| >16 genes | 24 | 100.0 | 100.0 | 81.2 | 81.5 | 78.7 | 1.0 | 6.59 | 17.30 | 2.63 |

