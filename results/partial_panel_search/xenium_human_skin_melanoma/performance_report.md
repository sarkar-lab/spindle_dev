# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 48 | 97.9 | 97.9 | 55.8 | 55.5 | 56.4 | 1.0 | 2.93 | 8.94 | 3.05 |
| 7-12 genes | 104 | 98.1 | 98.1 | 76.3 | 76.6 | 70.3 | 1.0 | 4.24 | 10.93 | 2.58 |
| 13-16 genes | 70 | 100.0 | 100.0 | 87.9 | 84.0 | 77.5 | 1.0 | 5.35 | 13.83 | 2.58 |
| >16 genes | 28 | 100.0 | 100.0 | 84.6 | 79.3 | 76.6 | 1.0 | 6.74 | 17.88 | 2.65 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 24 | 100.0 | 100.0 | 79.2 | 79.2 | 74.7 | 1.0 | 3.35 | 9.23 | 2.75 |
| 7-12 genes | 136 | 99.3 | 99.3 | 83.8 | 81.7 | 73.7 | 1.0 | 4.44 | 10.91 | 2.46 |
| 13-16 genes | 66 | 100.0 | 100.0 | 78.9 | 77.5 | 76.7 | 1.0 | 5.57 | 14.07 | 2.53 |
| >16 genes | 24 | 100.0 | 100.0 | 81.2 | 81.5 | 78.7 | 1.0 | 6.59 | 17.41 | 2.64 |

