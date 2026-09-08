# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 41 | 73.2 | 73.2 | 59.8 | 58.5 | 50.7 | 2.0 | 22.68 | 89.81 | 3.96 |
| 7-12 genes | 101 | 78.2 | 78.2 | 74.6 | 71.0 | 58.8 | 6.8 | 35.09 | 126.74 | 3.61 |
| 13-16 genes | 60 | 90.0 | 90.0 | 85.8 | 83.7 | 70.7 | 1.2 | 48.61 | 179.35 | 3.69 |
| >16 genes | 48 | 91.7 | 91.7 | 91.9 | 89.7 | 69.3 | 1.1 | 61.30 | 298.91 | 4.88 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 31 | 54.8 | 54.8 | 50.3 | 43.5 | 36.3 | 5.5 | 23.76 | 90.53 | 3.81 |
| 7-12 genes | 110 | 80.0 | 80.0 | 74.6 | 71.8 | 60.5 | 1.4 | 36.07 | 124.11 | 3.44 |
| 13-16 genes | 62 | 88.7 | 88.7 | 85.8 | 83.6 | 69.1 | 1.1 | 47.84 | 177.57 | 3.71 |
| >16 genes | 47 | 89.4 | 89.4 | 91.1 | 86.6 | 68.2 | 1.1 | 59.64 | 267.01 | 4.48 |

