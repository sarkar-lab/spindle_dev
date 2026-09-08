# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 20 | 90.0 | 90.0 | 44.0 | 43.8 | 39.3 | 1.2 | 4.01 | 17.21 | 4.29 |
| 7-12 genes | 92 | 96.7 | 96.7 | 79.9 | 76.2 | 64.4 | 1.1 | 5.56 | 20.40 | 3.67 |
| 13-16 genes | 63 | 96.8 | 96.8 | 87.0 | 81.3 | 67.3 | 1.0 | 7.25 | 26.04 | 3.59 |
| >16 genes | 75 | 97.3 | 97.3 | 86.7 | 79.1 | 63.1 | 1.0 | 9.13 | 38.13 | 4.18 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 25 | 92.0 | 92.0 | 60.4 | 57.6 | 44.3 | 1.1 | 3.84 | 17.02 | 4.43 |
| 7-12 genes | 91 | 97.8 | 97.8 | 80.8 | 77.0 | 66.0 | 1.1 | 5.60 | 20.06 | 3.58 |
| 13-16 genes | 67 | 100.0 | 100.0 | 89.0 | 86.6 | 71.2 | 1.0 | 7.55 | 26.06 | 3.45 |
| >16 genes | 67 | 100.0 | 100.0 | 86.7 | 79.8 | 67.5 | 1.0 | 9.03 | 34.48 | 3.82 |

