# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 20 | 90.0 | 90.0 | 44.0 | 43.8 | 39.3 | 1.2 | 4.02 | 17.13 | 4.26 |
| 7-12 genes | 92 | 96.7 | 96.7 | 79.9 | 76.2 | 64.4 | 1.1 | 5.60 | 20.27 | 3.62 |
| 13-16 genes | 63 | 96.8 | 96.8 | 87.0 | 81.3 | 67.3 | 1.0 | 7.29 | 25.98 | 3.56 |
| >16 genes | 75 | 97.3 | 97.3 | 86.7 | 79.1 | 63.1 | 1.0 | 9.21 | 38.54 | 4.18 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 25 | 92.0 | 92.0 | 60.4 | 57.6 | 44.3 | 1.1 | 3.84 | 16.70 | 4.35 |
| 7-12 genes | 91 | 97.8 | 97.8 | 80.8 | 77.0 | 66.0 | 1.1 | 5.63 | 19.86 | 3.53 |
| 13-16 genes | 67 | 100.0 | 100.0 | 89.0 | 86.6 | 71.2 | 1.0 | 7.56 | 26.13 | 3.46 |
| >16 genes | 67 | 100.0 | 100.0 | 86.7 | 79.8 | 67.5 | 1.0 | 9.04 | 35.92 | 3.97 |

