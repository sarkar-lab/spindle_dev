# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 50.0 | 50.0 | 15.0 | 17.5 | 25.0 | 2.0 | 7.53 | 39.99 | 5.31 |
| 7-12 genes | 10 | 100.0 | 100.0 | 50.0 | 51.0 | 51.4 | 1.0 | 11.23 | 49.74 | 4.43 |
| 13-16 genes | 3 | 100.0 | 100.0 | 70.0 | 75.0 | 64.7 | 1.0 | 14.42 | 64.05 | 4.44 |
| >16 genes | 10 | 100.0 | 100.0 | 98.0 | 94.5 | 73.6 | 1.0 | 24.24 | 111.53 | 4.60 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 60.0 | 50.0 | 58.0 | 1.0 | 7.26 | 38.17 | 5.26 |
| 7-12 genes | 7 | 100.0 | 100.0 | 47.1 | 48.6 | 53.1 | 1.0 | 11.71 | 49.79 | 4.25 |
| 13-16 genes | 11 | 100.0 | 100.0 | 89.1 | 87.7 | 69.5 | 1.0 | 15.42 | 62.81 | 4.07 |
| >16 genes | 6 | 100.0 | 100.0 | 91.7 | 80.0 | 66.0 | 1.0 | 19.73 | 84.26 | 4.27 |

