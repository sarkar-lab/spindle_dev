# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 80.0 | 80.0 | 44.0 | 32.0 | 19.6 | 1.4 | 8.36 | 50.82 | 6.08 |
| 13-16 genes | 9 | 77.8 | 77.8 | 81.1 | 72.2 | 59.8 | 1.4 | 13.06 | 66.54 | 5.09 |
| >16 genes | 11 | 100.0 | 100.0 | 83.6 | 70.9 | 58.0 | 1.0 | 17.17 | 93.84 | 5.47 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 5.0 | 15.0 | 17.0 | 1.0 | 7.44 | 45.06 | 6.06 |
| 7-12 genes | 4 | 100.0 | 100.0 | 65.0 | 65.0 | 47.0 | 1.0 | 11.14 | 53.06 | 4.76 |
| 13-16 genes | 7 | 100.0 | 100.0 | 90.0 | 76.4 | 62.9 | 1.0 | 15.14 | 68.32 | 4.51 |
| >16 genes | 12 | 100.0 | 100.0 | 91.7 | 85.0 | 61.5 | 1.0 | 17.21 | 88.09 | 5.12 |

