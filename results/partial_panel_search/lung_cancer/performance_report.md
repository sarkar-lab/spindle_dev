# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 90.0 | 60.0 | 1.0 | 6.84 | 35.72 | 5.22 |
| 7-12 genes | 7 | 100.0 | 100.0 | 82.9 | 83.6 | 75.1 | 1.0 | 10.42 | 44.23 | 4.24 |
| 13-16 genes | 5 | 100.0 | 100.0 | 100.0 | 100.0 | 80.4 | 1.0 | 12.60 | 52.46 | 4.16 |
| >16 genes | 12 | 100.0 | 100.0 | 96.7 | 93.7 | 68.7 | 1.0 | 18.40 | 81.60 | 4.43 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 95.0 | 87.5 | 80.0 | 1.0 | 6.53 | 34.99 | 5.35 |
| 7-12 genes | 5 | 100.0 | 100.0 | 98.0 | 94.0 | 77.2 | 1.0 | 9.76 | 45.19 | 4.63 |
| 13-16 genes | 8 | 100.0 | 100.0 | 97.5 | 93.8 | 79.5 | 1.0 | 13.05 | 55.81 | 4.28 |
| >16 genes | 10 | 90.0 | 90.0 | 90.0 | 87.5 | 73.8 | 1.1 | 16.10 | 73.17 | 4.54 |

