# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 41 | 73.2 | 73.2 | 59.8 | 58.5 | 50.7 | 2.0 | 22.59 | 90.33 | 4.00 |
| 7-12 genes | 101 | 78.2 | 78.2 | 74.6 | 71.0 | 58.8 | 6.8 | 34.85 | 126.28 | 3.62 |
| 13-16 genes | 60 | 90.0 | 90.0 | 85.8 | 83.7 | 70.7 | 1.2 | 48.21 | 179.62 | 3.73 |
| >16 genes | 48 | 91.7 | 91.7 | 91.9 | 89.7 | 69.3 | 1.1 | 60.96 | 299.19 | 4.91 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 31 | 54.8 | 54.8 | 50.3 | 43.5 | 36.3 | 5.5 | 23.67 | 90.02 | 3.80 |
| 7-12 genes | 110 | 80.0 | 80.0 | 74.6 | 71.8 | 60.5 | 1.4 | 35.81 | 123.03 | 3.44 |
| 13-16 genes | 62 | 88.7 | 88.7 | 85.8 | 83.6 | 69.1 | 1.1 | 47.67 | 176.42 | 3.70 |
| >16 genes | 47 | 89.4 | 89.4 | 91.1 | 86.6 | 68.2 | 1.1 | 59.54 | 266.19 | 4.47 |

