# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 11 | 90.9 | 90.9 | 16.4 | 16.8 | 14.9 | 1.3 | 4.22 | 22.52 | 5.34 |
| 7-12 genes | 81 | 95.1 | 95.1 | 40.4 | 36.5 | 29.6 | 1.0 | 5.86 | 25.37 | 4.33 |
| 13-16 genes | 52 | 94.2 | 94.2 | 65.0 | 60.2 | 50.8 | 1.1 | 7.61 | 30.67 | 4.03 |
| >16 genes | 106 | 97.2 | 97.2 | 85.6 | 81.9 | 66.8 | 1.0 | 10.04 | 45.09 | 4.49 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 12 | 100.0 | 100.0 | 22.5 | 22.5 | 20.7 | 1.0 | 4.47 | 22.32 | 4.99 |
| 7-12 genes | 72 | 98.6 | 98.6 | 53.2 | 48.3 | 39.6 | 1.0 | 6.29 | 25.38 | 4.04 |
| 13-16 genes | 57 | 96.5 | 96.5 | 64.4 | 62.1 | 52.4 | 1.0 | 8.17 | 30.93 | 3.79 |
| >16 genes | 109 | 95.4 | 95.4 | 86.5 | 81.2 | 68.8 | 1.1 | 9.99 | 42.16 | 4.22 |

