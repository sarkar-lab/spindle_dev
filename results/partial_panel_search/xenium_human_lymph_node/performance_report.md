# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 41 | 82.9 | 82.9 | 65.1 | 65.0 | 57.6 | 1.2 | 6.23 | 26.32 | 4.23 |
| 7-12 genes | 105 | 95.2 | 95.2 | 81.0 | 81.2 | 71.8 | 1.1 | 8.85 | 33.63 | 3.80 |
| 13-16 genes | 67 | 100.0 | 100.0 | 88.8 | 86.0 | 73.4 | 1.0 | 10.81 | 44.24 | 4.09 |
| >16 genes | 37 | 100.0 | 100.0 | 91.1 | 88.5 | 73.0 | 1.0 | 12.08 | 60.11 | 4.98 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 33 | 81.8 | 81.8 | 48.5 | 49.4 | 45.8 | 1.3 | 6.49 | 26.54 | 4.09 |
| 7-12 genes | 119 | 96.6 | 96.6 | 79.4 | 78.7 | 67.8 | 1.1 | 9.23 | 33.34 | 3.61 |
| 13-16 genes | 80 | 98.8 | 98.8 | 93.3 | 90.6 | 77.0 | 1.0 | 11.36 | 45.26 | 3.98 |
| >16 genes | 18 | 100.0 | 100.0 | 99.4 | 96.4 | 76.2 | 1.0 | 13.37 | 60.91 | 4.56 |

