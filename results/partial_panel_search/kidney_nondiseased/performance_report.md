# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 100.0 | 100.0 | 78.0 | 77.0 | 71.6 | 1.0 | 4.90 | 17.69 | 3.61 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 90.8 | 79.7 | 1.0 | 6.15 | 22.69 | 3.69 |
| >16 genes | 14 | 100.0 | 100.0 | 83.6 | 87.9 | 72.1 | 1.0 | 8.37 | 38.03 | 4.54 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 96.0 | 1.0 | 4.22 | 15.56 | 3.68 |
| 7-12 genes | 11 | 100.0 | 100.0 | 76.4 | 76.4 | 76.4 | 1.0 | 5.11 | 18.75 | 3.67 |
| 13-16 genes | 3 | 100.0 | 100.0 | 100.0 | 100.0 | 74.7 | 1.0 | 6.22 | 22.34 | 3.59 |
| >16 genes | 10 | 100.0 | 100.0 | 87.0 | 80.5 | 79.2 | 1.0 | 8.39 | 31.64 | 3.77 |

