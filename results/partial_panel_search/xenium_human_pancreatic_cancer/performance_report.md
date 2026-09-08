# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 11 | 90.9 | 90.9 | 16.4 | 16.8 | 14.9 | 1.3 | 4.25 | 22.66 | 5.33 |
| 7-12 genes | 81 | 95.1 | 95.1 | 40.4 | 36.5 | 29.6 | 1.0 | 5.86 | 25.48 | 4.35 |
| 13-16 genes | 52 | 94.2 | 94.2 | 65.0 | 60.2 | 50.8 | 1.1 | 7.65 | 33.39 | 4.36 |
| >16 genes | 106 | 97.2 | 97.2 | 85.6 | 81.9 | 66.8 | 1.0 | 10.04 | 47.27 | 4.71 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 12 | 100.0 | 100.0 | 22.5 | 22.5 | 20.7 | 1.0 | 4.49 | 22.37 | 4.98 |
| 7-12 genes | 72 | 98.6 | 98.6 | 53.2 | 48.3 | 39.6 | 1.0 | 6.30 | 25.61 | 4.06 |
| 13-16 genes | 57 | 96.5 | 96.5 | 64.4 | 62.1 | 52.4 | 1.0 | 8.19 | 33.78 | 4.13 |
| >16 genes | 109 | 95.4 | 95.4 | 86.5 | 81.2 | 68.8 | 1.1 | 10.01 | 44.88 | 4.48 |

