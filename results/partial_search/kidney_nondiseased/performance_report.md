# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 100.0 | 100.0 | 54.0 | 53.0 | 32.8 | 1.0 | 4.69 | 23.87 | 5.09 |
| 13-16 genes | 6 | 100.0 | 100.0 | 81.7 | 71.7 | 38.7 | 1.0 | 6.15 | 30.07 | 4.89 |
| >16 genes | 14 | 100.0 | 100.0 | 66.4 | 62.1 | 34.6 | 1.0 | 7.96 | 49.65 | 6.24 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 0.0 | 0.0 | 30.0 | 40.0 | 38.0 | 2.0 | 3.94 | 19.31 | 4.90 |
| 7-12 genes | 11 | 90.9 | 90.9 | 68.2 | 64.5 | 35.8 | 1.2 | 4.92 | 24.08 | 4.89 |
| 13-16 genes | 3 | 100.0 | 100.0 | 70.0 | 73.3 | 36.0 | 1.0 | 6.06 | 29.20 | 4.82 |
| >16 genes | 10 | 100.0 | 100.0 | 79.0 | 63.0 | 36.4 | 1.0 | 8.16 | 40.00 | 4.90 |

