# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 80.0 | 78.0 | 1.0 | 6.35 | 20.36 | 3.21 |
| 7-12 genes | 13 | 100.0 | 100.0 | 63.8 | 60.0 | 64.5 | 1.0 | 8.90 | 23.53 | 2.64 |
| 13-16 genes | 7 | 100.0 | 100.0 | 80.0 | 72.1 | 68.0 | 1.0 | 11.73 | 30.18 | 2.57 |
| >16 genes | 4 | 100.0 | 100.0 | 90.0 | 100.0 | 79.5 | 1.0 | 14.44 | 39.38 | 2.73 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 0.0 | 0.0 | 70.0 | 85.0 | 86.0 | 2.0 | 5.72 | 18.60 | 3.25 |
| 7-12 genes | 12 | 100.0 | 100.0 | 71.7 | 69.6 | 60.3 | 1.0 | 8.78 | 23.09 | 2.63 |
| 13-16 genes | 7 | 100.0 | 100.0 | 75.7 | 71.4 | 67.7 | 1.0 | 11.19 | 28.53 | 2.55 |
| >16 genes | 5 | 100.0 | 100.0 | 94.0 | 99.0 | 82.4 | 1.0 | 14.17 | 37.91 | 2.67 |

