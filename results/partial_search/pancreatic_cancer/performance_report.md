# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 4 | 25.0 | 25.0 | 25.0 | 21.2 | 16.5 | 7.0 | 34.53 | 170.76 | 4.95 |
| 13-16 genes | 6 | 66.7 | 66.7 | 38.3 | 36.7 | 25.3 | 2.3 | 45.11 | 340.64 | 7.55 |
| >16 genes | 15 | 100.0 | 100.0 | 70.7 | 54.0 | 30.7 | 1.0 | 86.94 | 861.98 | 9.92 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 20.0 | 10.0 | 13.0 | 1.0 | 28.36 | 289.34 | 10.20 |
| 7-12 genes | 10 | 90.0 | 90.0 | 66.0 | 54.0 | 27.8 | 1.1 | 40.93 | 242.17 | 5.92 |
| 13-16 genes | 7 | 85.7 | 85.7 | 82.9 | 67.9 | 34.9 | 1.1 | 48.74 | 277.26 | 5.69 |
| >16 genes | 6 | 100.0 | 100.0 | 86.7 | 67.5 | 30.7 | 1.0 | 59.54 | 426.53 | 7.16 |

