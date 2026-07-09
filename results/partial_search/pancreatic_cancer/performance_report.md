# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 4 | 25.0 | 25.0 | 25.0 | 21.2 | 16.5 | 7.0 | 14.53 | 78.25 | 5.38 |
| 13-16 genes | 6 | 66.7 | 66.7 | 38.3 | 36.7 | 25.3 | 2.3 | 25.35 | 211.18 | 8.33 |
| >16 genes | 15 | 100.0 | 100.0 | 70.7 | 54.0 | 30.7 | 1.0 | 34.41 | 455.58 | 13.24 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 20.0 | 10.0 | 13.0 | 1.0 | 16.44 | 425.60 | 25.88 |
| 7-12 genes | 10 | 90.0 | 90.0 | 66.0 | 54.0 | 27.8 | 1.1 | 21.06 | 194.69 | 9.24 |
| 13-16 genes | 7 | 85.7 | 85.7 | 82.9 | 67.9 | 34.9 | 1.1 | 26.46 | 305.07 | 11.53 |
| >16 genes | 6 | 100.0 | 100.0 | 86.7 | 67.5 | 30.7 | 1.0 | 30.30 | 393.17 | 12.97 |

