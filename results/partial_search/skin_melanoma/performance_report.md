# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 70.0 | 38.0 | 1.0 | 11.16 | 55.27 | 4.95 |
| 7-12 genes | 13 | 100.0 | 100.0 | 58.5 | 50.0 | 24.2 | 1.0 | 20.90 | 67.90 | 3.25 |
| 13-16 genes | 5 | 100.0 | 100.0 | 98.0 | 80.0 | 34.8 | 1.0 | 26.10 | 76.13 | 2.92 |
| >16 genes | 6 | 100.0 | 100.0 | 91.7 | 74.2 | 37.0 | 1.0 | 32.39 | 114.63 | 3.54 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 4 | 100.0 | 100.0 | 52.5 | 51.2 | 30.0 | 1.0 | 15.24 | 49.39 | 3.24 |
| 7-12 genes | 12 | 100.0 | 100.0 | 42.5 | 40.4 | 26.3 | 1.0 | 18.14 | 61.78 | 3.40 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 81.7 | 36.7 | 1.0 | 21.45 | 71.07 | 3.31 |
| >16 genes | 3 | 100.0 | 100.0 | 83.3 | 76.7 | 37.3 | 1.0 | 28.53 | 92.50 | 3.24 |

