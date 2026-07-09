# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 70.0 | 38.0 | 1.0 | 5.26 | 24.62 | 4.68 |
| 7-12 genes | 13 | 100.0 | 100.0 | 58.5 | 50.0 | 24.2 | 1.0 | 8.40 | 28.96 | 3.45 |
| 13-16 genes | 5 | 100.0 | 100.0 | 98.0 | 80.0 | 34.8 | 1.0 | 10.61 | 35.37 | 3.33 |
| >16 genes | 6 | 100.0 | 100.0 | 91.7 | 74.2 | 37.0 | 1.0 | 13.74 | 51.80 | 3.77 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 4 | 100.0 | 100.0 | 52.5 | 51.2 | 30.0 | 1.0 | 6.87 | 23.54 | 3.43 |
| 7-12 genes | 12 | 100.0 | 100.0 | 42.5 | 40.4 | 26.3 | 1.0 | 8.50 | 29.74 | 3.50 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 81.7 | 36.7 | 1.0 | 10.45 | 35.86 | 3.43 |
| >16 genes | 3 | 100.0 | 100.0 | 83.3 | 76.7 | 37.3 | 1.0 | 13.52 | 46.63 | 3.45 |

