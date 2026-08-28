# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 50.0 | 50.0 | 15.0 | 17.5 | 25.0 | 2.0 | 6.78 | 39.25 | 5.79 |
| 7-12 genes | 10 | 100.0 | 100.0 | 50.0 | 51.0 | 51.4 | 1.0 | 9.38 | 47.44 | 5.06 |
| 13-16 genes | 3 | 100.0 | 100.0 | 70.0 | 75.0 | 64.7 | 1.0 | 14.26 | 62.17 | 4.36 |
| >16 genes | 10 | 100.0 | 100.0 | 98.0 | 94.5 | 73.6 | 1.0 | 23.08 | 110.22 | 4.78 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 60.0 | 50.0 | 58.0 | 1.0 | 7.04 | 38.23 | 5.43 |
| 7-12 genes | 7 | 100.0 | 100.0 | 47.1 | 48.6 | 53.1 | 1.0 | 11.40 | 49.79 | 4.37 |
| 13-16 genes | 11 | 100.0 | 100.0 | 89.1 | 87.7 | 69.5 | 1.0 | 15.54 | 63.54 | 4.09 |
| >16 genes | 6 | 100.0 | 100.0 | 91.7 | 80.0 | 66.0 | 1.0 | 19.29 | 86.99 | 4.51 |

