# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 90.0 | 60.0 | 1.0 | 6.56 | 34.37 | 5.24 |
| 7-12 genes | 7 | 100.0 | 100.0 | 82.9 | 83.6 | 75.1 | 1.0 | 9.46 | 42.04 | 4.44 |
| 13-16 genes | 5 | 100.0 | 100.0 | 100.0 | 100.0 | 80.4 | 1.0 | 11.84 | 50.14 | 4.23 |
| >16 genes | 12 | 100.0 | 100.0 | 96.7 | 93.7 | 68.7 | 1.0 | 16.78 | 77.92 | 4.64 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 95.0 | 87.5 | 80.0 | 1.0 | 7.15 | 36.36 | 5.09 |
| 7-12 genes | 5 | 100.0 | 100.0 | 98.0 | 94.0 | 77.2 | 1.0 | 9.73 | 43.71 | 4.49 |
| 13-16 genes | 8 | 100.0 | 100.0 | 97.5 | 93.8 | 79.5 | 1.0 | 13.41 | 55.16 | 4.11 |
| >16 genes | 10 | 90.0 | 90.0 | 90.0 | 87.5 | 73.8 | 1.1 | 15.90 | 71.90 | 4.52 |

