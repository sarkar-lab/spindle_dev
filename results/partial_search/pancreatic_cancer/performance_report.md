# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 80.0 | 80.0 | 44.0 | 32.0 | 19.6 | 1.4 | 8.83 | 51.88 | 5.87 |
| 13-16 genes | 9 | 77.8 | 77.8 | 81.1 | 72.2 | 59.8 | 1.4 | 13.72 | 69.67 | 5.08 |
| >16 genes | 11 | 100.0 | 100.0 | 83.6 | 70.9 | 58.0 | 1.0 | 18.24 | 95.24 | 5.22 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 5.0 | 15.0 | 17.0 | 1.0 | 8.10 | 49.22 | 6.07 |
| 7-12 genes | 4 | 100.0 | 100.0 | 65.0 | 65.0 | 47.0 | 1.0 | 12.30 | 54.58 | 4.44 |
| 13-16 genes | 7 | 100.0 | 100.0 | 90.0 | 76.4 | 62.9 | 1.0 | 15.66 | 69.24 | 4.42 |
| >16 genes | 12 | 100.0 | 100.0 | 91.7 | 85.0 | 61.5 | 1.0 | 18.17 | 90.59 | 4.98 |

