# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 5 | 100.0 | 100.0 | 22.0 | 22.0 | 22.0 | 1.0 | 10.83 | 61.16 | 5.65 |
| 7-12 genes | 11 | 100.0 | 100.0 | 70.0 | 70.0 | 61.6 | 1.0 | 14.33 | 73.75 | 5.15 |
| 13-16 genes | 7 | 100.0 | 100.0 | 84.3 | 81.4 | 67.1 | 1.0 | 17.99 | 89.51 | 4.98 |
| >16 genes | 2 | 100.0 | 100.0 | 100.0 | 97.5 | 70.0 | 1.0 | 18.98 | 117.71 | 6.20 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 78.0 | 1.0 | 10.49 | 54.91 | 5.23 |
| 7-12 genes | 11 | 100.0 | 100.0 | 55.5 | 53.6 | 49.1 | 1.0 | 13.54 | 67.71 | 5.00 |
| 13-16 genes | 7 | 100.0 | 100.0 | 84.3 | 85.0 | 70.3 | 1.0 | 15.82 | 82.65 | 5.22 |
| >16 genes | 6 | 100.0 | 100.0 | 98.3 | 94.2 | 69.0 | 1.0 | 20.79 | 124.78 | 6.00 |

