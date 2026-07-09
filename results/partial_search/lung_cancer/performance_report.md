# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 90.0 | 72.5 | 38.0 | 1.0 | 18.87 | 105.91 | 5.61 |
| 7-12 genes | 6 | 83.3 | 83.3 | 75.0 | 71.7 | 37.3 | 1.2 | 29.33 | 148.78 | 5.07 |
| 13-16 genes | 4 | 100.0 | 100.0 | 72.5 | 72.5 | 34.0 | 1.0 | 29.30 | 138.95 | 4.74 |
| >16 genes | 13 | 100.0 | 100.0 | 91.5 | 72.7 | 34.8 | 1.0 | 43.04 | 203.22 | 4.72 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 0.0 | 0.0 | 45.0 | 47.5 | 37.0 | 2.0 | 15.49 | 100.21 | 6.47 |
| 7-12 genes | 10 | 90.0 | 90.0 | 83.0 | 66.5 | 34.6 | 1.3 | 23.00 | 117.03 | 5.09 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 72.5 | 35.0 | 1.0 | 31.35 | 163.99 | 5.23 |
| >16 genes | 7 | 100.0 | 100.0 | 88.6 | 73.6 | 35.7 | 1.0 | 39.18 | 169.45 | 4.32 |

