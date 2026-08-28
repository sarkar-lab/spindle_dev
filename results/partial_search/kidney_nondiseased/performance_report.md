# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 100.0 | 100.0 | 78.0 | 77.0 | 71.6 | 1.0 | 4.70 | 17.85 | 3.80 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 90.8 | 79.7 | 1.0 | 6.07 | 22.39 | 3.69 |
| >16 genes | 14 | 100.0 | 100.0 | 83.6 | 87.9 | 72.1 | 1.0 | 8.16 | 37.64 | 4.62 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 96.0 | 1.0 | 4.15 | 15.65 | 3.77 |
| 7-12 genes | 11 | 100.0 | 100.0 | 76.4 | 76.4 | 76.4 | 1.0 | 5.04 | 18.86 | 3.74 |
| 13-16 genes | 3 | 100.0 | 100.0 | 100.0 | 100.0 | 74.7 | 1.0 | 6.12 | 22.48 | 3.68 |
| >16 genes | 10 | 100.0 | 100.0 | 87.0 | 80.5 | 79.2 | 1.0 | 8.10 | 31.50 | 3.89 |

