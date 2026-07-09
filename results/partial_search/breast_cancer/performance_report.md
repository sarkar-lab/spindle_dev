# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 50.0 | 50.0 | 10.0 | 5.0 | 12.0 | 2.0 | 7.97 | 49.77 | 6.25 |
| 7-12 genes | 10 | 80.0 | 80.0 | 42.0 | 38.0 | 26.0 | 1.2 | 12.05 | 64.50 | 5.35 |
| 13-16 genes | 3 | 100.0 | 100.0 | 66.7 | 60.0 | 30.0 | 1.0 | 16.64 | 77.38 | 4.65 |
| >16 genes | 10 | 100.0 | 100.0 | 85.0 | 70.5 | 36.0 | 1.0 | 26.21 | 142.83 | 5.45 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 60.0 | 35.0 | 40.0 | 1.0 | 7.91 | 50.62 | 6.40 |
| 7-12 genes | 7 | 85.7 | 85.7 | 35.7 | 35.0 | 22.6 | 1.4 | 13.61 | 67.78 | 4.98 |
| 13-16 genes | 11 | 100.0 | 100.0 | 80.0 | 67.7 | 34.0 | 1.0 | 17.64 | 81.25 | 4.61 |
| >16 genes | 6 | 100.0 | 100.0 | 90.0 | 68.3 | 35.3 | 1.0 | 21.75 | 103.36 | 4.75 |

