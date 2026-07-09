# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 50.0 | 50.0 | 10.0 | 5.0 | 12.0 | 2.0 | 17.77 | 118.18 | 6.65 |
| 7-12 genes | 10 | 80.0 | 80.0 | 42.0 | 38.0 | 26.0 | 1.2 | 32.38 | 157.96 | 4.88 |
| 13-16 genes | 3 | 100.0 | 100.0 | 66.7 | 60.0 | 30.0 | 1.0 | 42.18 | 189.05 | 4.48 |
| >16 genes | 10 | 100.0 | 100.0 | 85.0 | 70.5 | 36.0 | 1.0 | 67.80 | 320.63 | 4.73 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 100.0 | 100.0 | 60.0 | 35.0 | 40.0 | 1.0 | 18.18 | 129.74 | 7.14 |
| 7-12 genes | 7 | 85.7 | 85.7 | 35.7 | 35.0 | 22.6 | 1.4 | 29.79 | 149.96 | 5.03 |
| 13-16 genes | 11 | 100.0 | 100.0 | 80.0 | 67.7 | 34.0 | 1.0 | 42.32 | 199.65 | 4.72 |
| >16 genes | 6 | 100.0 | 100.0 | 90.0 | 68.3 | 35.3 | 1.0 | 55.77 | 262.68 | 4.71 |

