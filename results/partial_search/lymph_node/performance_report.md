# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 6 | 50.0 | 50.0 | 45.0 | 62.5 | 35.3 | 3.0 | 12.53 | 71.45 | 5.70 |
| 7-12 genes | 12 | 91.7 | 91.7 | 51.7 | 55.8 | 31.8 | 1.5 | 21.95 | 183.52 | 8.36 |
| 13-16 genes | 6 | 100.0 | 100.0 | 71.7 | 76.7 | 36.7 | 1.0 | 27.08 | 315.12 | 11.64 |
| >16 genes | 1 | 100.0 | 100.0 | 90.0 | 80.0 | 40.0 | 1.0 | 25.22 | 134.44 | 5.33 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 30.0 | 25.0 | 19.0 | 1.0 | 10.28 | 67.07 | 6.52 |
| 7-12 genes | 15 | 93.3 | 93.3 | 78.0 | 71.3 | 36.4 | 1.5 | 16.57 | 91.03 | 5.49 |
| 13-16 genes | 7 | 85.7 | 85.7 | 68.6 | 85.0 | 38.6 | 1.1 | 24.29 | 123.29 | 5.08 |
| >16 genes | 1 | 100.0 | 100.0 | 90.0 | 70.0 | 36.0 | 1.0 | 31.42 | 224.86 | 7.16 |

