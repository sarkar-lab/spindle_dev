# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| 7-12 genes | 5 | 100.0 | 100.0 | 54.0 | 53.0 | 32.8 | 1.0 | 9.84 | 50.88 | 5.17 |
| 13-16 genes | 6 | 100.0 | 100.0 | 81.7 | 71.7 | 38.7 | 1.0 | 12.49 | 67.06 | 5.37 |
| >16 genes | 14 | 100.0 | 100.0 | 66.4 | 62.1 | 34.6 | 1.0 | 18.05 | 102.04 | 5.65 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 1 | 0.0 | 0.0 | 30.0 | 40.0 | 38.0 | 2.0 | 9.31 | 40.26 | 4.32 |
| 7-12 genes | 11 | 90.9 | 90.9 | 68.2 | 64.5 | 35.8 | 1.2 | 11.63 | 53.72 | 4.62 |
| 13-16 genes | 3 | 100.0 | 100.0 | 70.0 | 73.3 | 36.0 | 1.0 | 12.54 | 55.21 | 4.40 |
| >16 genes | 10 | 100.0 | 100.0 | 79.0 | 63.0 | 36.4 | 1.0 | 17.35 | 78.98 | 4.55 |

