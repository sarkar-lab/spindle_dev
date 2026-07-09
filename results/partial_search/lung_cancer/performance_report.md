# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 100.0 | 100.0 | 90.0 | 72.5 | 38.0 | 1.0 | 8.91 | 50.60 | 5.68 |
| 7-12 genes | 6 | 83.3 | 83.3 | 75.0 | 71.7 | 37.3 | 1.2 | 11.87 | 62.15 | 5.24 |
| 13-16 genes | 4 | 100.0 | 100.0 | 72.5 | 72.5 | 34.0 | 1.0 | 13.93 | 66.91 | 4.80 |
| >16 genes | 13 | 100.0 | 100.0 | 91.5 | 72.7 | 34.8 | 1.0 | 18.35 | 97.44 | 5.31 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 2 | 0.0 | 0.0 | 45.0 | 47.5 | 37.0 | 2.0 | 7.33 | 43.91 | 5.99 |
| 7-12 genes | 10 | 90.0 | 90.0 | 83.0 | 66.5 | 34.6 | 1.3 | 11.54 | 59.65 | 5.17 |
| 13-16 genes | 6 | 100.0 | 100.0 | 88.3 | 72.5 | 35.0 | 1.0 | 14.77 | 78.40 | 5.31 |
| >16 genes | 7 | 100.0 | 100.0 | 88.6 | 73.6 | 35.7 | 1.0 | 17.89 | 89.59 | 5.01 |

