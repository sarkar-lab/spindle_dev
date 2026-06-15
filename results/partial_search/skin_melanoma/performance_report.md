# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 36 | 94.4 | 94.4 | 67.8 | 55.8 | 26.6 | 1.1 | 1.46 | 8.47 |
| 7-12 genes | 93 | 94.6 | 94.6 | 77.5 | 57.7 | 26.9 | 1.1 | 0.90 | 9.01 |
| 13-16 genes | 52 | 98.1 | 98.1 | 75.6 | 54.8 | 27.0 | 1.0 | 1.18 | 10.19 |
| >16 genes | 19 | 89.5 | 89.5 | 73.2 | 49.2 | 26.0 | 1.1 | 2.54 | 14.36 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 25 | 96.0 | 96.0 | 62.4 | 50.0 | 25.8 | 1.1 | 0.63 | 6.39 |
| 7-12 genes | 104 | 96.2 | 96.2 | 79.1 | 59.5 | 27.6 | 1.1 | 0.99 | 8.83 |
| 13-16 genes | 59 | 98.3 | 98.3 | 77.5 | 58.0 | 27.7 | 1.0 | 1.66 | 11.40 |
| >16 genes | 12 | 100.0 | 100.0 | 75.8 | 53.3 | 27.3 | 1.0 | 1.96 | 14.24 |

