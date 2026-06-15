# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 24 | 75.0 | 75.0 | 54.6 | 41.2 | 23.1 | 2.8 | 4.32 | 28.06 |
| 7-12 genes | 82 | 95.1 | 95.1 | 77.0 | 58.5 | 27.7 | 1.1 | 4.04 | 27.95 |
| 13-16 genes | 60 | 93.3 | 93.3 | 75.0 | 55.0 | 26.7 | 1.1 | 3.83 | 35.65 |
| >16 genes | 34 | 100.0 | 100.0 | 77.6 | 53.2 | 26.4 | 1.0 | 1.57 | 26.31 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 19 | 84.2 | 84.2 | 60.5 | 50.8 | 27.4 | 1.3 | 1.19 | 21.13 |
| 7-12 genes | 108 | 95.4 | 95.4 | 75.6 | 59.1 | 27.9 | 1.1 | 4.42 | 27.67 |
| 13-16 genes | 47 | 93.6 | 93.6 | 79.4 | 58.1 | 27.1 | 1.1 | 3.71 | 32.28 |
| >16 genes | 26 | 100.0 | 100.0 | 83.1 | 57.9 | 27.5 | 1.0 | 3.22 | 19.75 |

