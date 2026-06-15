# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 45 | 86.7 | 86.7 | 21.3 | 25.8 | 18.0 | 1.3 | 4.72 | 45.83 |
| 7-12 genes | 105 | 84.8 | 84.8 | 33.2 | 32.0 | 21.5 | 1.4 | 7.16 | 57.78 |
| 13-16 genes | 50 | 82.0 | 82.0 | 37.6 | 33.9 | 20.0 | 1.5 | 7.05 | 69.51 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 35 | 88.6 | 88.6 | 25.4 | 25.4 | 16.5 | 1.3 | 3.32 | 42.18 |
| 7-12 genes | 131 | 85.5 | 85.5 | 29.3 | 27.4 | 19.4 | 1.4 | 5.72 | 53.84 |
| 13-16 genes | 34 | 85.3 | 85.3 | 37.1 | 30.6 | 19.5 | 1.4 | 6.48 | 62.93 |

