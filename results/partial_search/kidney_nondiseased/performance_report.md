# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 45 | 86.7 | 86.7 | 46.7 | 41.3 | 24.0 | 1.4 | 3.82 | 15.90 |
| 7-12 genes | 85 | 87.1 | 87.1 | 47.5 | 41.1 | 22.8 | 1.4 | 2.48 | 21.67 |
| 13-16 genes | 51 | 74.5 | 74.5 | 53.3 | 41.6 | 21.5 | 2.1 | 4.15 | 27.25 |
| >16 genes | 19 | 57.9 | 57.9 | 37.4 | 32.6 | 16.3 | 3.3 | 3.60 | 31.45 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 26 | 76.9 | 76.9 | 33.8 | 29.0 | 21.5 | 1.5 | 2.75 | 20.78 |
| 7-12 genes | 105 | 85.7 | 85.7 | 57.3 | 47.9 | 24.3 | 1.4 | 4.15 | 21.78 |
| 13-16 genes | 61 | 93.4 | 93.4 | 72.1 | 54.7 | 25.4 | 1.1 | 3.82 | 25.85 |
| >16 genes | 8 | 100.0 | 100.0 | 87.5 | 61.3 | 27.3 | 1.0 | 2.89 | 30.21 |

