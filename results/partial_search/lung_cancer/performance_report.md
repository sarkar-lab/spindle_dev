# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 25 | 60.0 | 60.0 | 36.0 | 32.8 | 21.2 | 2.4 | 2.71 | 37.30 |
| 7-12 genes | 67 | 79.1 | 79.1 | 62.2 | 47.8 | 25.2 | 2.0 | 5.91 | 41.36 |
| 13-16 genes | 40 | 87.5 | 87.5 | 69.3 | 48.5 | 25.1 | 1.3 | 7.62 | 57.35 |
| >16 genes | 68 | 98.5 | 98.5 | 82.5 | 56.7 | 26.9 | 1.0 | 9.01 | 74.49 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 13 | 61.5 | 61.5 | 40.0 | 35.8 | 20.3 | 3.2 | 2.27 | 32.06 |
| 7-12 genes | 77 | 85.7 | 85.7 | 62.3 | 47.7 | 25.6 | 1.5 | 6.04 | 43.49 |
| 13-16 genes | 53 | 98.1 | 98.1 | 81.3 | 58.6 | 27.2 | 1.0 | 6.21 | 53.02 |
| >16 genes | 57 | 100.0 | 100.0 | 81.4 | 57.7 | 27.9 | 1.0 | 9.85 | 65.27 |

