# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 14 | 100.0 | 100.0 | 51.4 | 42.9 | 22.0 | 1.0 | 1.49 | 10.34 |
| 7-12 genes | 65 | 96.9 | 96.9 | 70.5 | 52.2 | 26.3 | 1.1 | 2.13 | 17.16 |
| 13-16 genes | 40 | 97.5 | 97.5 | 70.7 | 52.3 | 26.2 | 1.0 | 4.23 | 17.48 |
| >16 genes | 81 | 96.3 | 96.3 | 72.8 | 50.7 | 26.2 | 1.1 | 2.52 | 33.71 |

## Non-Contiguous Random
| Query Size | Count | Top-1 (%) | Top-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Avg Rank | Spindle (ms) | BF (ms) |
|:----------:|:-----:|:---------:|:---------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|
| <=6 genes | 12 | 100.0 | 100.0 | 72.5 | 53.3 | 26.0 | 1.0 | 2.84 | 9.77 |
| 7-12 genes | 65 | 93.8 | 93.8 | 76.2 | 58.3 | 28.0 | 1.1 | 3.16 | 16.72 |
| 13-16 genes | 63 | 93.7 | 93.7 | 84.6 | 59.8 | 28.4 | 1.1 | 2.99 | 20.33 |
| >16 genes | 60 | 96.7 | 96.7 | 78.0 | 54.1 | 27.4 | 1.1 | 4.60 | 27.12 |

