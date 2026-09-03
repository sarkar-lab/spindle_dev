# Interval Index Partial Search Performance Report

This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.

## Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 6 | 50.0 | 50.0 | 50.0 | 52.5 | 50.0 | 2.0 | 9.73 | 50.01 | 5.14 |
| 7-12 genes | 16 | 93.8 | 93.8 | 65.6 | 65.6 | 59.2 | 1.1 | 15.15 | 61.22 | 4.04 |
| 13-16 genes | 11 | 100.0 | 100.0 | 90.9 | 90.0 | 70.5 | 1.0 | 20.07 | 82.42 | 4.11 |
| >16 genes | 17 | 88.2 | 88.2 | 95.9 | 92.9 | 70.8 | 1.1 | 35.77 | 208.19 | 5.82 |

## Non-Contiguous Random
| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |
|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|
| <=6 genes | 6 | 100.0 | 100.0 | 48.3 | 50.8 | 50.7 | 1.0 | 11.13 | 51.69 | 4.64 |
| 7-12 genes | 22 | 100.0 | 100.0 | 78.2 | 76.1 | 73.0 | 1.0 | 16.22 | 63.43 | 3.91 |
| 13-16 genes | 11 | 100.0 | 100.0 | 98.2 | 94.5 | 75.6 | 1.0 | 22.39 | 82.34 | 3.68 |
| >16 genes | 11 | 100.0 | 100.0 | 97.3 | 96.4 | 78.0 | 1.0 | 32.04 | 127.13 | 3.97 |

