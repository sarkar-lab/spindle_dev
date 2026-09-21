"""Per-dataset run logging: wall time, peak memory, and device specs.

Stdlib-only (``psutil`` is not installed in the ``spindle_env`` conda
environment) -- uses ``resource``/``platform``/``/proc`` for everything.
Each SLURM job processes one dataset per stage (index build or budget
sweep), so a run log is written per dataset per stage rather than once per
script invocation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import resource
import time


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except OSError:
        pass
    return float("nan")


class RunLogger:
    """Context manager that writes a JSON run log for one dataset/stage.

    Usage::

        with RunLogger(dataset_name="xenium_human_lung_cancer", stage="index_build",
                        out_dir=project_root / "results" / "run_logs",
                        seed=1, n_holdout=100):
            ... per-dataset work ...
    """

    def __init__(self, dataset_name: str, stage: str, out_dir: Path, **extra_fields):
        self.dataset_name = dataset_name
        self.stage = stage
        self.out_dir = Path(out_dir)
        self.extra_fields = extra_fields
        self._start_time = None

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = datetime.now(timezone.utc)
        self._start_perf = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now(timezone.utc)
        wall_time_s = time.perf_counter() - self._start_perf
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            "dataset_name": self.dataset_name,
            "stage": self.stage,
            "start_time_utc": self._start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "wall_time_s": round(wall_time_s, 3),
            "peak_rss_gb": round(peak_rss_kb / (1024 * 1024), 3),
            "hostname": platform.node(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "total_ram_gb": _total_ram_gb(),
            "exit_status": "ok" if exc_type is None else "error",
            "error_message": None if exc_type is None else f"{exc_type.__name__}: {exc_val}",
            **self.extra_fields,
        }
        out_path = self.out_dir / f"{self.dataset_name}_{self.stage}_run_log.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[RunLogger] Wrote {self.stage} run log for {self.dataset_name} to {out_path} "
              f"(wall_time_s={record['wall_time_s']}, peak_rss_gb={record['peak_rss_gb']}, "
              f"exit_status={record['exit_status']})")
        # Do not suppress exceptions.
        return False
