"""Run a list of worker commands through Slurm when it exists, else one after another on this machine.

    import slurm
    jobs = slurm.run([["build", "skin_melanoma", "0"], ...], script="experiments/common/index_cache.py",
                     sbatch="experiments/common/timing.sbatch", name="R0_index_cache", serial=True)

Under Slurm each command is one array task of ``sbatch`` (run.sbatch: spindle_env, one thread; timing.sbatch: one fixed
node, one job at a time), logs in results/_logs/<name>/; ``wait`` polls squeue and fails on any task that did not
complete. Without Slurm each command is its own ``python`` process (so its peak memory is its own), in order.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "_logs"


def available():
    return shutil.which("sbatch") is not None


def submit(commands, script, sbatch, name, serial=False, dependency=None, extra=()):
    """One sbatch array over ``commands`` (lists of string arguments to ``script``); returns the job id. The task
    reads its arguments from a file written next to the logs."""
    log = LOGS / name
    log.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(script).stem}_{commands[0][0]}_{time.strftime('%Y%m%d_%H%M%S')}"
    argfile = log / f"{stem}.args"
    argfile.write_text("\n".join(" ".join(c) for c in commands) + "\n")
    array = f"0-{len(commands) - 1}" + ("%1" if serial else "")
    cmd = ["sbatch", "--parsable", f"--array={array}", f"--output={log}/%x_%A_%a.out",
           f"--export=ALL,SPINDLE_ARGFILE={argfile}", *extra]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd += [str(ROOT / sbatch), str(ROOT / script)]
    job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
    print(f"submitted {job}: {len(commands)} x {Path(script).name} {commands[0][0]} ({'serial' if serial else 'parallel'})"
          f"; logs {log.relative_to(ROOT)}/", flush=True)
    return job


def wait(jobs, poll=60):
    """Block until the jobs leave the queue; raise if any task did not complete."""
    jobs = [j for j in jobs if j]
    t0 = time.time()
    while True:
        q = subprocess.run(["squeue", "-h", "-j", ",".join(jobs), "-o", "%T"], capture_output=True, text=True).stdout
        states = q.split()
        if not states:
            break
        print(f"\r{time.time() - t0:6.0f} s: {states.count('RUNNING')} running, {states.count('PENDING')} pending   ",
              end="", flush=True)
        time.sleep(poll)
    print()
    bad = []
    for j in jobs:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", j, "-o", "JobID,State"], capture_output=True,
                             text=True).stdout
        bad += [line for line in out.split() if not line.endswith("|COMPLETED")]
    if bad:
        raise RuntimeError(f"tasks that did not complete: {bad}")
    print(f"all done in {time.time() - t0:.0f} s")


def run_local(commands, script):
    for c in commands:
        print(f"$ python {Path(script).name} {' '.join(c)}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / script), *c], cwd=ROOT, check=True)


def run(commands, script, sbatch="experiments/common/run.sbatch", name="jobs", serial=False, dependency=None,
        extra=(), block=True):
    """Submit (Slurm) or run locally; returns the job id ('' locally)."""
    if not commands:
        return ""
    if not available():
        run_local(commands, script)
        return ""
    job = submit(commands, script, sbatch, name, serial=serial, dependency=dependency, extra=extra)
    if block:
        wait([job])
    return job
