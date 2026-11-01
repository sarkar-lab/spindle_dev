"""Execute a notebook top to bottom and save it in place with its outputs (jupyter_client only; no nbclient).

    python experiments/common/run_nb.py <notebook.ipynb> [--kernel spindle_env]

Login node: figure notebooks (light). Compute nodes: result notebooks through run_nb.sbatch / timing_nb.sbatch.
Stops at the first failing cell (the notebook is saved up to it).
"""
import argparse
import json
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel

ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("--kernel", default="spindle_env")
a = ap.parse_args()
nb_path = Path(a.notebook).resolve()
nb = json.loads(nb_path.read_text())
km, kc = start_new_kernel(kernel_name=a.kernel, cwd=str(nb_path.parent))
count = 0
try:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["outputs"], cell["execution_count"] = [], count
        msg_id = kc.execute("".join(cell["source"]))
        err = False
        while True:
            msg = kc.get_iopub_msg(timeout=86400)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                break
            if t == "stream":
                outs = cell["outputs"]
                if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == c["name"]:
                    outs[-1]["text"] += c["text"]
                else:
                    outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                print(c["text"], end="", flush=True)
            elif t in ("display_data", "execute_result"):
                out = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                if t == "execute_result":
                    out["execution_count"] = count
                cell["outputs"].append(out)
                if "text/plain" in c["data"] and "image/png" not in c["data"]:
                    print(c["data"]["text/plain"], flush=True)
            elif t == "error":
                cell["outputs"].append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                        "traceback": c["traceback"]})
                print("\n".join(c["traceback"]), flush=True)
                err = True
        nb_path.write_text(json.dumps(nb, indent=1) + "\n")
        if err:
            sys.exit(f"{nb_path.name}: cell {count} failed")
finally:
    kc.stop_channels()
    km.shutdown_kernel()
print(f"\n{nb_path.name}: done")
