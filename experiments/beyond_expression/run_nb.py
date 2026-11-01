"""Execute a notebook in place with jupyter_client (no nbclient in sparta_spatial). Usage: run_nb.py NB.ipynb"""
import json
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel

nb_path = Path(sys.argv[1]).resolve()
nb = json.loads(nb_path.read_text())
km, kc = start_new_kernel(kernel_name="python3", cwd=str(nb_path.parent))
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
            msg = kc.get_iopub_msg(timeout=7200)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                break
            if t == "stream":
                cell["outputs"].append({"output_type": "stream", "name": c["name"], "text": c["text"]})
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
        nb_path.write_text(json.dumps(nb, indent=1))
        if err:
            sys.exit(f"cell {count} failed")
finally:
    kc.stop_channels()
    km.shutdown_kernel()
print("\ndone")
