"""Check the rewritten ``spindle_dev.search.search_index`` against the committed original.

The original implementation is loaded from ``git show <rev>:src/spindle_dev/search.py``
(default rev HEAD) as ``spindle_dev._search_ref``. For every held-out query of the given
datasets (seed-73 production indexes in results/indexes/) and every niche, both are run with
holdout_core.perform_search's configuration and must return identical paths
(``node_path`` and ``total_distance``, compared exactly) and identical leaf SPD sets. Also
checked, on the same queries:
- the ``run_sanity_search`` configuration (max_results=2, 5 failed starts, 10 failed paths,
  budget = 1.5 * eps * n_blocks);
- a partial query covering only the first half of each niche's blocks (the
  determine_active_blocks path);
- ``search_top_c``: its scores never decrease, and its top-c agrees with the top-c of the
  exhaustive traversal ranked by ``path_score`` (reported as overlap: the DFS stops at its
  path limits, the best-first search does not).

Usage: python benchmarks/check_fast_search.py --dataset-paths <ds.h5ad> ... [--max-queries N] [--rev HEAD]
"""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
for p in (current_dir, project_root / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spindle_dev  # noqa: F401  (package for the reference module's relative imports)
import spindle_dev.search as search
from holdout_core import _niche_scale_factor, extract_query_matrices, load_index


def load_reference(rev):
    src = subprocess.run(["git", "show", f"{rev}:src/spindle_dev/search.py"], cwd=project_root,
                         check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "_search_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("spindle_dev._search_ref", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "spindle_dev"
    sys.modules["spindle_dev._search_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def paths_of(res):
    return [] if isinstance(res, list) else [(p.node_path, p.total_distance) for p in res.paths]


def leaf_sets_ref(handle, res):
    out = set()
    if isinstance(res, list):
        return out
    for p in res.paths:
        out |= set.intersection(*[{int(s) for s, _ in handle.nodes[n].metadata.members} for n in p.node_path])
    return out


def leaf_sets_new(res):
    return set() if isinstance(res, list) else {s for p in res.paths for s in p.member_ids}


def check_dataset(ref, dataset_path, max_queries):
    name = Path(dataset_path).stem
    saved = load_index(project_root / "results" / "indexes", name)
    data, dag_dict, config = saved['data'], saved['dag_dict'], saved['config']
    queries = extract_query_matrices(saved['test_tile_covs'])
    if max_queries is not None:
        queries = queries[:max_queries]
    niches = sorted(set(int(c) for c in data.labels))
    cfg, budget, cfg_s, budget_s = {}, {}, {}, {}
    for c in niches:
        f = _niche_scale_factor(int(np.sum(data.labels == c)))
        kw = dict(max_results=None, debug=False, max_failed_starts=max(1, round(100 * f)),
                  max_failed_paths=max(1, round(200 * f)), total_paths_limit=max(1, round(3000 * f)))
        cfg[c] = (search.SearchConfig(**kw), ref.SearchConfig(**kw))
        nb = len(dag_dict[c].sorted_blocks)
        budget[c] = float(config.epsilon_dict[c]) * nb * 1.0 * f
        kw_s = dict(max_results=2, debug=False, max_failed_starts=5, max_failed_paths=10)
        cfg_s[c] = (search.SearchConfig(**kw_s), ref.SearchConfig(**kw_s))
        budget_s[c] = float(config.epsilon_dict[c]) * nb * 1.5

    n = {'paths': 0, 'leaves': 0, 'sanity': 0, 'partial': 0, 'monotone': 0}
    t_ref = t_new = 0.0
    overlaps = {10: [], 100: [], 400: []}
    total = 0
    for qi, q in enumerate(queries):
        per_q_ok = {'paths': True, 'leaves': True, 'sanity': True, 'partial': True}
        full = []
        tops = []
        for c in niches:
            h = dag_dict[c]
            perm = data.perm_list[c]
            qp = q[np.ix_(perm, perm)]
            runs = data.block_dict[c]
            t0 = time.perf_counter()
            r_ref = ref.search_index(h, qp, [], runs, budget[c], config=cfg[c][1])
            t1 = time.perf_counter()
            r_new = search.search_index(h, qp, [], runs, budget[c], config=cfg[c][0])
            t2 = time.perf_counter()
            t_ref += t1 - t0
            t_new += t2 - t1
            per_q_ok['paths'] &= paths_of(r_ref) == paths_of(r_new)
            per_q_ok['leaves'] &= leaf_sets_ref(h, r_ref) == leaf_sets_new(r_new)
            if not isinstance(r_new, list):
                full.extend((p.path_score, c, s) for p in r_new.paths for s in p.member_ids)
            per_q_ok['sanity'] &= paths_of(ref.search_index(h, qp, [], runs, budget_s[c], config=cfg_s[c][1])) == \
                paths_of(search.search_index(h, qp, [], runs, budget_s[c], config=cfg_s[c][0]))
            half = runs[:max(1, len(runs) // 2)]
            per_q_ok['partial'] &= paths_of(ref.search_index(h, qp, [], half, budget[c], config=cfg[c][1])) == \
                paths_of(search.search_index(h, qp, [], half, budget[c], config=cfg[c][0]))
            tops.append((h, qp, runs))
        for k in per_q_ok:
            n[k] += int(per_q_ok[k])
        full.sort()
        for c_top in overlaps:
            top = search.search_top_c(tops, c_top, budgets=[budget[c] for c in niches])
            scores = [s for s, _, _ in top]
            if c_top == max(overlaps):
                n['monotone'] += int(all(a <= b for a, b in zip(scores, scores[1:])))
            got = {(niches[pos], sid) for _, pos, sid in top}
            want = {(c, s) for _, c, s in full[:c_top]}
            overlaps[c_top].append(len(got & want) / max(1, len(want)))
        total += 1
    print(f"\n{name}: {total} queries x {len(niches)} niches")
    for k, v in n.items():
        print(f"  {k:10s} identical {v}/{total}")
    for c_top, v in overlaps.items():
        print(f"  search_top_c({c_top}) vs exhaustive-DFS top-{c_top} by path_score: mean overlap {np.mean(v):.4f}, "
              f"min {np.min(v):.4f}")
    print(f"  search_index time per query: original {t_ref / total * 1000:.1f} ms, new {t_new / total * 1000:.1f} ms "
          f"({t_ref / max(t_new, 1e-12):.1f}x)")
    return all(v == total for v in n.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset-paths', nargs='+', required=True)
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--rev', default='HEAD', help='git revision holding the original search.py')
    args = ap.parse_args()
    ref = load_reference(args.rev)
    ok = all([check_dataset(ref, p, args.max_queries) for p in args.dataset_paths])
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
