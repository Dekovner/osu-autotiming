"""Sweep combine() settings on cached data: steady lines + cached tracked beats (cross-fitted probs).

    TRACK_ACT=prob TRACK_SMOOTH=30 uv run python bench/combine_eval.py '[{"HYBRID_ACCEPT_GAIN": "-1"}, ...]'

Each config runs in a fresh process with those environment variables (the
module constants read them at import). Reports coverage within 10 ms per
timing style and "bloat": single-red-line maps that got extra lines.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def worker():
    """Run in a child process: evaluate one config, print a JSON summary."""
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    sys.path.insert(0, str(HERE))
    from track_eval import beats_file, items_and_current

    items, cur = items_and_current()
    items = [it for it in items if beats_file(it["id"]).exists() and cur[it["id"]].get("pred")]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "12"))) as ex:
        res = list(ex.map(_one, [(it, cur[it["id"]]["pred"]) for it in items], chunksize=4))
    by = defaultdict(list)
    for it, r in zip(items, res):
        by[it["style"]].append(r)
        by["all"].append(r)
    out = {}
    for s, rs in by.items():
        out[s] = {"n": len(rs), "cov10": float(np.mean([r["cov10"] for r in rs])), "cov5": float(np.mean([r["cov5"] for r in rs])),
                  "bloat": float(np.mean([r["lines"] > r["lines_s"] for r in rs]))}
    print(json.dumps(out))


def _one(args):
    it, pred = args
    import numpy as np

    sys.path.insert(0, str(HERE))
    from cache_onsets import load_onsets
    from track_eval import beats_file

    from autotiming.detect import combine
    from autotiming.grid import Segment, compare
    from autotiming.track import Beats

    o = load_onsets(it["id"])
    steady = [Segment(t, L) for t, L in pred]
    z = np.load(beats_file(it["id"]))
    segs = combine(o, steady, Beats(z["times"], z["supported"]))
    r = compare(segs, it["truth"], *it["span"])
    return {"cov10": r.get("cover10", 0), "cov5": r.get("cover5", 0), "lines": len(segs), "lines_s": len(steady)}


def main():
    if sys.argv[1] == "--worker":
        return worker()
    configs = json.loads(sys.argv[1])
    print(f"{'config':58} {'single':>14} {'sections':>9} {'bar':>6} {'beat':>6} {'all':>6}   bloat(single)")
    for c in configs:
        env = dict(os.environ, **{k: str(v) for k, v in c.items()})
        out = json.loads(subprocess.run([sys.executable, __file__, "--worker"], env=env, check=True,
                                        capture_output=True, text=True).stdout.strip().splitlines()[-1])
        g = lambda s, k="cov10": out.get(s, {}).get(k, float("nan"))
        print(f"{json.dumps(c):58} {g('single'):14.1%} {g('sections'):9.1%} {g('bar-by-bar'):6.1%} {g('beat-by-beat'):6.1%} "
              f"{g('all'):6.1%}   {g('single', 'bloat'):.1%}", flush=True)


if __name__ == "__main__":
    main()
