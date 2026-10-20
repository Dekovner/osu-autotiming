"""Can we pick between the steady detector and the beat tracker without labels?

    uv run python bench/select_exp.py [workers] '<tracker params json>'

Score = onset mass on each timing's grid (piecewise_excess) minus a cost per
red line; sweep the trade-offs and report coverage per style vs the oracle.
"""

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets  # noqa: E402
from track_eval import beats_file, items_and_current  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS, piecewise_excess, weights  # noqa: E402
from autotiming.grid import Segment, compare  # noqa: E402
from autotiming.track import Beats, lines  # noqa: E402

PARAMS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"sigma_ms": 6, "line_cost": 100}


def one(args):
    it, cur_pred = args
    o = load_onsets(it["id"])
    t, w = o.times, weights(o)
    z = np.load(beats_file(it["id"]))
    b = Beats(z["times"], z["supported"])
    trk = lines(o, b, **PARAMS) if len(b.times) > 1 else []
    cur = [Segment(tm - OSU_OFFSET_MS, L) for tm, L in cur_pred]  # back to the onset clock
    out = {"style": it["style"], "n_beats": max(1, len(b.times))}
    for name, segs in (("cur", cur), ("trk", trk)):
        if not segs:
            out[name] = {"ex": -1, "lines": 0, "cov10": 0, "cov5": 0}
            continue
        segs = sorted(segs, key=lambda s: s.time)
        r = compare([Segment(s.time + OSU_OFFSET_MS, s.beat_length) for s in segs], it["truth"], *it["span"])
        out[name] = {"ex": piecewise_excess(t, w, segs), "ex4": piecewise_excess(t, w, segs, tol=4.0),
                     "lines": len(segs), "cov10": r.get("cover10", 0), "cov5": r.get("cover5", 0)}
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    items, cur = items_and_current()
    items = [it for it in items if beats_file(it["id"]).exists()]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, [(it, cur[it["id"]]["pred"]) for it in items], chunksize=4))
    (Path(__file__).resolve().parents[1] / "data" / "select_rows.json").write_text(json.dumps(rows))
    styles = ("single", "sections", "bar-by-bar", "beat-by-beat")

    def evaluate(choose):
        res = defaultdict(list)
        for r in rows:
            pick = "trk" if choose(r) else "cur"
            res[r["style"]].append(r[pick]["cov10"])
            res["all"].append(r[pick]["cov10"])
        return {k: np.mean(v) for k, v in res.items()}

    base = evaluate(lambda r: False)
    oracle = evaluate(lambda r: r["trk"]["cov10"] > r["cur"]["cov10"])
    print("coverage ±10ms   " + "  ".join(f"{s:>12}" for s in (*styles, "all")))
    print("current          " + "  ".join(f"{base[s]:12.1%}" for s in (*styles, "all")))
    print("oracle pick      " + "  ".join(f"{oracle[s]:12.1%}" for s in (*styles, "all")))
    best = None
    for key in ("ex", "ex4"):
        for per_line in (0.0, 0.0005, 0.001, 0.002, 0.005):
            for margin in (0.0, 0.01, 0.02, 0.04):
                ch = lambda r: (r["trk"][key] - r["cur"][key]) - per_line * (r["trk"]["lines"] - r["cur"]["lines"]) > margin
                e = evaluate(ch)
                if best is None or e["all"] > best[0]["all"]:
                    best = (e, key, per_line, margin)
    e, key, per_line, margin = best
    print(f"best rule ({key}, per_line={per_line}, margin={margin})")
    print("learned pick     " + "  ".join(f"{e[s]:12.1%}" for s in (*styles, "all")))


if __name__ == "__main__":
    main()
