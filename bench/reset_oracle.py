"""Our same-tempo resets (a red line at the previous BPM that shifts the phase): worth keeping?

    uv run python bench/reset_oracle.py heavy _hbase

For every line of ours at the tempo of the one before (within 0.1%) whose grid is off the
previous grid by more than 3 ms and less than 0.4 beat, try putting it back on the previous
grid; "oracle" keeps whichever covers the mapper's beats better, "all" always does it.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, compare  # noqa: E402


def resets(pred):
    out = []
    for i in range(1, len(pred)):
        a, b = pred[i - 1], pred[i]
        if abs(np.log(b.beat_length / a.beat_length)) > 0.001:
            continue
        x = (b.time - a.time) / a.beat_length
        off = (x - np.round(x)) * a.beat_length
        if 3 < abs(off) < 0.4 * a.beat_length:
            out.append((i, off))
    return out


def on_prev(pred, i):
    a, b = pred[i - 1], pred[i]
    n = np.round((b.time - a.time) / a.beat_length)
    return pred[:i] + [Segment(a.time + n * a.beat_length, a.beat_length, b.meter)] + pred[i + 1:]


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    items = {it["id"]: it for it in DATASETS[ds]()}
    rows = [json.loads(l) for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open()]
    agg = defaultdict(lambda: np.zeros(8))
    offs = []
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    for r in rows:
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda s: s.time)
        rs = resets(pred)
        base = compare(pred, it["truth"], *it["span"])
        allp, orc, best = pred, pred, base
        for i, off in rs:
            allp = on_prev(allp, i)
            cand = on_prev(orc, i)
            res = compare(cand, it["truth"], *it["span"])
            better = res.get("ok") and res["cover10"] > best["cover10"]
            offs.append((off, better))
            if better:
                orc, best = cand, res
        al = compare(allp, it["truth"], *it["span"])
        agg[r["tags"]["kind"]] += [1, len(rs), base["cover10"], al.get("cover10", 0), best["cover10"], good(base), good(al), good(best)]
    for k, (n, nr, c0, ca, co, g0, ga, go) in sorted(agg.items()):
        print(f"{k:7} {n:4.0f} maps, {nr:4.0f} resets  beats within 10 ms {c0 / n:.1%} / all {ca / n:.1%} / oracle {co / n:.1%}"
              f"   ready as is {g0:.0f} / {ga:.0f} / {go:.0f}")
    o = np.array(offs) if offs else np.zeros((0, 2))
    for lo, hi in ((3, 8), (8, 15), (15, 30), (30, 1000)):
        m = (np.abs(o[:, 0]) >= lo) & (np.abs(o[:, 0]) < hi)
        print(f"  phase jump {lo}-{hi} ms: {m.sum()} resets, undoing helps {o[m, 1].mean() if m.any() else 0:.0%}")


if __name__ == "__main__":
    main()
