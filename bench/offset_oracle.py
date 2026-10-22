"""How much would a perfect offset per section of ours buy?

    uv run python bench/offset_oracle.py heavy _sp1 [max_ms]

Each of our sections moves by the median of (mapper's beat - ours) over its beats that are
within max_ms (default 25) of one of the mapper's: a constant shift, tempo untouched.
"Ready as is" and coverage before / after, per kind; also with one global shift per map.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, _nearest_dist, beats, compare, subdivide  # noqa: E402


def shifted(pred, truth, s, e, max_ms, per_section=True):
    tb = beats(truth, s - 3000, e + 3000)
    pb = beats(pred, s, e)
    d = -_nearest_dist(pb, tb)  # mapper's - ours
    starts = np.array([x.time for x in pred])
    sec = np.clip(np.searchsorted(starts, pb, side="right") - 1, 0, len(pred) - 1)
    ok = np.abs(d) < max_ms
    if not per_section:
        g = np.median(d[ok]) if ok.any() else 0.0
        return [Segment(x.time + g, x.beat_length, x.meter) for x in pred]
    out = []
    for i, x in enumerate(pred):
        m = ok & (sec == i)
        out.append(Segment(x.time + (np.median(d[m]) if m.sum() >= 4 else 0.0), x.beat_length, x.meter))
    return out


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    max_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    items = {it["id"]: it for it in DATASETS[ds]()}
    agg = defaultdict(lambda: np.zeros(7))
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open():
        r = json.loads(l)
        it = items.get(r["id"])
        if not it or not r.get("ok") or r["octave"] not in (0.5, 1, 2):
            continue
        s, e = it["span"]
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda x: x.time)
        truth = subdivide(it["truth"], 2) if r["octave"] == 2 else it["truth"]
        base = pred if r["octave"] != 0.5 else subdivide(pred, 2)
        a = compare(shifted(base, truth, s, e, max_ms), truth, s, e)
        g = compare(shifted(base, truth, s, e, max_ms, per_section=False), truth, s, e)
        agg[r["tags"]["kind"]] += [1, good(r), good(a), good(g), r["cover10"], a.get("cover10", 0), g.get("cover10", 0)]
    for k, (n, g0, g1, g2, c0, c1, c2) in sorted(agg.items()):
        print(f"{k:7} {n:4.0f} maps  ready as is {g0:.0f} -> per section {g1:.0f} / one shift per map {g2:.0f}"
              f"   beats within 10 ms {c0 / n:.1%} -> {c1 / n:.1%} / {c2 / n:.1%}")


if __name__ == "__main__":
    main()
