"""How much would a perfect x2 / x0.5 choice per section of ours buy?

    uv run python bench/octave_oracle.py heavy _pcf

Each of our red lines in turn tries double and half tempo (half on either of its two
beats); a change is kept when beats match better both ways (the mapper's beats we hit +
ours that hit theirs, within 10 ms, over both counts), greedily. "Ready as is" before / after.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, _nearest_dist, beats, compare  # noqa: E402


def both_ways(pred, tb, s, e):
    pb = beats(pred, s, e)
    if len(pb) < 2:
        return 0.0
    a = np.abs(_nearest_dist(tb, beats(pred, s - 3000, e + 3000))) < 10
    b = np.abs(_nearest_dist(pb, tb)) < 10
    return (a.sum() + b.sum()) / (len(tb) + len(pb))


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    items = {it["id"]: it for it in DATASETS[ds]()}
    rows = [json.loads(l) for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open()]
    agg = defaultdict(lambda: np.zeros(6))
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    for r in rows:
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        s, e = it["span"]
        tb = beats(it["truth"], s, e)
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda x: x.time)
        best, n = both_ways(pred, tb, s, e), 0
        for i in range(len(pred)):
            p = pred[i]
            alts = [Segment(p.time, p.beat_length / 2, p.meter), Segment(p.time, p.beat_length * 2, p.meter),
                    Segment(p.time + p.beat_length, p.beat_length * 2, p.meter)]
            scored = [(both_ways(pred[:i] + [a] + pred[i + 1:], tb, s, e), a) for a in alts]
            sc, a = max(scored, key=lambda x: x[0])
            if sc > best + 0.002:
                pred, best, n = pred[:i] + [a] + pred[i + 1:], sc, n + 1
        new = compare(pred, it["truth"], s, e)
        agg[r["tags"]["kind"]] += [1, good(r), good(new), n, r["cover10"], new.get("cover10", 0)]
    for k, (m, g0, g1, n, c0, c1) in sorted(agg.items()):
        print(f"{k:7} {m:4.0f} maps  ready as is {g0:.0f} -> {g1:.0f}   sections rescaled {n:.0f}   beats within 10 ms {c0 / m:.1%} -> {c1 / m:.1%}")


if __name__ == "__main__":
    main()
