"""How much would putting our tempo changes exactly where the mapper's are buy?

    uv run python bench/boundary_oracle.py heavy _hbase

Each of our red lines that changes the tempo by > 2% and has a mapper's change within
32 beats moves onto it: the section before runs on to that point, the section after
keeps its own grid (its beat nearest the new time). Coverage and "ready as is" before
/ after, per kind.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_anatomy import stable_jumps  # noqa: E402
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, compare  # noqa: E402


def moved(pred, truth, span, max_beats=32):
    tj = np.array([t for t, _, _ in stable_jumps(truth, *span, min_beats=8)])
    if not len(tj):
        return pred
    out = list(pred)
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if abs(np.log(b.beat_length / a.beat_length)) < 0.02:
            continue
        k = int(np.argmin(np.abs(tj - b.time)))
        if abs(tj[k] - b.time) > max_beats * b.beat_length:
            continue
        lo = out[i - 1].time if i > 1 else -np.inf
        hi = out[i + 1].time if i + 1 < len(out) else np.inf
        if not lo < tj[k] < hi:
            continue
        n = np.round((tj[k] - b.time) / b.beat_length)
        out[i] = Segment(b.time + n * b.beat_length, b.beat_length, b.meter)
    return out


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    items = {it["id"]: it for it in DATASETS[ds]()}
    rows = [json.loads(l) for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open()]
    agg = defaultdict(lambda: np.zeros(5))
    for r in rows:
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda s: s.time)
        new = compare(moved(pred, it["truth"], it["span"]), it["truth"], *it["span"])
        good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
        agg[r["tags"]["kind"]] += [1, r["cover10"], new.get("cover10", 0), good(r), good(new)]
    for k, (n, c0, c1, g0, g1) in sorted(agg.items()):
        print(f"{k:7} {n:4.0f} maps  beats within 10 ms {c0 / n:.1%} -> {c1 / n:.1%}   ready as is {g0:.0f} -> {g1:.0f}")


if __name__ == "__main__":
    main()
