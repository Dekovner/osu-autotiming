"""How much would a perfect beat-or-"and" choice per section of ours buy?

    uv run python bench/half_oracle.py heavy _hbase

Each of our red lines in turn moves by half a beat (its section only); kept when the
mapper's beats are covered better. Also counts how many sections that is and how long.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, compare  # noqa: E402


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    items = {it["id"]: it for it in DATASETS[ds]()}
    rows = [json.loads(l) for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open()]
    agg = defaultdict(lambda: np.zeros(6))
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    lens = []
    for r in rows:
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda s: s.time)
        s, e = it["span"]
        best = compare(pred, it["truth"], s, e)
        n = 0
        for i in range(len(pred)):
            cand = pred[:i] + [Segment(pred[i].time + pred[i].beat_length / 2, pred[i].beat_length, pred[i].meter)] + pred[i + 1:]
            res = compare(cand, it["truth"], s, e)
            if res.get("ok") and res["cover10"] > best["cover10"] + 0.005:
                end = pred[i + 1].time if i + 1 < len(pred) else e
                lens.append((end - max(pred[i].time, s)) / pred[i].beat_length)
                pred, best, n = cand, res, n + 1
        agg[r["tags"]["kind"]] += [1, r["cover10"], best["cover10"], good(r), good(best), n]
    for k, (m, c0, c1, g0, g1, n) in sorted(agg.items()):
        print(f"{k:7} {m:4.0f} maps  beats within 10 ms {c0 / m:.1%} -> {c1 / m:.1%}   ready as is {g0:.0f} -> {g1:.0f}   sections flipped {n:.0f}")
    if lens:
        print("flipped section length in beats: quantiles", np.quantile(lens, [0.1, 0.25, 0.5, 0.75, 0.9]).round(0))


if __name__ == "__main__":
    main()
