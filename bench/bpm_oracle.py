"""How much would the mapper's exact BPM per section of ours buy (our phase kept)?

    uv run python bench/bpm_oracle.py heavy _sp1

Each section of ours within 0.5% of the mapper's tempo in force at its middle takes that
beat length, pivoting on our beat nearest the section's middle (so the phase there stays ours).
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
    agg = defaultdict(lambda: np.zeros(5))
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open():
        r = json.loads(l)
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        s, e = it["span"]
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda x: x.time)
        truth = sorted(it["truth"], key=lambda x: x.time)
        ts = np.array([x.time for x in truth])
        out = []
        for i, p in enumerate(pred):
            end = pred[i + 1].time if i + 1 < len(pred) else e
            mid = (max(p.time, s) + end) / 2
            tl = truth[max(0, int(np.searchsorted(ts, mid, side="right")) - 1)].beat_length
            for k in (1, 2, 0.5):
                if abs(np.log(p.beat_length * k / tl)) < 0.005:
                    L = tl / k
                    pivot = p.time + np.round((mid - p.time) / p.beat_length) * p.beat_length
                    n = np.round((pivot - p.time) / p.beat_length)
                    out.append(Segment(pivot - n * L, L, p.meter))
                    break
            else:
                out.append(p)
        new = compare(out, it["truth"], s, e)
        agg[r["tags"]["kind"]] += [1, good(r), good(new), r["cover10"], new.get("cover10", 0)]
    for k, (n, g0, g1, c0, c1) in sorted(agg.items()):
        print(f"{k:7} {n:4.0f} maps  ready as is {g0:.0f} -> {g1:.0f}   beats within 10 ms {c0 / n:.1%} -> {c1 / n:.1%}")


if __name__ == "__main__":
    main()
