"""How much would a perfect "keep this tempo change or not" decision buy?

    uv run python bench/section_oracle.py ranked _fast1

For every red line of ours that changes the tempo by > 2%, try dropping it (the
section before runs on over its span); keep the drop when the mapper's beats are
covered better. Greedy, front to back. Prints coverage and whole-map p95 < 10 ms
before / after per kind.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, compare  # noqa: E402


def candidates(pred):
    return [i for i in range(1, len(pred)) if abs(np.log(pred[i].beat_length / pred[i - 1].beat_length)) > 0.02]


def oracle(pred, truth, span):
    base = compare(pred, truth, *span)
    if not base.get("ok"):
        return base, base, 0
    best, n = base, 0
    i = 1
    while i < len(pred):
        if abs(np.log(pred[i].beat_length / pred[i - 1].beat_length)) > 0.02:
            alt = pred[:i] + pred[i + 1:]
            r = compare(alt, truth, *span)
            if r.get("ok") and r["cover10"] > best["cover10"] + 1e-9:
                pred, best, n = alt, r, n + 1
                continue
        i += 1
    return base, best, n


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    items = {it["id"]: it for it in DATASETS[ds]()}
    rows = [json.loads(l) for l in (RESULTS / f"{ds}_ours{tag}.jsonl").open()]
    agg = defaultdict(lambda: np.zeros(6))
    for r in rows:
        it = items.get(r["id"])
        if not it or not r.get("ok"):
            continue
        pred = sorted((Segment(t, L) for t, L in r["pred"]), key=lambda s: s.time)
        base, best, n = oracle(pred, it["truth"], it["span"])
        k = r["tags"]["kind"]
        agg[k] += [1, base["cover10"], best["cover10"], base["p95_ms"] < 10, best["p95_ms"] < 10, n]
    for k, (m, c0, c1, w0, w1, n) in sorted(agg.items()):
        print(f"{k:7} n={m:4.0f}  cover10 {c0 / m:.1%} -> {c1 / m:.1%}   whole<10 {w0 / m:.1%} -> {w1 / m:.1%}   lines dropped {n:.0f}")


if __name__ == "__main__":
    main()
