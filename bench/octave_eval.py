"""Where is our tempo at the wrong octave? Time-weighted, from cached bench predictions.

    uv run python bench/octave_eval.py ranked ours_full1 [--ids data/subset.txt] [--list]

Every STEP_MS of the mapped span (first to last object), our tempo is
compared with the mapper's: same (x1), double (x2), half (x0.5), 3:2 / 2:3, or other. A map
is "whole x2" if most of its time is x2; "part" errors are sections at the wrong octave in a
map that is right elsewhere (the per-section octave choice, not the main one).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

STEP_MS = 250.0
CLASSES = {"x1": 0.0, "x2": 1.0, "x0.5": -1.0, "x1.5": np.log2(1.5), "x0.67": -np.log2(1.5),
           "x4": 2.0, "x0.25": -2.0, "x0.75": np.log2(0.75), "x1.33": np.log2(4 / 3)}
TOL = 0.015  # log2 units, ~1%


def at(lines, ts):
    starts = np.array([s[0] for s in lines])
    k = np.clip(np.searchsorted(starts, ts, side="right") - 1, 0, len(lines) - 1)
    return np.array([s[1] for s in lines])[k]


def classify(row, span):
    truth, pred = sorted(row["truth"]), sorted(row.get("pred") or [])
    ts = np.arange(span[0], span[1], STEP_MS)
    if not pred or not truth or not len(ts):
        return None
    r = np.log2(at(truth, ts) / at(pred, ts))  # our bpm / mapper bpm
    lab = np.full(len(ts), "other", dtype=object)
    for name, v in CLASSES.items():
        lab[np.abs(r - v) < TOL] = name
    return Counter(lab.tolist()), len(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("tag")
    p.add_argument("--ids")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    ids = set(Path(a.ids).read_text().split()) if a.ids else None
    rows = [json.loads(x) for x in open(RESULTS / f"{a.dataset}_{a.tag}.jsonl")]
    spans = {it["id"]: it["span"] for it in DATASETS[a.dataset]()}
    rows = [r for r in rows if (ids is None or r["id"] in ids) and r["id"] in spans]
    total, kinds, bad = Counter(), Counter(), []
    for r in rows:
        c = classify(r, spans[r["id"]])
        if c is None:
            continue
        c, n = c
        for k, v in c.items():
            total[k] += v / n
        top = max(c, key=c.get)
        if top != "x1":
            kinds[f"whole {top}"] += 1
            bad.append((r["id"], f"whole {top}", {k: round(v / n, 2) for k, v in c.most_common(3)}))
        elif c["x1"] / n < 0.9:
            worst = max((k for k in c if k != "x1"), key=c.get)
            kinds[f"part {worst}"] += 1
            bad.append((r["id"], f"part {worst}", {k: round(v / n, 2) for k, v in c.most_common(3)}))
        else:
            kinds["ok"] += 1
    n = len(rows)
    print(f"{n} maps; mean share of time:", {k: f"{100 * v / n:.1f}%" for k, v in total.most_common()})
    print("maps:", dict(kinds.most_common()))
    if a.list:
        for b in sorted(bad, key=lambda x: x[1]):
            print(*b)


if __name__ == "__main__":
    main()
