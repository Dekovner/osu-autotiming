"""Score bench/source_dump.py's stages against the mapper's beats.

    uv run python bench/source_eval.py ranked _fast1 [--kind many]

cover10 = share of true beats with one of ours within 10 ms, at the map's octave
(from the bench row), for: steady lines, tracked beats, hybrid beats, final lines;
"oracle" picks steady or tracked per true section, whichever covers it better.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402
from source_dump import DUMP  # noqa: E402

from autotiming.grid import Segment, _nearest_dist, beats, subdivide  # noqa: E402


def halves(b):
    return np.sort(np.concatenate([b, (b[1:] + b[:-1]) / 2])) if len(b) > 1 else b


def grid(arr, lo, hi):
    return beats([Segment(t, L) for t, L in arr], lo, hi) if len(arr) else np.zeros(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("tag")
    p.add_argument("--kind", default="many")
    args = p.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (RESULTS / f"{args.dataset}_ours{args.tag}.jsonl").open())}
    md5 = {}
    tot = defaultdict(float)
    n_maps = 0
    for it in DATASETS[args.dataset]():
        r = rows.get(it["id"])
        if not r or not r.get("ok") or (args.kind != "all" and it["tags"]["kind"] != args.kind):
            continue
        if it["audio"] not in md5:
            md5[it["audio"]] = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        f = DUMP / f"{md5[it['audio']]}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        s, e = it["span"]
        oct_ = r["octave"] if r["octave"] in (0.5, 1, 2) else 1
        truth = sorted(it["truth"], key=lambda x: x.time)
        tf = subdivide(truth, 2) if oct_ == 2 else truth
        tb = beats(tf, s, e)
        if len(tb) < 2:
            continue
        src = {"steady": grid(z["steady"], s - 5000, e + 5000), "final": grid(z["final"], s - 5000, e + 5000)}
        if "tracked" in z:
            src["tracked"], src["merged"] = z["tracked"], z["merged"]
        else:
            src["tracked"] = src["merged"] = src["steady"]
        if oct_ == 0.5:
            src = {k: halves(v) for k, v in src.items()}
        cov = {k: np.abs(_nearest_dist(tb, v)) < 10 for k, v in src.items()}
        # oracle: per true section, the better of steady and tracked
        starts = np.array([x.time for x in tf])
        sec = np.clip(np.searchsorted(starts, tb, side="right") - 1, 0, len(tf) - 1)
        best = 0
        for k in np.unique(sec):
            m = sec == k
            best += max(cov["steady"][m].sum(), cov["tracked"][m].sum())
        n_maps += 1
        tot["beats"] += len(tb)
        for k, c in cov.items():
            tot[k] += c.sum()
        tot["oracle"] += best
        tot["share"] += float(z["merged_share"]) if "merged_share" in z else 0
    n = tot["beats"]
    print(f"{n_maps} maps, {n:.0f} true beats; mean tracked share in hybrid {tot['share'] / max(1, n_maps):.2f}")
    for k in ("steady", "tracked", "merged", "final", "oracle"):
        print(f"  {k:8} cover10 {tot[k] / n:.1%}")


if __name__ == "__main__":
    main()
