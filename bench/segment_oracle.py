"""Perfect segmentation, our own fit: the mapper's section boundaries, each section's grid
fitted to the onsets by fit_grid (from the mapper's tempo, +-1.5%, our phase).

    uv run python bench/segment_oracle.py ranked _pcf --kind many

Sections shorter than --min-s seconds keep the mapper's line (marked as copied). If this
covers the mapper's beats much better than our timing does, finding the tempo changes is
the bottleneck; if not, fitting each section is.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS, background  # noqa: E402


def one(job):
    audio, truth, span, min_s, row = job
    from autotiming import detect as D
    from autotiming.grid import Segment, compare
    from cache_onsets import exact_onsets

    pcm, sr = D.decode_bass(audio)
    o = exact_onsets(audio, pcm, sr)
    t, w = o.times, D.weights(o)
    wl = o.strength * (o.sub + 0.5 * o.low)
    truth = sorted((Segment(*x) for x in truth), key=lambda s: s.time)
    out, copied = [], 0.0
    for i, s in enumerate(truth):
        a = s.time - D.OSU_OFFSET_MS
        b = (truth[i + 1].time if i + 1 < len(truth) else span[1]) - D.OSU_OFFSET_MS
        sel = (t >= a) & (t < b)
        if (b - a) / 1000 < min_s or sel.sum() < 8:
            out.append(s)
            copied += max(0.0, min(b, span[1]) - max(a, span[0]))
            continue
        phi, L, _ = D.fit_grid(t[sel], w[sel], s.beat_length, span=0.015, w_low=wl[sel])
        k = np.ceil((a - phi) / L - 0.5)  # the section starts on its own beat nearest the mapper's line
        out.append(Segment(phi + k * L + D.OSU_OFFSET_MS, L, s.meter))
    res = compare(out, truth, *span)
    good = lambda x: x.get("ok") and x["octave"] in (0.5, 1, 2) and x["p95_ms"] < 10
    return res.get("cover10", 0), row["cover10"], good(res), good(row), copied / max(1.0, span[1] - span[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("tag")
    p.add_argument("--kind", default="many")
    p.add_argument("--min-s", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (RESULTS / f"{args.dataset}_ours{args.tag}.jsonl").open())}
    jobs = [(it["audio"], [(s.time, s.beat_length, s.meter) for s in it["truth"]], it["span"], args.min_s, rows[it["id"]])
            for it in DATASETS[args.dataset]()
            if it["id"] in rows and rows[it["id"]].get("ok") and (args.kind == "all" or it["tags"]["kind"] == args.kind)]
    with ProcessPoolExecutor(args.workers, initializer=background) as ex:
        res = np.array(list(ex.map(one, jobs)), float)
    print(f"{len(res)} maps: beats within 10 ms ours {res[:, 1].mean():.1%} -> mapper's boundaries, our fit {res[:, 0].mean():.1%}"
          f"   ready as is {res[:, 3].sum():.0f} -> {res[:, 2].sum():.0f}   (song share in copied short sections {res[:, 4].mean():.1%})")


if __name__ == "__main__":
    main()
