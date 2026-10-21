"""Tracked beats alone against the mapper's, for tracker settings from the environment.

    TRACK_SMOOTH=100 uv run python bench/track_sweep.py ranked _pcf --kind many

The steady lines come from bench/source_dump.py's dump (data/source_dump), so only the
tracker runs; prints the share of true beats with a tracked beat within 5 / 10 ms, at
the map's octave.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS, background  # noqa: E402
from source_dump import DUMP  # noqa: E402


def one(job):
    audio, truth, span, octave = job
    from autotiming import detect as D
    from autotiming import track as T
    from autotiming.beatprob import beat_prob
    from autotiming.grid import Segment, _nearest_dist, beats, subdivide
    from cache_onsets import exact_onsets

    f = DUMP / (hashlib.md5(Path(audio).read_bytes()).hexdigest() + ".npz")
    if not f.exists():
        return None
    z = np.load(f)
    steady = [Segment(t, L) for t, L in z["steady"]]
    pcm, sr = D.decode_bass(audio)
    o = exact_onsets(audio, pcm, sr)
    if o.profile is None or o.times[-1] - o.times[0] < D.TRACK_MIN_MS:
        return None
    main_L = D.main_beat_length(o, steady)
    secs = D.align_octave(T.fine_sections(o, main_L), steady)
    tr = T.track_beats(o, secs, probs=beat_prob(o, main_L)).times + D.OSU_OFFSET_MS
    truth = [Segment(*x) for x in truth]
    tf = subdivide(truth, 2) if octave == 2 else truth
    tb = beats(tf, *span)
    if octave == 0.5:
        tr = np.sort(np.r_[tr, (tr[1:] + tr[:-1]) / 2])
    d = np.abs(_nearest_dist(tb, tr))
    return len(tb), int((d < 5).sum()), int((d < 10).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("tag")
    p.add_argument("--kind", default="many")
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (RESULTS / f"{args.dataset}_ours{args.tag}.jsonl").open())}
    jobs = []
    for it in DATASETS[args.dataset]():
        r = rows.get(it["id"])
        if r and r.get("ok") and (args.kind == "all" or it["tags"]["kind"] == args.kind):
            oc = r["octave"] if r["octave"] in (0.5, 1, 2) else 1
            jobs.append((it["audio"], [(s.time, s.beat_length) for s in it["truth"]], it["span"], oc))
    with ProcessPoolExecutor(args.workers, initializer=background) as ex:
        res = [r for r in ex.map(one, jobs) if r]
    n, c5, c10 = np.sum(res, axis=0)
    print(f"{len(res)} maps: tracked beats within 5 ms {c5 / n:.1%}, within 10 ms {c10 / n:.1%}")


if __name__ == "__main__":
    main()
