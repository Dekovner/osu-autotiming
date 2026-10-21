"""Dump what each stage of detect_full makes of a song, for offline study.

    OCTAVE_CROSSFIT=1 OCTAVE_CROSSFIT_DIR=data/octave_crossfit_sec \
        uv run python bench/source_dump.py ranked --kind many --workers 10

Per distinct audio file, data/source_dump/<md5>.npz: steady red lines (osu! clock),
tracked beats, hybrid beats (+ which are tracked) and the final lines, the way
detect_full builds them (crossfit models when asked, as in baselines.ours).
bench/source_eval.py scores them against the mapper's beats.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, background  # noqa: E402

DUMP = Path(os.environ.get("SOURCE_DUMP", Path(__file__).resolve().parents[1] / "data" / "source_dump"))


def crossfit(path):
    from autotiming import detect as D

    for env, name, attr in (("OCTAVE_CROSSFIT", "octave", "OCTAVE_TREES"), ("PHASE_CROSSFIT", "phase", "PHASE_TREES"),
                            ("SECPHASE_CROSSFIT", "secphase", "SECPHASE_TREES")):
        if os.environ.get(env) == "1":
            cf = Path(os.environ.get(f"{env}_DIR") or DUMP.parent / f"{name}_crossfit")
            fold = json.loads((cf / "assign.json").read_text()).get(hashlib.md5(Path(path).read_bytes()).hexdigest())
            setattr(D, attr, cf / f"fold{fold}.npz" if fold is not None else Path(D.__file__).resolve().parent / "models" / f"{name}_trees.npz")
    if os.environ.get("RUN_CROSSFIT") == "1":
        from autotiming import track as T

        cf = Path(os.environ["RUN_CROSSFIT_DIR"])
        fold = json.loads((cf / "assign.json").read_text()).get(hashlib.md5(Path(path).read_bytes()).hexdigest(), 0)
        T.RUN_TREES = cf / f"fold{fold}.npz"


def stages(path):
    from autotiming import detect as D
    from autotiming.beatprob import beat_prob
    from autotiming.downbeat import harmony
    from autotiming.track import fine_sections, hybrid, lines, track_beats
    from cache_onsets import exact_onsets

    crossfit(path)
    pcm, sr = D.decode_bass(path)
    o = exact_onsets(path, pcm, sr)
    ch = harmony(pcm, sr)
    steady, _ = D.steady_phase(o, pcm, sr, ch.chroma, D.detect_onsets(o, {}))
    out = {"steady": np.array([[s.time, s.beat_length] for s in steady])}
    if o.profile is not None and o.times[-1] - o.times[0] >= D.TRACK_MIN_MS:
        main_L = D.main_beat_length(o, steady)
        secs = D.align_octave(fine_sections(o, main_L), steady)
        tracked = track_beats(o, secs, probs=beat_prob(o, main_L))
        phys = sorted([D.Segment(s.time - D.OSU_OFFSET_MS, s.beat_length) for s in steady], key=lambda s: s.time)
        thirds = [(a.time, b.time) for a, b in zip(phys, phys[1:]) if D.interlock(a, b)]
        merged = hybrid(o, phys, tracked, thirds=thirds)
        out["secs"] = np.array([[s.time + D.OSU_OFFSET_MS, s.beat_length] for s in secs])
        out["tracked"] = tracked.times + D.OSU_OFFSET_MS
        out["merged"] = merged.times + D.OSU_OFFSET_MS
        out["merged_share"] = np.array(merged.tracked_share)
        # the merged beats as red lines, before combine's gates, and what combine keeps
        raw = sorted(lines(o, merged, line_cost=D.LINE_COST, sigma_ms=D.LINE_SIGMA_MS), key=lambda s: s.time)
        out["lines"] = np.array([[s.time + D.OSU_OFFSET_MS, s.beat_length] for s in raw])
        out["combined"] = np.array([[s.time, s.beat_length] for s in D.combine(o, steady, tracked, None, ch)])
    final = D.detect_full(o, pcm, sr)
    out["final"] = np.array([[s.time, s.beat_length] for s in final])
    return out


def one(path):
    f = DUMP / (hashlib.md5(Path(path).read_bytes()).hexdigest() + ".npz")
    if not f.exists():
        try:
            np.savez(f, **stages(path))
        except Exception as e:
            return path, repr(e)
    return path, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--kind", default="all")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    DUMP.mkdir(parents=True, exist_ok=True)
    paths = sorted({it["audio"] for it in DATASETS[args.dataset]() if args.kind == "all" or it["tags"]["kind"] == args.kind},
                   key=lambda a: -Path(a).stat().st_size)
    print(f"{len(paths)} audio files", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=background) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, a) for a in paths]), 1):
            path, err = f.result()
            if err:
                print(f"  {path}: {err}", flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
