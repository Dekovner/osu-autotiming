"""Beat tracker vs the current detector, per timing style, on cached onsets.

    uv run python bench/track_eval.py beats [workers]          # track + cache beats (slow, once)
    uv run python bench/track_eval.py sweep [workers] '<json>'  # compress cached beats with params

Styles from the ranked timing: single line, sections (>8 beats), bar-by-bar
(2-8), beat-by-beat (<=2). "current" comes from data/results/ranked_ours_steady.jsonl (steady detector alone).
"""

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from run import DATA, RESULTS, ranked_items  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS, _detect_raw, choose_octave  # noqa: E402
from autotiming.grid import Segment, compare  # noqa: E402
from autotiming.track import HYBRID_MARGIN, Beats, fine_sections, hybrid, lines, track_beats  # noqa: E402

BEATS = DATA / "beats"
N_SINGLE = 80


def style(it):
    t = sorted(it["truth"], key=lambda s: s.time)
    if len(t) == 1:
        return "single"
    lens = [((t[i + 1].time if i + 1 < len(t) else it["span"][1]) - s.time) / s.beat_length for i, s in enumerate(t)]
    med = np.median(lens)
    return "beat-by-beat" if med <= 2 else "bar-by-bar" if med <= 8 else "sections"


def tracker_key() -> str:
    """Cache key from the tracker's tunables, so going back to old settings costs nothing."""
    import hashlib

    from autotiming import track as T

    knobs = {k: getattr(T, k) for k in ("FPS", "OBS_LAMBDA", "TRANSITION_LAMBDA", "TEMPO_PRIOR", "SPAN",
                                        "FINE_WIN_MS", "FINE_HOP_MS", "SECTION_TOL", "FINE_RANGE", "LOW_MIX",
                                        "ACT_SOURCE", "PROB_STRENGTH_POW", "SMOOTH_LAMBDA", "SMOOTH_HUBER_MS")}
    return hashlib.md5(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[:8]


def beats_file(item_id):
    return BEATS / tracker_key() / (item_id.replace("/", "__") + ".npz")


def compute_beats(it):
    try:
        t0 = time.time()
        o = load_onsets(it["id"])
        coarse = choose_octave(o, _detect_raw(o))
        ends = [s.time for s in coarse[1:]] + [o.times[-1]]
        main_L = coarse[int(np.argmax([e - s.time for s, e in zip(coarse, ends)]))].beat_length
        probs = None
        from autotiming import track as T
        if T.ACT_SOURCE == "prob":  # cross-fitted: from a model that never saw this song
            key = it["id"].replace("/", "__")
            cf = np.load(DATA / "beatprob" / "crossfit.npz")
            idx = np.load(DATA / "beatprob" / (key + ".npz"))["idx"]
            probs = np.zeros(len(o.times))
            probs[idx] = cf[key]
        b = track_beats(o, fine_sections(o, main_L), probs=probs)
        np.savez(beats_file(it["id"]), times=b.times, supported=b.supported, secs=time.time() - t0)
        return None
    except Exception:
        return it["id"], traceback.format_exc(limit=4)


def compress(args):
    it, params, steady_pred = args
    try:
        z = np.load(beats_file(it["id"]))
        b = Beats(z["times"], z["supported"])
        if len(b.times) < 2:
            return it["id"], {"cover5": 0, "cover10": 0, "lines": 0}
        o = load_onsets(it["id"])
        params = dict(params)
        if params.pop("hybrid", False):
            steady = sorted((Segment(tm - OSU_OFFSET_MS, L) for tm, L in steady_pred), key=lambda s: s.time)
            b = hybrid(o, steady, b, margin=params.pop("margin", HYBRID_MARGIN))
        segs = lines(o, b, **params)
        segs = [Segment(s.time + OSU_OFFSET_MS, s.beat_length) for s in segs]
        r = compare(segs, it["truth"], *it["span"])
        return it["id"], {"cover5": r.get("cover5", 0), "cover10": r.get("cover10", 0), "lines": len(segs)}
    except Exception:
        return it["id"], {"cover5": 0, "cover10": 0, "lines": 0, "error": traceback.format_exc(limit=4)}


DEV = os.environ.get("DEV", "0") == "1"  # 15 maps per style: a one-minute check before a full run
DEV_PER_STYLE = 15


def items_and_current():
    cur = {r["id"]: r for r in map(json.loads, (RESULTS / "ranked_ours_steady.jsonl").open())}
    items, singles = [], []
    for it in ranked_items():
        if not path_for(it["id"]).exists() or it["id"] not in cur:
            continue
        it["style"] = style(it)
        (singles if it["style"] == "single" else items).append(it)
    items = items + singles[:N_SINGLE]
    if DEV:  # fixed, every 3rd map of each style so it spreads over sets
        per = defaultdict(list)
        for it in items:
            per[it["style"]].append(it)
        items = [it for g in per.values() for it in g[::3][:DEV_PER_STYLE]]
    return items, cur


def main():
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    items, cur = items_and_current()
    if mode == "beats":
        (BEATS / tracker_key()).mkdir(parents=True, exist_ok=True)
        todo = [it for it in items if not beats_file(it["id"]).exists()]
        print(f"{len(todo)} to track", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            errs = [e for e in ex.map(compute_beats, todo, chunksize=1) if e]
        for i, e in errs[:3]:
            print(i, e)
        return
    spec = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 and Path(sys.argv[3]).exists() else \
        json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    for params in spec if isinstance(spec, list) else [spec]:
        report(items, cur, params, workers)


def report(items, cur, params, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(compress, [(it, params, cur[it["id"]].get("pred", [])) for it in items], chunksize=4))
    by = defaultdict(list)
    for it in items:
        by[it["style"]].append(it)
    print(f"params {params or 'default'}")
    print(f"{'style':14} {'n':>4} | {'current ±5':>10} {'±10':>5} | {'tracker ±5':>10} {'±10':>5} {'lines/truth':>11} | {'best-of ±10':>11}")
    for s in ("single", "sections", "bar-by-bar", "beat-by-beat"):
        g = by.get(s, [])
        if not g:
            continue
        c5 = np.mean([cur[i["id"]].get("cover5", 0) for i in g])
        c10 = np.mean([cur[i["id"]].get("cover10", 0) for i in g])
        t5 = np.mean([res[i["id"]]["cover5"] for i in g])
        t10 = np.mean([res[i["id"]]["cover10"] for i in g])
        ratio = np.median([res[i["id"]]["lines"] / len(i["truth"]) for i in g])
        best = np.mean([max(cur[i["id"]].get("cover10", 0), res[i["id"]]["cover10"]) for i in g])
        print(f"{s:14} {len(g):4} | {c5:10.0%} {c10:5.0%} | {t5:10.0%} {t10:5.0%} {ratio:11.2f} | {best:11.0%}")
    errs = {k: v["error"] for k, v in res.items() if "error" in v}
    if errs:
        print(f"{len(errs)} errors, first: {next(iter(errs))}\n{next(iter(errs.values()))}")
    (RESULTS.parent / "track_eval.json").write_text(json.dumps({"params": params, "res": res}))


if __name__ == "__main__":
    main()
