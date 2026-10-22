"""Run methods over a dataset and summarize.

    uv run python bench/run.py synth --methods ours librosa essentia
    uv run python bench/run.py ranked --methods ours
    uv run python bench/run.py synth --summary-only --by style
    uv run python bench/run.py ranked --ids data/subset_ramp.txt   # a subset first

Results are cached per (dataset, method) in data/results/; delete a file to rerun.

Each distinct audio file is detected once (byte-identical songs in several mapsets
are ~1/5 of ranked), onsets come from bench/cache_onsets.py's exact cache
(BENCH_ONSET_CACHE=0 to recompute), and workers run SCHED_IDLE: they only take
CPU the desktop leaves free.
"""

import os

# one BLAS/FFT thread per worker: otherwise every worker grabs all cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS  # noqa: E402
from autotiming.grid import Segment, compare  # noqa: E402
from autotiming.osufile import read_osu  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = DATA / "results"


def synth_items():
    for d in sorted((DATA / "synth").glob("s*")):
        if not (d / "truth.json").exists():  # still being generated
            continue
        t = json.loads((d / "truth.json").read_text())
        for f, shift in t["shift_ms"].items():
            # truth is the physical attack; expressed in osu!'s clock like ranked maps are
            shift += OSU_OFFSET_MS
            segs = [Segment(s["time"] + shift, s["beat_length"], s["meter"]) for s in t["segments"]]
            yield {
                "id": f"{d.name}/{f}", "audio": str(d / f), "truth": segs,
                "span": [t["span"][0] + shift, t["span"][1] + shift],
                "tags": {"style": t["style"], "enc": f.split(".")[0], "segments": len(segs)},
            }


def ranked_items(root=None):
    for d in sorted((root or DATA / "sets").glob("*")):
        if not (d / "ok").exists():
            continue
        by_audio = defaultdict(list)
        for f in d.glob("*.osu"):
            try:
                o = read_osu(f)
            except Exception:
                continue
            if o.red_lines and o.first_object is not None:
                by_audio[o.audio].append(o)
        for audio, diffs in by_audio.items():
            if not (d / audio).exists():
                continue
            key = lambda o: tuple((round(r.time), round(r.beat_length, 3)) for r in o.red_lines)
            common, _ = Counter(key(o) for o in diffs).most_common(1)[0]
            host = next(o for o in diffs if key(o) == common)
            agree = sum(key(o) == common for o in diffs) / len(diffs)
            n = len(host.red_lines)
            yield {
                "id": f"{d.name}/{audio}", "audio": str(d / audio),
                "truth": [Segment(r.time, r.beat_length, r.meter) for r in host.red_lines],
                "span": [min(o.first_object for o in diffs), max(o.last_object for o in diffs)],
                "tags": {
                    "set": int(d.name), "segments": n, "kind": "single" if n == 1 else ("few" if n <= 5 else "many"),
                    "ext": Path(audio).suffix.lower(), "diff_agree": agree, "bpm": round(host.red_lines[0].bpm, 3),
                },
            }


def site_items(root=None):
    """Fixes visitors sent from the editor (web/api.py POST /feedback, copied here by
    bench/fetch_site.sh): <audio md5>/audio.<ext> + fix-*.json; the latest fix is the truth."""
    for d in sorted((root or DATA / "site").glob("*")):
        audio = next(d.glob("audio.*"), None)
        fixes = sorted(d.glob("fix-*.json"))
        if audio is None or not fixes:
            continue
        rec = json.loads(fixes[-1].read_text())
        truth = [Segment(l["time"], l["beat_length"], l["meter"]) for l in sorted(rec["fixed"], key=lambda l: l["time"])]
        n = len(truth)
        yield {
            "id": f"{d.name}/{audio.name}", "audio": str(audio), "truth": truth,
            "span": [max(0.0, truth[0].time), rec["seconds"] * 1000],
            "tags": {"segments": n, "kind": "single" if n == 1 else ("few" if n <= 5 else "many"),
                     "ext": audio.suffix.lower(), "fixes": len(fixes), "note": rec.get("note", "")},
        }


DATASETS = {
    "synth": synth_items,
    "ranked": ranked_items,
    # sets nominated by auneu (taiko BN): offsets the user vouches for more than average ranked ones
    "nominated": lambda: ranked_items(DATA / "nominated"),
    # maps mappers reported after testing the bot by hand (bench/make_feedback.py)
    "feedback": lambda: ranked_items(DATA / "feedback"),
    # ranked sets picked for their many tempo changes (bench/fetch_heavy.py)
    "heavy": lambda: ranked_items(DATA / "heavy"),
    # songs visitors corrected on auneu.space/timing (bench/fetch_site.sh): what the detector gets wrong
    "site": site_items,
}


def run_group(method, items):
    """One detection per audio file, scored against every item on it: a song ranked in
    several mapsets is often the very same file, and the detector is deterministic.
    The repeats get secs 0, so summed secs stay the CPU time spent."""
    t0 = time.time()
    try:
        pred, err = METHODS[method](items[0]["audio"]), None
    except Exception as e:
        pred, err = None, {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=3)}
    secs = round(time.time() - t0, 2)
    rows = []
    for k, item in enumerate(items):
        res = dict(err) if err else None
        if res is None:
            try:
                res = compare(pred, item["truth"], *item["span"])
                res["pred"] = [[s.time, s.beat_length] for s in pred]
                res["pred_meter"] = [s.meter for s in pred]
            except Exception as e:
                res = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(limit=3)}
        res.update(id=item["id"], tags=item["tags"], secs=secs if k == 0 else 0.0,
                   truth=[[s.time, s.beat_length] for s in item["truth"]])
        if k:
            res["same_audio_as"] = items[0]["id"]
        rows.append(res)
    return rows


def by_audio(items):
    """Items grouped by the md5 of their audio file, in first-seen order."""
    from concurrent.futures import ThreadPoolExecutor

    md5 = lambda it: hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
    with ThreadPoolExecutor(8) as ex:
        keys = list(ex.map(md5, items))
    groups = defaultdict(list)
    for key, it in zip(keys, items):
        groups[key].append(it)
    return list(groups.values())


def background():
    """Bench workers only take CPU nobody else wants (SCHED_IDLE), so the desktop stays smooth."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        os.nice(19)


TAG = os.environ.get("BENCH_TAG", "")  # results of a variant go to their own file


def load(dataset, method):
    f = RESULTS / f"{dataset}_{method}{TAG}.jsonl"
    return [json.loads(l) for l in f.open()] if f.exists() else []


def summarize(rows, label):
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        print(f"  {label:28} n={len(rows)} (no valid results)")
        return
    octs = Counter(r["octave"] for r in ok)
    good = [r for r in ok if r["octave"] in (0.5, 1, 2)]
    p95 = np.array([r["p95_ms"] for r in good]) if good else np.array([np.inf])
    share = lambda x: f"{100 * np.mean(p95 < x) * len(good) / len(rows):5.1f}%"
    bias = np.median([r["bias_ms"] for r in good if r["p95_ms"] < 20]) if any(r["p95_ms"] < 20 for r in good) else float("nan")
    cover = lambda k: np.sum([r.get(k, 0) for r in ok]) / len(rows)
    print(f"  {label:28} n={len(rows):4}  grid p95<5ms {share(5)}  <10ms {share(10)}  <20ms {share(20)}"
          f"   song covered ±5ms {cover('cover5'):4.0%} ±10ms {cover('cover10'):4.0%}"
          f"   tempo: x1 {octs[1]:3} x2/half {octs[2] + octs[0.5]:3} wrong {octs[0]:3} fail {len(rows) - len(ok):3}"
          f"   bias(good) {bias:+.1f}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--methods", nargs="+", default=["ours"])
    p.add_argument("--workers", type=int, default=12)  # SCHED_IDLE: they only take spare CPU
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--rescore", action="store_true", help="recompute metrics from cached predictions")
    p.add_argument("--by", help="tag to break the summary down by")
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="file with one item id per line: run only these (a quick screen before the full set)")
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = list(DATASETS[args.dataset]())[: args.limit]
    if args.ids:
        keep = set(Path(args.ids).read_text().split("\n")) - {""}
        items = [it for it in items if it["id"] in keep]
    if not args.summary_only:
        for method in args.methods:
            done = {r["id"] for r in load(args.dataset, method)}
            todo = [it for it in items if it["id"] not in done]
            groups = by_audio(todo)
            print(f"{method}: {len(todo)} to run, {len(groups)} distinct audio files", flush=True)
            # longest songs first: a 17-minute marathon started last would leave one worker running alone
            groups.sort(key=lambda g: -Path(g[0]["audio"]).stat().st_size)
            with ProcessPoolExecutor(args.workers, initializer=background) as ex, (RESULTS / f"{args.dataset}_{method}{TAG}.jsonl").open("a") as out:
                futs = [ex.submit(run_group, method, g) for g in groups]
                for i, f in enumerate(as_completed(futs), 1):
                    for row in f.result():
                        out.write(json.dumps(row) + "\n")
                    out.flush()
                    if i % 50 == 0:
                        print(f"  {i}/{len(groups)}", flush=True)
    ids = {it["id"] for it in items}
    if args.rescore:  # metric changed: recompute from cached predictions, no detection
        by_id = {it["id"]: it for it in items}
        for method in args.methods:
            f = RESULTS / f"{args.dataset}_{method}{TAG}.jsonl"
            rows = load(args.dataset, method)
            with f.open("w") as out:
                for r in rows:
                    it = by_id.get(r["id"])
                    if it and r.get("pred"):
                        res = compare([Segment(t, L) for t, L in r["pred"]], it["truth"], *it["span"])
                        r = {**r, **res}
                    out.write(json.dumps(r) + "\n")
    for method in args.methods:
        rows = [r for r in load(args.dataset, method) if r["id"] in ids]
        print(f"{method}:")
        summarize(rows, "all")
        if args.by:
            groups = defaultdict(list)
            for r in rows:
                groups[str(r["tags"].get(args.by))].append(r)
            for k in sorted(groups):
                summarize(groups[k], f"{args.by}={k}")


if __name__ == "__main__":
    main()
