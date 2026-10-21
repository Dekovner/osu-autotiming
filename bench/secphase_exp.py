"""Section phase trees: should this red line's section move by half a beat (beat <-> "and")?

    uv run python bench/secphase_exp.py rows ranked:_fast1 heavy:_hbase nominated:_nbase feedback:_nbase
    uv run python bench/secphase_exp.py cv
    uv run python bench/secphase_exp.py crossfit data/secphase_crossfit
    uv run python bench/secphase_exp.py export

Rows are the final red lines of a bench run (dataset:tag): per section, detect.section_phase_features
on its span and, from the mapper's lines, how many of their beats in it our grid puts within
10 ms as it is (cov_keep) and moved by half a beat (cov_flip). A perfect choice here is worth
114 -> 122 "ready as is" maps on ranked few-line maps, 242 -> 253 on heavy.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATA, DATASETS, RESULTS, background  # noqa: E402
from source_dump import crossfit  # noqa: E402

ROWS = Path(os.environ.get("SECPHASE_ROWS", DATA / "secphase_rows.jsonl"))
TREES = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=20, random_state=0)
META = ("id", "group", "dataset", "kind", "i", "time", "cov_keep", "cov_flip", "n_true")


def covered(tb, grid, tol=10.0):
    if len(grid) == 0 or len(tb) == 0:
        return np.zeros(len(tb), bool)
    j = np.clip(np.searchsorted(grid, tb), 1, max(1, len(grid) - 1))
    return np.minimum(np.abs(tb - grid[j - 1]), np.abs(tb - grid[np.minimum(j, len(grid) - 1)])) < tol


def one(job):
    ds, items, pred = job
    from autotiming import detect as D
    from autotiming.downbeat import harmony
    from autotiming.grid import Segment, beats
    from cache_onsets import exact_onsets

    path = items[0]["audio"]
    try:
        crossfit(path)  # PHASE_CROSSFIT=1: the phase model's opinion is one that never saw this song
        pcm, sr = D.decode_bass(path)
        o = exact_onsets(path, pcm, sr)
        ch = harmony(pcm, sr)
        segs = sorted((Segment(t, L) for t, L in pred), key=lambda s: s.time)
        feats = D.section_phase_features(o, pcm, sr, ch.chroma, segs)
    except Exception as e:
        print(f"  {path}: {e!r}", flush=True)
        return []
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    out = []
    for it in items:
        s0, e0 = it["span"]
        truth = sorted(it["truth"], key=lambda x: x.time)
        for i, f in feats.items():
            a = max(segs[i].time, s0)
            b = min(segs[i + 1].time if i + 1 < len(segs) else e0, e0)
            if b - a < 4 * segs[i].beat_length:
                continue
            tb = beats(truth, a, b)
            L = segs[i].beat_length
            keep = segs[i].time + np.arange(np.floor((a - segs[i].time) / L) - 1, np.ceil((b - segs[i].time) / L) + 2) * L
            ck, cf = covered(tb, keep).sum(), covered(tb, keep + L / 2).sum()
            out.append({"id": it["id"], "group": md5, "dataset": ds, "kind": it["tags"]["kind"], "i": i,
                        "time": segs[i].time, "cov_keep": int(ck), "cov_flip": int(cf), "n_true": len(tb),
                        **{k: (float(v) if v is not None and np.isfinite(v) else 0.0) for k, v in f.items()}})
    return out


def cmd_rows(specs, workers=14):
    jobs = []
    for spec in specs:
        ds, tag = spec.split(":")
        res = {r["id"]: r for r in map(json.loads, (RESULTS / f"{ds}_ours{tag}.jsonl").open())}
        by = defaultdict(list)
        for it in DATASETS[ds]():
            r = res.get(it["id"])
            if r and r.get("pred"):
                by[(it["audio"], json.dumps(r["pred"]))].append(it)
        jobs += [(ds, v, json.loads(k[1])) for k, v in by.items()]
    jobs.sort(key=lambda j: -Path(j[1][0]["audio"]).stat().st_size)
    print(f"{len(jobs)} songs", flush=True)
    n = 0
    with ProcessPoolExecutor(workers, initializer=background) as ex, ROWS.open("w") as f:
        for i, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            for r in fut.result():
                f.write(json.dumps(r) + "\n")
                n += 1
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} songs, {n} sections", flush=True)


def load_rows():
    rows = [json.loads(l) for l in ROWS.open()]
    names = sorted(k for k in rows[0] if k not in META)
    return rows, names


def matrix(rows, names):
    X = np.array([[r[n] for n in names] for r in rows], float)
    y = np.array([r["cov_flip"] > r["cov_keep"] for r in rows])
    w = np.array([abs(r["cov_flip"] - r["cov_keep"]) for r in rows], float) + 0.1
    w[np.array([r["kind"] == "single" for r in rows])] *= float(os.environ.get("SECPHASE_SINGLE_W", "1"))
    return X, y, w


def folds(rows, k=5):
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    return {g: i % k for i, g in enumerate(groups)}


TRAIN = ("ranked", "heavy")


def report(rows, flip, label):
    by = defaultdict(lambda: np.zeros(4))
    for r, fl in zip(rows, flip):
        d = r["cov_flip"] - r["cov_keep"]
        by[(r["dataset"], r["kind"])] += [d if fl else 0, max(d, 0), fl, 1]
    print(f"  {label}")
    for key in sorted(by):
        g, best, nf, n = by[key]
        print(f"    {key[0]:9} {key[1]:7} sections {n:5.0f} flipped {nf:4.0f}  beats gained {g:+7.0f}  (oracle {best:+7.0f})")


def cmd_cv():
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows, names = load_rows()
    train = [r for r in rows if r["dataset"] in TRAIN]
    X, y, w = matrix(train, names)
    fa = folds(train)
    fold = np.array([fa[r["group"]] for r in train])
    p = np.zeros(len(train))
    for f in range(5):
        m = HistGradientBoostingClassifier(**TREES).fit(X[fold != f], y[fold != f], sample_weight=w[fold != f])
        p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    if os.environ.get("SECPHASE_MAIN_MIN_SECS"):  # the main section of a timing with fewer lines keeps its phase
        k = int(os.environ["SECPHASE_MAIN_MIN_SECS"])
        nm = np.array([r["is_main"] == 0 or r["n_secs"] >= k for r in train])
        p = np.where(nm, p, 0.0)
    base = np.array([r["phase_p"] for r in train])
    report(train, (base < 0.5) & (base > 0), "phase model alone, p(keep) < 0.5")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        report(train, p >= t, f"section trees (5-fold) p(flip) >= {t}")
    other = [r for r in rows if r["dataset"] not in TRAIN]
    if other:
        m = HistGradientBoostingClassifier(**TREES).fit(X, y, sample_weight=w)
        Xo, _, _ = matrix(other, names)
        po = m.predict_proba(Xo)[:, 1]
        for t in (0.7, 0.8):
            report(other, po >= t, f"held-out sets, trees on all of {TRAIN}, p >= {t}")


def fit_export(rows, names, path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from octave_trees_exp import export

    X, y, w = matrix(rows, names)
    export(HistGradientBoostingClassifier(**TREES).fit(X, y, sample_weight=w), path, names=names)


def cmd_export():
    from autotiming.detect import SECPHASE_TREES

    rows, names = load_rows()
    fit_export([r for r in rows if r["dataset"] in TRAIN], names, SECPHASE_TREES)
    print(f"wrote {SECPHASE_TREES}")


def cmd_crossfit(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows, names = load_rows()
    train = [r for r in rows if r["dataset"] in TRAIN]
    fa = folds(train)
    for f in range(5):
        fit_export([r for r in train if fa[r["group"]] != f], names, dest / f"fold{f}.npz")
    (dest / "assign.json").write_text(json.dumps(fa))
    print(f"wrote {dest}: 5 folds, {len(fa)} songs")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rows":
        cmd_rows(sys.argv[2:])
    elif cmd == "cv":
        cmd_cv()
    elif cmd == "crossfit":
        cmd_crossfit(sys.argv[2])
    elif cmd == "export":
        cmd_export()
