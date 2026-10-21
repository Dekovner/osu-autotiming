"""Run trees: learn when the hybrid should take the tracker's beats over the steady grid.

    uv run python bench/run_trees_exp.py rows ranked heavy nominated feedback   # data/run_rows.jsonl
    uv run python bench/run_trees_exp.py cv                                     # grouped 5-fold, coverage gained
    uv run python bench/run_trees_exp.py crossfit data/run_crossfit             # per-fold models for honest benches
    uv run python bench/run_trees_exp.py export                                 # src/autotiming/models/run_trees.npz

Every candidate run of track.hybrid (local gain over the steady grid > HYBRID_MARGIN) is
one row: its features (track.run_features), the rule's verdict and, from the mapper's
red lines, how many true beats in the run the tracked beats and the steady grid each put
within 10 ms. The trees are trained on "tracked covers more", each run weighted by the
beats at stake, so they buy coverage where it is.
Steady lines come from the crossfit octave trees (OCTAVE_CROSSFIT_DIR), as in the bench.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OCTAVE_CROSSFIT", "1")
os.environ.setdefault("OCTAVE_CROSSFIT_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "octave_crossfit_sec"))
os.environ["RUN_TREE"] = "0"  # rows are the rule's candidates, never a model's

import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATA, DATASETS, background  # noqa: E402
from source_dump import crossfit  # noqa: E402

ROWS = Path(os.environ.get("RUN_ROWS", DATA / "run_rows.jsonl"))
TREES = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=20, random_state=0)
META = ("id", "group", "dataset", "kind", "a", "b", "cov_t", "cov_s", "n_true", "rule", "p", "q")


def covered(tb, src, tol=10.0):
    if len(src) == 0 or len(tb) == 0:
        return np.zeros(len(tb), bool)
    j = np.clip(np.searchsorted(src, tb), 1, max(1, len(src) - 1))
    lo = src[np.maximum(j - 1, 0)]
    hi = src[np.minimum(j, len(src) - 1)]
    return np.minimum(np.abs(tb - lo), np.abs(tb - hi)) < tol


def split(b, q):
    """Each interval of beat list b cut into q equal parts."""
    if q == 1 or len(b) < 2:
        return b
    return np.concatenate([b[:-1, None] + np.diff(b)[:, None] * np.arange(q)[None, :] / q, b[-1:, None]], axis=None)


# mapper's beat length / our steady one, as p:q: compared on the common tick (truth / p, ours / q).
# 3:2 is a song triplet_beats() rescales after the hybrid (216 steady, mapper 144).
RATIOS = ((1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (4, 3), (3, 4), (4, 1), (1, 4))


def score(truth_ticks, src_ticks, tol=10.0):
    """Mapper's ticks we hit, minus ours that hit nothing: a denser grid doesn't win by chance."""
    return int(covered(truth_ticks, src_ticks, tol).sum()) - int((~covered(src_ticks, truth_ticks, tol)).sum())


def one(job):
    ds, items = job
    from autotiming import detect as D
    from autotiming import track as T
    from autotiming.beatprob import beat_prob
    from autotiming.downbeat import harmony
    from autotiming.grid import beats
    from cache_onsets import exact_onsets

    path = items[0]["audio"]
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    try:
        crossfit(path)
        pcm, sr = D.decode_bass(path)
        o = exact_onsets(path, pcm, sr)
        if o.profile is None or o.times[-1] - o.times[0] < D.TRACK_MIN_MS:
            return []
        ch = harmony(pcm, sr)
        steady, _ = D.steady_phase(o, pcm, sr, ch.chroma, D.detect_onsets(o, {}))
        main_L = D.main_beat_length(o, steady)
        secs = D.align_octave(T.fine_sections(o, main_L), steady)
        tracked = T.track_beats(o, secs, probs=beat_prob(o, main_L))
        phys = sorted([D.Segment(s.time - D.OSU_OFFSET_MS, s.beat_length) for s in steady], key=lambda s: s.time)
        thirds = [(a.time, b.time) for a, b in zip(phys, phys[1:]) if D.interlock(a, b)]
        T.RUN_LOG = []
        T.hybrid(o, phys, tracked, thirds=thirds)
        runs, T.RUN_LOG = T.RUN_LOG, None
    except Exception as e:  # one broken song shouldn't stop the rows
        print(f"  {path}: {e!r}", flush=True)
        return []
    off = D.OSU_OFFSET_MS
    tr = tracked.times + off
    out = []
    for it in items:
        truth = sorted(it["truth"], key=lambda s: s.time)
        s0, e0 = it["span"]
        starts = np.array([x.time for x in truth])
        for r in runs:
            a, b = r["a"] + off, r["b"] + off
            lo, hi = max(a, s0), min(b, e0)
            if hi - lo < 200:
                continue
            k = max(0, int(np.searchsorted(starts, (lo + hi) / 2, side="right")) - 1)
            tbl = truth[k].beat_length
            sg = beats(steady, lo - 2000, hi + 2000)
            sL = float(np.median(np.diff(sg))) if len(sg) > 1 else tbl
            pq = min(RATIOS, key=lambda x: abs(np.log(tbl / sL * x[1] / x[0])))
            if abs(np.log(tbl / sL * pq[1] / pq[0])) > 0.03:  # the mapper's tempo here is none of ours
                continue
            p, q = pq
            inside = lambda x: x[(x >= lo) & (x <= hi)]
            tb = inside(split(beats(truth, lo - 2 * tbl, hi + 2 * tbl), p))
            st, tk = inside(split(sg, q)), inside(split(tr, q))
            if len(tb) < 2:
                continue
            out.append({"id": it["id"], "group": md5, "dataset": ds, "kind": it["tags"]["kind"],
                        "cov_t": score(tb, tk), "cov_s": score(tb, st), "n_true": len(tb), "p": p, "q": q,
                        **{k2: (v if np.isfinite(v) else 1e3) for k2, v in r.items()}})
    return out


def cmd_rows(datasets, workers=12):
    jobs = []
    for ds in datasets:
        by = defaultdict(list)
        for it in DATASETS[ds]():
            by[hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()].append(it)
        jobs += [(ds, v) for v in by.values()]
    jobs.sort(key=lambda j: -Path(j[1][0]["audio"]).stat().st_size)
    print(f"{len(jobs)} songs", flush=True)
    n = 0
    with ProcessPoolExecutor(workers, initializer=background) as ex, ROWS.open("w") as f:
        for i, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            for r in fut.result():
                f.write(json.dumps(r) + "\n")
                n += 1
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} songs, {n} runs", flush=True)


def load_rows():
    # RUN_ROWS_IN: comma-separated row files to train / score on together (default: RUN_ROWS)
    files = os.environ.get("RUN_ROWS_IN", str(ROWS)).split(",")
    rows = [json.loads(l) for f in files for l in open(f)]
    names = sorted(k for k in rows[0] if k not in META)
    return rows, names


def matrix(rows, names):
    X = np.array([[r[n] for n in names] for r in rows], float)
    y = np.array([r["cov_t"] > r["cov_s"] for r in rows])
    w = np.array([abs(r["cov_t"] - r["cov_s"]) for r in rows], float) + 0.1
    w[~y] *= float(os.environ.get("RUN_NEG_W", "1"))  # a wrong switch costs more than a missed one (it can break a clean map)
    return X, y, w


def folds(rows, k=5):
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    return {g: i % k for i, g in enumerate(groups)}


def report(rows, accept, label):
    by = defaultdict(lambda: np.zeros(3))
    for r, a in zip(rows, accept):
        d = r["cov_t"] - r["cov_s"]
        key = (r["dataset"], r["kind"])
        by[key] += [d if a else 0, max(d, 0), 1]
    print(f"  {label}")
    for key in sorted(by):
        g, best, n = by[key]
        print(f"    {key[0]:9} {key[1]:7} runs {n:5.0f}  beats gained {g:+7.0f}  (oracle {best:+7.0f})")


def cmd_cv(thr=0.5):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows, names = load_rows()
    train = [r for r in rows if r["dataset"] in ("ranked", "heavy")]
    X, y, w = matrix(train, names)
    fa = folds(train)
    fold = np.array([fa[r["group"]] for r in train])
    p = np.zeros(len(train))
    for f in range(5):
        m = HistGradientBoostingClassifier(**TREES).fit(X[fold != f], y[fold != f], sample_weight=w[fold != f])
        p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    report(train, [r["rule"] > 0 for r in train], "rule")
    for t in (0.4, 0.5, 0.6, 0.7):
        report(train, p >= t, f"trees (5-fold) p >= {t}")
    rule = np.array([r["rule"] > 0 for r in train])
    for t in (0.6, 0.7, 0.8):
        report(train, rule | (p >= t), f"rule or trees p >= {t}")
        report(train, (rule & (p >= 0.3)) | (p >= t), f"(rule and p >= 0.3) or p >= {t}")
    if os.environ.get("RUN_CV_LIST"):  # worst runs the trees take on single-line maps
        bad = sorted((r["cov_t"] - r["cov_s"], r["id"], round(r["a"]), pi, {n: round(r[n], 3) for n in names})
                     for r, pi in zip(train, p) if pi >= thr and r["kind"] == "single" and r["cov_t"] < r["cov_s"])
        for b in bad[:15]:
            print("   ", b)
    m = HistGradientBoostingClassifier(**TREES).fit(X, y, sample_weight=w)
    other = [r for r in rows if r["dataset"] not in ("ranked", "heavy")]
    if other:
        Xo, _, _ = matrix(other, names)
        po = m.predict_proba(Xo)[:, 1]
        report(other, [r["rule"] > 0 for r in other], "rule (held-out sets)")
        report(other, po >= thr, f"trees on all ranked+heavy, p >= {thr}")


def fit_export(rows, names, path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from octave_trees_exp import export

    X, y, w = matrix(rows, names)
    m = HistGradientBoostingClassifier(**TREES).fit(X, y, sample_weight=w)
    export(m, path, names=names)


def cmd_export():
    from autotiming.track import RUN_TREES

    rows, names = load_rows()
    fit_export([r for r in rows if r["dataset"] in ("ranked", "heavy")], names, RUN_TREES)
    print(f"wrote {RUN_TREES}")


def cmd_crossfit(dest):
    """One model per fold, each song's fold in assign.json (bench: RUN_CROSSFIT=1)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows, names = load_rows()
    train = [r for r in rows if r["dataset"] in ("ranked", "heavy")]
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
