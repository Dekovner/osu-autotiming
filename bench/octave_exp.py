"""Learn which tempo octave mappers pick, from musical features.

    uv run python bench/octave_exp.py raw [workers]   # raw (pre-octave) segments for every cached item
    uv run python bench/octave_exp.py fit             # grouped-CV logistic regression vs the 175-BPM rule

For each item the main red line's tempo b gives candidates b/2, b, 2b; the label
is whichever matches the ranked tempo. Folds are grouped by audio hash so the
same song never sits in both train and test.
"""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming.detect import _detect_raw, octave_candidates, octave_features  # noqa: E402

RAW = DATA / "raw_segments.json"
C_REG = float(os.environ.get("OCTAVE_C", "1.0"))
def raw_one(it):
    try:
        o = load_onsets(it["id"])
        segs = _detect_raw(o)
        return it["id"], [[s.time, s.beat_length] for s in segs]
    except Exception as e:
        return it["id"], str(e)


def cmd_raw(workers):
    items = [it for it in ranked_items() if path_for(it["id"]).exists()]
    out = json.loads(RAW.read_text()) if RAW.exists() else {}
    todo = [it for it in items if it["id"] not in out]
    print(f"{len(todo)} to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (k, v) in enumerate(ex.map(raw_one, todo, chunksize=1), 1):
            out[k] = v
            if i % 25 == 0 or i == len(todo):
                RAW.write_text(json.dumps(out))  # partial results survive an interrupted run
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def main_segment(segs, t_end):
    spans = [((segs[i + 1][0] if i + 1 < len(segs) else t_end) - segs[i][0]) for i in range(len(segs))]
    return segs[int(np.argmax(spans))]


def candidates(o, seg, lo, hi):
    return {k: octave_features(o, p, l, lo, hi) for k, (p, l) in octave_candidates(o, seg[0], seg[1], lo, hi).items()}


def build():
    raw = json.loads(RAW.read_text())
    rows = []
    for it in ranked_items():
        segs = raw.get(it["id"])
        if not isinstance(segs, list) or not segs:
            continue
        o = load_onsets(it["id"])
        seg = main_segment(segs, o.times[-1])
        i = [k for k in range(len(segs)) if segs[k] == seg][0]
        lo = seg[0] if i > 0 else o.times[0]
        hi = segs[i + 1][0] if i + 1 < len(segs) else o.times[-1] + 1
        # ranked tempo in effect over that span (longest overlap)
        truth = sorted(it["truth"], key=lambda s: s.time)
        best, best_ov = None, -1
        for k, s in enumerate(truth):
            e = truth[k + 1].time if k + 1 < len(truth) else hi
            ov = min(e, hi) - max(s.time if k else -1e9, lo)
            if ov > best_ov:
                best, best_ov = s, ov
        cands = candidates(o, seg, lo, hi)
        label = None
        for k in cands:
            if abs(np.log((seg[1] / k) / best.beat_length)) < 0.02:
                label = k
        if label is None:
            continue
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        rows.append({"id": it["id"], "group": md5, "label": label, "bpm": 60000 / seg[1], "cands": cands,
                     "kind": it["tags"]["kind"]})
    return rows


def cmd_fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rows = build()
    keys = sorted(rows[0]["cands"][1.0])
    print(f"{len(rows)} items with a matching octave; label share: "
          + ", ".join(f"x{k}: {np.mean([r['label'] == k for r in rows]):.0%}" for k in (0.5, 1.0, 2.0)))

    def X_of(r):
        return np.array([[r["cands"][k][f] for f in keys] for k in (0.5, 1.0, 2.0)])

    X = np.stack([X_of(r) for r in rows])  # items x 3 candidates x features
    y = np.array([[r["label"] == k for k in (0.5, 1.0, 2.0)] for r in rows], dtype=int)
    groups = [r["group"] for r in rows]
    rule = np.array([min((0.5, 1.0, 2.0), key=lambda k: abs(np.log(r["bpm"] * k / 175.0))) == r["label"] for r in rows])
    pred_ok = np.zeros(len(rows), bool)
    for tr, te in GroupKFold(5).split(X, groups=groups):
        mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-9
        clf = LogisticRegression(C=C_REG, max_iter=2000).fit(((X[tr] - mu) / sd).reshape(-1, X.shape[2]), y[tr].ravel())
        s = clf.decision_function(((X[te] - mu) / sd).reshape(-1, X.shape[2])).reshape(len(te), 3)
        pred_ok[te] = y[te][np.arange(len(te)), s.argmax(1)] == 1
    kinds = np.array([r["kind"] for r in rows])
    for name, m in (("all", np.ones(len(rows), bool)), ("single", kinds == "single"), ("few", kinds == "few"), ("many", kinds == "many")):
        print(f"  {name:7} n={m.sum():4}  175-rule {rule[m].mean():.1%}   learned (CV) {pred_ok[m].mean():.1%}")
    # final fit on everything, for the detector
    Xa = X.reshape(-1, X.shape[2])
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    clf = LogisticRegression(C=C_REG, max_iter=2000).fit((Xa - mu) / sd, y.ravel())
    coef = (clf.coef_[0] / sd).tolist()
    model = {"features": keys, "coef": coef, "intercept": float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())}
    (DATA / "octave_model.json").write_text(json.dumps(model, indent=1))
    for f, c in sorted(zip(model["features"], coef), key=lambda x: -abs(x[1]))[:10]:
        print(f"    {f:14} {c:+.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "raw":
        cmd_raw(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        cmd_fit()
