"""Which tempo octave do mappers pick? Boosted trees over the grid features.

    uv run python bench/octave_trees_exp.py rows [workers]   # features for every ranked map (needs raw_segments.json)
    uv run python bench/octave_trees_exp.py cv               # grouped 5-fold CV: rule vs logistic vs trees
    uv run python bench/octave_trees_exp.py final            # train on all, export models/octave_trees.npz
    uv run python bench/octave_trees_exp.py crossfit         # 5 fold models for an honest bench (OCTAVE_CROSSFIT=1)

Candidates are the main raw red line at x0.5 / x1 / x2; the label is the one
matching the ranked tempo there. Folds are grouped by audio hash.

Result (733 maps): nearest-175 rule 81.6%, logistic (previous) 85.7%, trees 89.5%.
Pulse features (envelope autocorrelation at 1/2/half beats, kick spacing, strong
hits per beat) were tried and didn't add anything (89.5% either way).
"""

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from octave_exp import RAW, main_segment  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming.detect import OCTAVE_KEYS, OCTAVE_KS, OCTAVE_TREES, octave_candidates, octave_features, octave_matrix  # noqa: E402

OUT = DATA / "octave_tree_rows.json"
KS = OCTAVE_KS
OLD = OCTAVE_KEYS
TREES = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=0)


def one(args):
    it, segs = args
    try:
        o = load_onsets(it["id"])
        seg = main_segment(segs, o.times[-1])
        i = segs.index(seg)
        lo = seg[0] if i > 0 else o.times[0]
        hi = segs[i + 1][0] if i + 1 < len(segs) else o.times[-1] + 1
        truth = sorted(it["truth"], key=lambda s: s.time)
        best, best_ov = None, -1
        for k, s in enumerate(truth):  # ranked tempo in effect over that span (longest overlap)
            e = truth[k + 1].time if k + 1 < len(truth) else hi
            ov = min(e, hi) - max(s.time if k else -1e9, lo)
            if ov > best_ov:
                best, best_ov = s, ov
        cands = octave_candidates(o, seg[0], seg[1], lo, hi)
        label = next((k for k in cands if abs(np.log((seg[1] / k) / best.beat_length)) < 0.02), None)
        if label is None:
            return None
        feats = {str(k): {f: float(v) for f, v in octave_features(o, p, L, lo, hi).items()} for k, (p, L) in cands.items()}
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        return {"id": it["id"], "group": md5, "label": float(label), "kind": it["tags"]["kind"], "feats": feats}
    except Exception as e:
        return {"id": it["id"], "error": repr(e)}


def cmd_rows(workers):
    raw = json.loads(RAW.read_text())
    todo = [(it, raw[it["id"]]) for it in ranked_items() if path_for(it["id"]).exists() and isinstance(raw.get(it["id"]), list) and raw[it["id"]]]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, todo, chunksize=4))
    errs = [r for r in rows if r and "error" in r]
    rows = [r for r in rows if r and "error" not in r]
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} rows, {len(errs)} errors{': ' + errs[0]['error'] if errs else ''}")


def matrix(r):
    return octave_matrix({k: r["feats"][str(k)] for k in KS})


def cmd_cv():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rows = json.loads(OUT.read_text())
    y = np.array([[r["label"] == k for k in KS] for r in rows], int)
    groups = np.array([r["group"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    rule = np.array([np.argmin([abs(r["feats"][str(k)]["log2_rel"]) for k in KS]) for r in rows])
    rule_ok = y[np.arange(len(rows)), rule] == 1
    folds = list(GroupKFold(5).split(rows, groups=groups))

    def run(make, use_matrix=True):
        X = np.stack([matrix(r) if use_matrix else np.nan_to_num(np.array([[r["feats"][str(k)][f] for f in OLD] for k in KS])) for r in rows])
        ok = np.zeros(len(rows), bool)
        for tr, te in folds:
            Xtr = X[tr].reshape(-1, X.shape[2])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = make().fit((Xtr - mu) / sd, y[tr].ravel())
            p = m.predict_proba(((X[te] - mu) / sd).reshape(-1, X.shape[2]))[:, 1].reshape(len(te), 3)
            ok[te] = y[te][np.arange(len(te)), p.argmax(1)] == 1
        return ok

    hgb = lambda: HistGradientBoostingClassifier(**TREES)
    res = {
        "nearest 175 BPM": rule_ok,
        "logistic (previous)": run(lambda: LogisticRegression(C=1.0, max_iter=3000), use_matrix=False),
        "trees": run(hgb),
    }
    print(f"{len(rows)} maps; label share " + ", ".join(f"x{k}: {y[:, i].mean():.0%}" for i, k in enumerate(KS)))
    for name, ok in res.items():
        parts = "  ".join(f"{g} {ok[kinds == g].mean():.1%}" for g in ("single", "few", "many"))
        print(f"  {name:34} all {ok.mean():.1%}   {parts}")


def cmd_final():
    """Train on every map (no feature scaling: trees don't need it) and export like the beat-probability model."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from autotiming.beatprob import load_model, predict_proba

    rows = json.loads(OUT.read_text())
    X = np.concatenate([matrix(r) for r in rows])
    y = np.array([[r["label"] == k for k in KS] for r in rows], int).ravel()
    m = HistGradientBoostingClassifier(**TREES).fit(X, y)
    export(m, OCTAVE_TREES)
    trees = m._predictors
    load_model.cache_clear()
    diff = np.abs(predict_proba(load_model(OCTAVE_TREES), X) - m.predict_proba(X)[:, 1]).max()
    print(f"{len(trees)} trees -> {OCTAVE_TREES} ({OCTAVE_TREES.stat().st_size // 1024} KB); max |p - sklearn| = {diff:.2e}")


CROSSFIT = DATA / "octave_crossfit"


def cmd_crossfit():
    """One model per grouped fold, trained without that fold's songs, for an honest bench run
    (bench/baselines.py picks the model that never saw the song)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows = json.loads(OUT.read_text())
    X = np.stack([matrix(r) for r in rows])
    y = np.array([[r["label"] == k for k in KS] for r in rows], int)
    groups = np.array([r["group"] for r in rows])
    CROSSFIT.mkdir(exist_ok=True)
    assign = {}
    for f, (tr, te) in enumerate(GroupKFold(5).split(rows, groups=groups)):
        m = HistGradientBoostingClassifier(**TREES).fit(X[tr].reshape(-1, X.shape[2]), y[tr].ravel())
        export(m, CROSSFIT / f"fold{f}.npz")
        assign.update({groups[i]: f for i in te})
    (CROSSFIT / "assign.json").write_text(json.dumps(assign))
    print(f"5 fold models, {len(assign)} songs assigned -> {CROSSFIT}")


def export(m, path, names=None):
    trees = [p[0].nodes for p in m._predictors]
    cat = np.concatenate(trees)
    np.savez_compressed(
        path, n_nodes=np.array([len(t) for t in trees]), feature=cat["feature_idx"].astype(np.int16),
        threshold=cat["num_threshold"], left=cat["left"].astype(np.int32), right=cat["right"].astype(np.int32),
        leaf=cat["is_leaf"].astype(bool), value=cat["value"], missing_left=cat["missing_go_to_left"].astype(bool),
        baseline=np.atleast_1d(m._baseline_prediction).ravel()[:1], names=np.array(OLD if names is None else names),
    )


if __name__ == "__main__":
    {"rows": lambda: cmd_rows(int(sys.argv[2]) if len(sys.argv) > 2 else 12), "cv": cmd_cv, "final": cmd_final, "crossfit": cmd_crossfit}[sys.argv[1]]()
