"""Is the beat on the grid or half a beat later? Boosted trees over phase_features.

    uv run python bench/phase_trees_exp.py rows [workers]   # features at the mapper's grid, every ranked map
    uv run python bench/phase_trees_exp.py cv               # grouped 5-fold CV: kick rule vs logistic vs trees
    uv run python bench/phase_trees_exp.py final            # train on all, export models/phase_trees.npz
    uv run python bench/phase_trees_exp.py crossfit         # 5 fold models for an honest bench (PHASE_CROSSFIT=1)

Each map's longest ranked red line gives the true beat phase; the features are
"on the beat minus on the and" at that grid, so the label is always "keep".
Trees see every vector and its negation (labels 1 / 0), so the candidate order
carries no information. Folds are grouped by audio hash.
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
from run import DATA, ranked_items  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.detect import OSU_OFFSET_MS, PHASE_COEF_LOUD, PHASE_COEF_SPECTRAL, PHASE_TREES, phase_features  # noqa: E402
from autotiming.downbeat import chroma  # noqa: E402

OUT = DATA / "phase_tree_rows.json"
TREES = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=0)


def one(it):
    try:
        if not path_for(it["id"]).exists():
            return None
        truth = sorted(it["truth"], key=lambda s: s.time)
        ends = [s.time for s in truth[1:]] + [it["span"][1]]
        k = int(np.argmax([e - s.time for s, e in zip(truth, ends)]))
        seg = truth[k]
        lo, hi = max(seg.time, it["span"][0]), ends[k]
        if hi - lo < 16 * seg.beat_length:
            return None
        o = load_onsets(it["id"])
        pcm, sr = decode_bass(it["audio"])
        f = phase_features(o, pcm, sr, chroma(pcm, sr), seg.time - OSU_OFFSET_MS, seg.beat_length,
                           lo - OSU_OFFSET_MS, hi - OSU_OFFSET_MS)
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        return {"id": it["id"], "group": md5, "kind": it["tags"]["kind"], "f": {k: float(v) for k, v in f.items()}}
    except Exception as e:
        return {"id": it["id"], "error": repr(e)}


def cmd_rows(workers):
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, list(ranked_items()), chunksize=2))
    errs = [r for r in rows if r and "error" in r]
    rows = [r for r in rows if r and "error" not in r]
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} rows, {len(errs)} errors{': ' + errs[0]['error'] if errs else ''}")


def data():
    rows = json.loads(OUT.read_text())
    keys = sorted(rows[0]["f"])
    D = np.nan_to_num(np.array([[r["f"][k] for k in keys] for r in rows]))
    return rows, keys, D


def cmd_cv():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows, keys, D = data()
    groups = np.array([r["group"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    kick = D[:, keys.index("d_kick")] > 0
    logit = sum(c * D[:, keys.index("d_" + k)] for k, c in PHASE_COEF_SPECTRAL.items() if "d_" + k in keys) \
        + sum(c * D[:, keys.index(k)] for k, c in PHASE_COEF_LOUD.items())
    ok = np.zeros(len(rows), bool)
    p_keep = np.zeros(len(rows))
    for tr, te in GroupKFold(5).split(D, groups=groups):
        X = np.concatenate([D[tr], -D[tr]])
        y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        m = HistGradientBoostingClassifier(**TREES).fit(X, y)
        p = (m.predict_proba(D[te])[:, 1] + 1 - m.predict_proba(-D[te])[:, 1]) / 2
        p_keep[te] = p
        ok[te] = p > 0.5
    print(f"{len(rows)} maps")
    for name, v in (("kick rule (current phase)", kick), ("logistic (current 2nd opinion)", logit > 0), ("trees", ok)):
        parts = "  ".join(f"{g} {v[kinds == g].mean():.1%}" for g in ("single", "few", "many"))
        print(f"  {name:32} all {v.mean():.2%}   {parts}")
    for lo, hi in ((0.35, 0.65), (0.25, 0.75)):
        u = (p_keep > lo) & (p_keep < hi)
        print(f"  trees unsure ({lo}-{hi}): {u.mean():.1%} of maps, holding {(~ok & u).sum()} of {(~ok).sum()} errors")
    m = HistGradientBoostingClassifier(**TREES).fit(np.concatenate([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))])
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(m, np.concatenate([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))], n_repeats=3, random_state=0)
    print("  top features:", ", ".join(f"{keys[i]} {imp.importances_mean[i]:.3f}" for i in np.argsort(-imp.importances_mean)[:8]))


def cmd_final():
    from sklearn.ensemble import HistGradientBoostingClassifier

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from octave_trees_exp import export

    from autotiming.beatprob import load_model
    from autotiming.detect import phase_keep_prob

    rows, keys, D = data()
    m = HistGradientBoostingClassifier(**TREES).fit(np.concatenate([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))])
    export(m, PHASE_TREES, names=keys)
    load_model.cache_clear()
    p = np.array([phase_keep_prob(dict(zip(keys, d))) for d in D[:50]])
    ref = (m.predict_proba(D[:50])[:, 1] + 1 - m.predict_proba(-D[:50])[:, 1]) / 2
    print(f"{len(m._predictors)} trees -> {PHASE_TREES} ({PHASE_TREES.stat().st_size // 1024} KB); max |p - sklearn| = {np.abs(p - ref).max():.2e}")


CROSSFIT = DATA / "phase_crossfit"


def cmd_crossfit():
    """One model per grouped fold, without that fold's songs, for an honest bench run (PHASE_CROSSFIT=1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from octave_trees_exp import export

    rows, keys, D = data()
    groups = np.array([r["group"] for r in rows])
    CROSSFIT.mkdir(exist_ok=True)
    assign = {}
    for f, (tr, te) in enumerate(GroupKFold(5).split(D, groups=groups)):
        m = HistGradientBoostingClassifier(**TREES).fit(np.concatenate([D[tr], -D[tr]]), np.r_[np.ones(len(tr)), np.zeros(len(tr))])
        export(m, CROSSFIT / f"fold{f}.npz", names=keys)
        assign.update({groups[i]: f for i in te})
    (CROSSFIT / "assign.json").write_text(json.dumps(assign))
    print(f"5 fold models, {len(assign)} songs assigned -> {CROSSFIT}")


if __name__ == "__main__":
    {"rows": lambda: cmd_rows(int(sys.argv[2]) if len(sys.argv) > 2 else 12), "cv": cmd_cv, "final": cmd_final, "crossfit": cmd_crossfit}[sys.argv[1]]()
