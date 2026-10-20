"""Train "is this onset a beat?" on ranked timing, with grouped cross-fitting.

    uv run python bench/beatprob_exp.py data [workers]   # features + labels per map -> data/beatprob/
    uv run python bench/beatprob_exp.py fit              # grouped 5-fold CV, cross-fitted probs, final model

Label: the onset (in osu! time) is within BEAT_TOL ms of a ranked beat. Every
map's probabilities used for evaluation come from a model that never saw that
song (folds grouped by audio hash).
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
from run import DATA, RESULTS, ranked_items  # noqa: E402

from autotiming.beatprob import onset_features  # noqa: E402
from autotiming.detect import OSU_OFFSET_MS  # noqa: E402
from autotiming.grid import beats as grid_beats  # noqa: E402

OUT = DATA / "beatprob"
BEAT_TOL = 12.0


def main_L(pred, t_end):
    segs = sorted(pred, key=lambda p: p[0])
    ends = [s[0] for s in segs[1:]] + [t_end]
    return segs[int(np.argmax([e - s[0] for s, e in zip(segs, ends)]))][1]


def build_one(args):
    it, pred = args
    o = load_onsets(it["id"])
    L = main_L(pred, o.times[-1] + OSU_OFFSET_MS)
    X, names = onset_features(o, L)
    tb = grid_beats(it["truth"], it["span"][0] - 2000, it["span"][1] + 2000)
    t = o.times + OSU_OFFSET_MS
    j = np.clip(np.searchsorted(tb, t), 1, len(tb) - 1)
    d = np.minimum(np.abs(t - tb[j - 1]), np.abs(t - tb[j]))
    inside = (t >= it["span"][0] - 500) & (t <= it["span"][1] + 500)
    y = (d < BEAT_TOL).astype(np.int8)
    np.savez_compressed(OUT / (it["id"].replace("/", "__") + ".npz"), X=X[inside], y=y[inside], idx=np.flatnonzero(inside))
    return names


def cmd_data(workers):
    OUT.mkdir(exist_ok=True)
    cur = {r["id"]: r for r in map(json.loads, (RESULTS / "ranked_ours_steady.jsonl").open())}
    todo = [(it, cur[it["id"]]["pred"]) for it in ranked_items()
            if path_for(it["id"]).exists() and cur.get(it["id"], {}).get("pred")]
    with ProcessPoolExecutor(workers) as ex:
        names = list(ex.map(build_one, todo, chunksize=4))[0]
    (OUT / "names.json").write_text(json.dumps(names))
    groups = {it["id"]: hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest() for it, _ in todo}
    (OUT / "groups.json").write_text(json.dumps(groups))
    print(f"{len(todo)} maps")


def cmd_fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    groups = json.loads((OUT / "groups.json").read_text())
    ids = sorted(groups)
    data = {i: np.load(OUT / (i.replace("/", "__") + ".npz")) for i in ids}
    gid = np.array([groups[i] for i in ids])
    rng = np.random.default_rng(0)
    folds = list(GroupKFold(5).split(ids, groups=gid))
    probs = {}
    aucs = {"strength only": [], "logistic": [], "boosted trees": []}
    for k, (tr, te) in enumerate(folds):
        Xs, ys = [], []
        for i in np.array(ids)[tr]:
            X, y = data[i]["X"], data[i]["y"]
            sel = rng.choice(len(y), min(len(y), 1500), replace=False)  # cap per map: long songs don't dominate
            Xs.append(X[sel]); ys.append(y[sel])
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0)
        hgb.fit(Xtr, ytr)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        lr = LogisticRegression(C=0.1, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        Xte = np.concatenate([data[i]["X"] for i in np.array(ids)[te]])
        yte = np.concatenate([data[i]["y"] for i in np.array(ids)[te]])
        names = json.loads((OUT / "names.json").read_text())
        aucs["strength only"].append(roc_auc_score(yte, Xte[:, names.index("log_strength")]))
        aucs["logistic"].append(roc_auc_score(yte, lr.predict_proba((Xte - mu) / sd)[:, 1]))
        p = hgb.predict_proba(Xte)[:, 1]
        aucs["boosted trees"].append(roc_auc_score(yte, p))
        pos = 0
        for i in np.array(ids)[te]:  # cross-fitted probabilities for honest downstream evaluation
            n = len(data[i]["y"])
            probs[i] = p[pos:pos + n]
            pos += n
        print(f"fold {k}: AUC strength {aucs['strength only'][-1]:.3f}  logistic {aucs['logistic'][-1]:.3f}  trees {aucs['boosted trees'][-1]:.3f}", flush=True)
    for k, v in aucs.items():
        print(f"  {k:14} AUC {np.mean(v):.3f}")
    np.savez_compressed(OUT / "crossfit.npz", **{i.replace("/", "__"): probs[i] for i in ids})


MODEL_OUT = Path(__file__).resolve().parents[1] / "src" / "autotiming" / "models" / "beatprob_trees.npz"


def cmd_final():
    """Train on every map and export the trees as plain arrays (no sklearn needed to use them)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    groups = json.loads((OUT / "groups.json").read_text())
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for i in sorted(groups):
        z = np.load(OUT / (i.replace("/", "__") + ".npz"))
        sel = rng.choice(len(z["y"]), min(len(z["y"]), 1500), replace=False)
        Xs.append(z["X"][sel]); ys.append(z["y"][sel])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0).fit(X, y)
    trees = [p[0].nodes for p in hgb._predictors]
    n_nodes = np.array([len(t) for t in trees])
    cat = np.concatenate(trees)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        MODEL_OUT, n_nodes=n_nodes, feature=cat["feature_idx"].astype(np.int16), threshold=cat["num_threshold"],
        left=cat["left"].astype(np.int32), right=cat["right"].astype(np.int32), leaf=cat["is_leaf"].astype(bool),
        value=cat["value"], missing_left=cat["missing_go_to_left"].astype(bool),
        baseline=np.atleast_1d(hgb._baseline_prediction).ravel()[:1], names=np.array(json.loads((OUT / "names.json").read_text())),
    )
    # the exported evaluator must agree with sklearn
    from autotiming.beatprob import load_model, predict_proba

    sample = X[rng.choice(len(X), 20000, replace=False)]
    diff = np.abs(predict_proba(load_model(MODEL_OUT), sample) - hgb.predict_proba(sample)[:, 1]).max()
    print(f"{len(trees)} trees, {cat.size} nodes -> {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB); max |p - sklearn| = {diff:.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "data":
        cmd_data(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    elif sys.argv[1] == "final":
        cmd_final()
    else:
        cmd_fit()
