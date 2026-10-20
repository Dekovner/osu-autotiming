"""Where is "1" in the bar? Learn the bar phase from ranked maps' red lines.

    uv run python bench/downbeat_exp.py [workers]

Mappers put red lines on downbeats, so each ranked map's main 4/4 red line
gives the bar phase. Candidates are the 4 beat phases of the main segment; each
gets features (kick / bass / harmonic change / loudness / onset strength on
those beats); a logistic model picks one. Grouped 5-fold CV by audio hash.
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
from autotiming.downbeat import bar_features, chroma  # noqa: E402
from autotiming.detect import OSU_OFFSET_MS  # noqa: E402

OUT = DATA / "downbeat_rows.json"


def one(it):
    try:
        if not path_for(it["id"]).exists():
            return None
        truth = sorted(it["truth"], key=lambda s: s.time)
        ends = [s.time for s in truth[1:]] + [it["span"][1]]
        k = int(np.argmax([e - s.time for s, e in zip(truth, ends)]))
        seg = truth[k]
        if seg.meter != 4 or ends[k] - seg.time < 20 * seg.beat_length:
            return None
        o = load_onsets(it["id"])
        pcm, sr = decode_bass(it["audio"])
        ch = chroma(pcm, sr)
        lo, hi = max(seg.time, it["span"][0]), ends[k]
        # start the window off the mapper's line (a downbeat by definition): window-based features
        # must not learn "the window starts on 1", which the detector can't know
        lo += (int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 4 + 0.5) * seg.beat_length
        # four candidate bar phases in the onset clock; candidate 0 is the mapper's downbeat
        phi = seg.time - OSU_OFFSET_MS
        feats = [bar_features(o, phi + j * seg.beat_length, seg.beat_length, lo - OSU_OFFSET_MS, hi - OSU_OFFSET_MS, ch=ch)
                 for j in range(4)]
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        return {"id": it["id"], "group": md5, "feats": feats}
    except Exception as e:
        return {"id": it["id"], "error": repr(e)}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(one, list(ranked_items()), chunksize=4) if r and "error" not in r]
    OUT.write_text(json.dumps(rows))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    keys = sorted(rows[0]["feats"][0])
    F = np.nan_to_num(np.array([[[r["feats"][j][k] for k in keys] for j in range(4)] for r in rows]))  # maps x 4 x feats
    # each candidate relative to the mean of the four: what makes this beat stand out as "1"
    D = F - F.mean(axis=1, keepdims=True)
    X = D.reshape(-1, len(keys))
    y = np.tile([1, 0, 0, 0], len(rows))
    groups = np.repeat([r["group"] for r in rows], 4)
    ok = np.zeros(len(rows), bool)
    for tr, te in GroupKFold(5).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=0.3, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        maps = np.unique(te // 4)
        s = clf.decision_function((D[maps].reshape(-1, len(keys)) - mu) / sd).reshape(len(maps), 4)
        ok[maps] = s.argmax(1) == 0
    base = {k: np.mean(F[:, :, keys.index(k)].argmax(1) == 0) for k in keys}
    print(f"{len(rows)} maps; downbeat picked right (CV): {ok.mean():.1%}  (chance 25%)")
    for k, v in sorted(base.items(), key=lambda x: -x[1])[:6]:
        print(f"   single feature '{k}': {v:.1%}")
    Xa = X
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    clf = LogisticRegression(C=0.3, max_iter=3000).fit((Xa - mu) / sd, y)
    model = {"features": keys, "coef": (clf.coef_[0] / sd).tolist()}
    (DATA / "downbeat_model.json").write_text(json.dumps(model, indent=1))


def final():
    """Train the trees on every map and export them like the beat-probability model."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from autotiming.beatprob import load_model, predict_proba
    from autotiming.downbeat import FEATURE_KEYS, MODEL_PATH, candidate_matrix

    rows = json.loads(OUT.read_text())
    assert sorted(rows[0]["feats"][0]) == sorted(FEATURE_KEYS)
    # candidates in random order: the true one must not always sit first
    shift = np.random.default_rng(0).integers(0, 4, len(rows))
    X = np.concatenate([np.roll(candidate_matrix(r["feats"]), s, axis=0) for r, s in zip(rows, shift)])
    y = np.zeros((len(rows), 4), int)
    y[np.arange(len(rows)), shift] = 1
    y = y.ravel()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(X, y)
    trees = [p[0].nodes for p in m._predictors]
    cat = np.concatenate(trees)
    np.savez_compressed(
        MODEL_PATH, n_nodes=np.array([len(t) for t in trees]), feature=cat["feature_idx"].astype(np.int16),
        threshold=cat["num_threshold"], left=cat["left"].astype(np.int32), right=cat["right"].astype(np.int32),
        leaf=cat["is_leaf"].astype(bool), value=cat["value"], missing_left=cat["missing_go_to_left"].astype(bool),
        baseline=np.atleast_1d(m._baseline_prediction).ravel()[:1], names=np.array(FEATURE_KEYS),
    )
    load_model.cache_clear()
    diff = np.abs(predict_proba(load_model(MODEL_PATH), X) - m.predict_proba(X)[:, 1]).max()
    print(f"{len(trees)} trees -> {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KB); max |p - sklearn| = {diff:.2e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        final()
    else:
        main()
