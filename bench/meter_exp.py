"""Bars of 3 or of 4? Learn the meter from ranked maps' red lines.

    uv run python bench/meter_exp.py rows [workers]   # features per song -> data/meter_rows.jsonl
    uv run python bench/meter_exp.py cv               # trees out of fold: how many 3/4 songs at what false alarm
    uv run python bench/meter_exp.py phase            # which bar feature finds "1" of a bar of 3
    uv run python bench/meter_exp.py export           # trees on every song -> models/meter_trees.npz
    uv run python bench/meter_exp.py crossfit [dir]   # 5 fold models for the bench (METER_CROSSFIT)

The detector puts every song in 4/4: 7.5% of ranked maps are mostly 3/4 or 6/4 (bars of
6 beats = two bars of 3 at the bar level), and their barlines are wrong in game whatever
the grid. Label: the meter that holds most of the mapped time (>= 80%), 3 and 6 -> 3, 4 -> 4,
other songs left out. Features on the mapper's own grid (its longest section of that meter):
how strongly per-beat signals (bass and chord changes, onsets per band, loudness) repeat
every 3 / 6 / 12 beats against every 4 / 8 / 16. Grouped CV by audio hash.
"""

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import exact_onsets  # noqa: E402
from run import DATA, DATASETS  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.detect import OSU_OFFSET_MS  # noqa: E402
from autotiming.downbeat import harmony  # noqa: E402
from autotiming.meter import PAIRS, features  # noqa: E402

OUT = DATA / "meter_rows.jsonl"
MIN_BEATS = 48
SETS = ("ranked", "heavy", "nominated")


def label(truth, span) -> tuple[int | None, object]:
    """(3, 4 or None, the longest section of that meter as (time, beat_length, end))."""
    truth = sorted(truth, key=lambda s: s.time)
    lo, hi = span
    dur, best = {}, {}
    for i, s in enumerate(truth):
        e = min(truth[i + 1].time if i + 1 < len(truth) else hi, hi)
        a = max(s.time, lo)
        if e <= a:
            continue
        m = {3: 3, 6: 3, 4: 4}.get(s.meter)
        if m is None:
            continue
        dur[m] = dur.get(m, 0) + e - a
        if e - a > best.get(m, (0,))[0]:
            best[m] = (e - a, s.time, s.beat_length, e)
    if not dur:
        return None, None
    m = max(dur, key=dur.get)
    if dur[m] < 0.8 * (hi - lo):
        return None, None
    return m, best[m][1:]


def one(it):
    try:
        m, sec = label(it["truth"], it["span"])
        if m is None:
            return None
        t, L, end = sec
        lo = max(t, it["span"][0])
        n = int((end - lo) // L)
        if n < MIN_BEATS:
            return None
        pcm, sr = decode_bass(it["audio"])
        o = exact_onsets(it["audio"], pcm, sr)
        h = harmony(pcm, sr)
        first = t + np.ceil((lo - t) / L) * L
        beats = first + np.arange(n) * L - OSU_OFFSET_MS  # the mapper's beats, onset clock
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        return {"id": it["id"], "ds": it["ds"], "group": md5, "meter": m, "beats": n, "bpm": 60000 / L,
                "f": features(o, h, beats)}
    except Exception as e:
        return {"id": it["id"], "ds": it.get("ds"), "error": repr(e)}


def rows(workers: int):
    items, seen = [], set()
    for ds in SETS:
        for it in DATASETS[ds]():
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            it["ds"] = ds
            items.append(it)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.open()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(todo)} to do")
    with ProcessPoolExecutor(workers) as ex, OUT.open("a") as f:
        for r in ex.map(one, todo, chunksize=2):
            if r is not None:
                f.write(json.dumps(r) + "\n")
                f.flush()


def load():
    rs = [json.loads(l) for l in OUT.open()]
    rs = [r for r in rs if "error" not in r]
    for r in rs:  # rows from before features() added the pairs: add them the same way
        f = r["f"]
        names = sorted({k.rsplit("_", 1)[0] for k in f if "-" not in k})
        for nm in names:
            for kind in ("c", "r"):
                for a, b in PAIRS:
                    f.setdefault(f"{nm}_{kind}{a}-{b}", f[f"{nm}_{kind}{a}"] - f[f"{nm}_{kind}{b}"])
    keys = sorted(rs[0]["f"])
    X = np.array([[r["f"][k] for k in keys] for r in rs])
    y = np.array([r["meter"] == 3 for r in rs], int)
    return rs, X, y, keys


TREES = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, class_weight={0: 1, 1: 3})


def fit_export(X, y, names, path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from octave_trees_exp import export

    export(HistGradientBoostingClassifier(**TREES).fit(X, y), path, names=names)


def cmd_export():
    from autotiming.meter import METER_TREES

    rs, X, y, names = load()
    fit_export(X, y, names, METER_TREES)
    print(f"wrote {METER_TREES}")


def cmd_crossfit(dest):
    """5 folds by audio hash for the bench (baselines METER_CROSSFIT): no song judged by trees that saw it."""
    from sklearn.model_selection import StratifiedGroupKFold

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rs, X, y, names = load()
    groups = [r["group"] for r in rs]
    fa = {}
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups)):
        fit_export(X[tr], y[tr], names, dest / f"fold{f}.npz")
        fa.update({groups[i]: f for i in te})
    (dest / "assign.json").write_text(json.dumps(fa))
    print(f"wrote {dest}: 5 folds, {len(fa)} songs")


def cv():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    rs, X, y, names = load()
    groups = [r["group"] for r in rs]
    print(f"{len(rs)} songs: {y.sum()} in 3, {len(y) - y.sum()} in 4  "
          f"(by set: " + ", ".join(f"{d} {sum(r['ds'] == d for r in rs)}/{sum(r['ds'] == d and r['meter'] == 3 for r in rs)}" for d in SETS) + ")")
    P = np.zeros(len(y))
    for seed in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            m = HistGradientBoostingClassifier(**TREES)
            m.fit(X[tr], y[tr])
            P[te] += m.predict_proba(X[te])[:, 1] / 3
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pick = P >= thr
        print(f"  p >= {thr:.2f}: called 3 on {pick.sum():4}, right {y[pick].sum():4}, false on 4/4 {(pick & (y == 0)).sum():3}"
              f" ({(pick & (y == 0)).sum() / max(1, (y == 0).sum()):.2%} of 4/4)   3/4 songs found {y[pick].sum() / max(1, y.sum()):.0%}")
    # single features: which ones separate at all
    from sklearn.metrics import roc_auc_score

    auc = sorted(((roc_auc_score(y, X[:, j]), n) for j, n in enumerate(names)), key=lambda a: -abs(a[0] - 0.5))
    print("  best single features (AUC):", ", ".join(f"{n} {a:.2f}" for a, n in auc[:10]))
    np.save(DATA / "meter_oof.npy", P)


PHASE_OUT = DATA / "meter_phase_rows.jsonl"


def phase_one(it):
    """bar_features for bars of 3 starting on each of the 3 beats from the mapper's "1" (candidate 0)."""
    from autotiming.downbeat import bar_features

    try:
        m, (t, L, end) = label(it["truth"], it["span"])
        lo = max(t, it["span"][0])
        # the window must not start on the mapper's line: features would learn "the window starts on 1"
        lo += (int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 3 + 0.5) * L
        pcm, sr = decode_bass(it["audio"])
        o = exact_onsets(it["audio"], pcm, sr)
        h = harmony(pcm, sr)
        phi = t - OSU_OFFSET_MS
        feats = [bar_features(o, phi + j * L, L, lo - OSU_OFFSET_MS, end - OSU_OFFSET_MS, meter=3, ch=h) for j in range(3)]
        return {"id": it["id"], "feats": feats}
    except Exception as e:
        return {"id": it["id"], "error": repr(e)}


def phase(workers: int):
    """Which bar feature puts "1" where the mapper did on the 3/4 songs."""
    threes = {r["id"] for r in map(json.loads, OUT.open()) if r.get("meter") == 3}
    items = [it for ds in SETS for it in DATASETS[ds]() if it["id"] in threes]
    items = list({it["id"]: it for it in items}.values())
    if not PHASE_OUT.exists():
        with ProcessPoolExecutor(workers) as ex, PHASE_OUT.open("w") as f:
            for r in ex.map(phase_one, items):
                f.write(json.dumps(r) + "\n")
    rs = [r for r in map(json.loads, PHASE_OUT.open()) if "error" not in r]
    keys = sorted(rs[0]["feats"][0])
    F = np.nan_to_num(np.array([[[r["feats"][j][k] for k in keys] for j in range(3)] for r in rs]))
    print(f"{len(rs)} songs in 3; '1' picked right by one feature (chance 33%):")
    def fair(S):  # ties share the credit: argmax would hand them to candidate 0, the right one
        top = S == S.max(1, keepdims=True)
        return float(np.mean(top[:, 0] / top.sum(1)))

    acc = {k: fair(F[:, :, i]) for i, k in enumerate(keys)}
    for k, a in sorted(acc.items(), key=lambda x: -x[1])[:10]:
        print(f"   {k:16} {a:.0%}")
    # z-scored sum of the best few, each candidate against the song's three
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)
    for n in (2, 3, 4, 6):
        top = [keys.index(k) for k, _ in sorted(acc.items(), key=lambda x: -x[1])[:n]]
        print(f"   sum of top {n} ({', '.join(keys[i] for i in top)}): {fair(Z[:, :, top].sum(2)):.0%}")
    for combo in (["ch_change2", "bass_change2"], ["ch_change2", "bass_change2", "bass_phrase"],
                  ["ch_change2", "chroma_change", "bass_change2", "bass_phrase", "strength_on_one"]):
        print(f"   {' + '.join(combo)}: {fair(Z[:, :, [keys.index(k) for k in combo]].sum(2)):.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    elif sys.argv[1] == "export":
        cmd_export()
    elif sys.argv[1] == "crossfit":
        cmd_crossfit(sys.argv[2] if len(sys.argv) > 2 else DATA / "meter_crossfit")
    elif sys.argv[1] == "phase":
        phase(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    else:
        cv()
