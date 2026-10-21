"""Does the beat tracker's phase help where the phase trees get the "and"?

    uv run python bench/phase_tracker_exp.py rows --workers 12   # tracker vote per ranked map
    uv run python bench/phase_tracker_exp.py eval                 # against cross-validated phase trees
    uv run python bench/phase_tracker_exp.py trees                # the vote as a phase-tree feature

The tracker runs on beat probabilities (cross-fitted, so no map sees a model
trained on it) and picks its own phase; the steady grid's phase comes from the
phase trees. For each map: on the mapper's main section, the share of tracked
beats on the mapper's beat minus the share on its "and" (+1: tracker agrees
with the mapper, -1: it sits on the "and").
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming import detect as D  # noqa: E402
from autotiming import track as T  # noqa: E402

OUT = DATA / "phase_tracker_rows.json"
CROSSFIT = DATA / "beatprob" / "crossfit.npz"


def vote(beats, phi, L, lo, hi, tol=0.15):
    """(on beat - on "and") / beats in [lo, hi), positions within tol beats."""
    b = beats[(beats >= lo) & (beats < hi)]
    if len(b) < 8:
        return None
    pos = ((b - phi) / L) % 1.0
    on = np.abs((pos + 0.5) % 1.0 - 0.5) < tol
    off = np.abs(pos - 0.5) < tol
    return float((on.sum() - off.sum()) / len(b))


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
        z = np.load(CROSSFIT)
        key = it["id"].replace("/", "__")
        if key not in z.files:
            return None
        steady = D.detect_onsets(o)
        main_L = D.main_beat_length(o, steady)
        # cross-fitted inside the map's span (all that is scored), the final model outside it
        from autotiming.beatprob import beat_prob
        probs = beat_prob(o, main_L)
        probs[np.load(DATA / "beatprob" / (key + ".npz"))["idx"]] = z[key]
        secs = D.align_octave(T.fine_sections(o, main_L), steady)
        tr = T.track_beats(o, secs, probs=probs)
        # onset clock, like tracked beats
        phi, L = seg.time - D.OSU_OFFSET_MS, seg.beat_length
        lo, hi = lo - D.OSU_OFFSET_MS, hi - D.OSU_OFFSET_MS
        # tracker at the mapper's octave only: on x2 every "and" is a beat too
        iv = np.median(np.diff(tr.times[(tr.times >= lo) & (tr.times < hi)])) if len(tr.times) > 8 else np.nan
        return {"id": it["id"], "vote": vote(tr.times, phi, L, lo, hi), "ratio": float(iv / L)}
    except Exception as e:
        return {"id": it["id"], "error": repr(e)}


def cmd_rows(workers):
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, list(ranked_items()), chunksize=2))
    errs = [r for r in rows if r and "error" in r]
    rows = [r for r in rows if r and "error" not in r]
    OUT.write_text(json.dumps(rows))
    print(f"{len(rows)} rows, {len(errs)} errors{': ' + errs[0]['error'] if errs else ''}")


def cmd_eval():
    import phase_trees_exp as P
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows, keys, X = P.data()
    groups = np.array([r["group"] for r in rows])
    p = np.zeros(len(rows))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        m = HistGradientBoostingClassifier(**P.TREES).fit(np.concatenate([X[tr], -X[tr]]),
                                                         np.r_[np.ones(len(tr)), np.zeros(len(tr))])
        p[te] = (m.predict_proba(X[te])[:, 1] + 1 - m.predict_proba(-X[te])[:, 1]) / 2
    tv = {r["id"]: r for r in json.loads(OUT.read_text())}
    ids = [r["id"] for r in rows]
    have = np.array([i in tv and tv[i]["vote"] is not None and abs(tv[i]["ratio"] - 1) < 0.1 for i in ids])
    v = np.array([tv[i]["vote"] if h else 0.0 for i, h in zip(ids, have)])
    print(f"{len(rows)} maps, tracker at the mapper's octave on {have.sum()}")
    print(f"  trees wrong {np.sum(p < 0.5)}; tracker wrong {np.sum(have & (v < 0))}, both wrong {np.sum(have & (v < 0) & (p < 0.5))}")
    for conf in (0.75, 0.85, 0.9, 1.0):
        for vt in (0.3, 0.5, 0.7):
            unsure = np.maximum(p, 1 - p) < conf
            # flip the trees' choice where they are unsure and the tracker clearly sits on the other phase
            fix = have & unsure & (p < 0.5) & (v > vt)
            brk = have & unsure & (p > 0.5) & (v < -vt)
            print(f"  trees conf < {conf:.2f}, tracker vote > {vt}: fixes {fix.sum():2}, breaks {brk.sum():2}")



def cmd_trees():
    """Phase trees with the tracker's vote as one more feature, grouped 5-fold CV."""
    import phase_trees_exp as P
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    rows, keys, X = P.data()
    tv = {r["id"]: r for r in json.loads(OUT.read_text())}
    # no vote (tracker at another octave, or too few beats) is 0: no opinion
    v = np.array([tv[r["id"]]["vote"] if r["id"] in tv and tv[r["id"]]["vote"] is not None
                  and abs(tv[r["id"]]["ratio"] - 1) < 0.1 else 0.0 for r in rows])
    groups = np.array([r["group"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    for name, F in (("trees", X), ("trees + tracker vote", np.c_[X, v])):
        p = np.zeros(len(rows))
        for tr, te in GroupKFold(5).split(F, groups=groups):
            m = HistGradientBoostingClassifier(**P.TREES).fit(np.concatenate([F[tr], -F[tr]]),
                                                             np.r_[np.ones(len(tr)), np.zeros(len(tr))])
            p[te] = (m.predict_proba(F[te])[:, 1] + 1 - m.predict_proba(-F[te])[:, 1]) / 2
        wrong = p < 0.5
        c = np.maximum(p, 1 - p)
        parts = "  ".join(f"{g} {np.mean(wrong[kinds == g]):.1%}" for g in ("single", "few", "many"))
        print(f"  {name:22} wrong {wrong.sum():2} ({wrong.mean():.1%})  {parts}   "
              f"flag <0.85: {np.mean(c < 0.85):.1%} of maps, silent errors {np.sum(wrong & (c >= 0.85))}")


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        cmd_rows(int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4)
    elif sys.argv[1] == "trees":
        cmd_trees()
    else:
        cmd_eval()
