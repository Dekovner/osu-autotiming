"""Octave of the sections other than the main one: which x0.5 / x1 / x2 do mappers pick there?

    uv run python bench/octave_section_exp.py rows [workers]   # raw sections + features for every ranked map
    uv run python bench/octave_section_exp.py score            # decision rules, octave trees from the crossfit folds

choose_octave() lets the main (longest) section's octave come from the trees and every other
section of OCT_SECTION_MIN_BEATS or more decide on its own, with OCT_CONT_LAMBDA per
|log2(bpm / main bpm)| against straying; shorter ones follow the main one. Rows hold every raw
section with its candidates' features and the mapper's tempo over it (longest overlap).
"""

import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS, _detect_raw, octave_candidates, octave_features  # noqa: E402

OUT = DATA / "octave_section_rows.json"
CROSSFIT = DATA / "octave_crossfit"


def one(it):
    try:
        o = load_onsets(it["id"])
        segs = _detect_raw(o)
        ends = [segs[i + 1].time if i + 1 < len(segs) else o.times[-1] + 1 for i in range(len(segs))]
        main = max(range(len(segs)), key=lambda i: ends[i] - segs[i].time)
        truth = sorted(it["truth"], key=lambda s: s.time)
        tt = [s.time - OSU_OFFSET_MS for s in truth]  # onset clock, like the raw lines
        secs = []
        for i, s in enumerate(segs):
            lo = s.time if i > 0 else o.times[0]
            hi = ends[i]
            best, best_ov = None, 0.0
            for k, m in enumerate(truth):
                e = tt[k + 1] if k + 1 < len(truth) else 1e12
                ov = min(e, hi) - max(tt[k] if k else -1e12, lo)
                if ov > best_ov:
                    best, best_ov = m, ov
            cands = octave_candidates(o, s.time, s.beat_length, lo, hi)
            label = None
            if best is not None:
                label = next((k for k in cands if abs(np.log((s.beat_length / k) / best.beat_length)) < 0.02), None)
            feats = {str(k): {f: float(v) for f, v in octave_features(o, p, L, lo, hi).items()} for k, (p, L) in cands.items()}
            secs.append({"bpm": s.bpm, "lo": float(lo), "hi": float(hi), "beats": float((hi - lo) / s.beat_length),
                         "label": label, "truth_bpm": None if best is None else best.bpm,
                         "cover": float(best_ov / (hi - lo)), "feats": feats})
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        return {"id": it["id"], "group": md5, "main": main, "secs": secs}
    except Exception as e:  # noqa: BLE001
        return {"id": it["id"], "error": repr(e)}


def cmd_rows(workers):
    items = [it for it in ranked_items() if path_for(it["id"]).exists()]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items, chunksize=1))
    OUT.write_text(json.dumps(rows))
    print(len(rows), "maps,", sum("error" in r for r in rows), "errors")


def scorers():
    """md5 -> function(feats) -> {k: log-odds}, from the fold model that never saw that song."""
    from autotiming.beatprob import load_model, predict_proba
    from autotiming.detect import OCTAVE_KS, octave_matrix

    cf = Path(os.environ.get("OCTAVE_CROSSFIT_DIR") or CROSSFIT)
    assign = json.loads((cf / "assign.json").read_text())
    models = {f: load_model(cf / f"fold{f}.npz") for f in set(assign.values())}

    def score(md5, feats):
        m = models[assign[md5]]
        F = {float(k): v for k, v in feats.items()}
        p = np.clip(predict_proba(m, octave_matrix(F)), 1e-6, 1 - 1e-6)
        return {k: float(np.log(p[i] / (1 - p[i]))) for i, k in enumerate(OCTAVE_KS)}

    return assign, score


def cmd_eval():
    rows = [r for r in json.loads(OUT.read_text()) if "error" not in r]
    assign, score = scorers()
    rows = [r for r in rows if r["group"] in assign]
    for r in rows:
        for s in r["secs"]:
            s["sc"] = score(r["group"], s["feats"])
    # main section accuracy (sanity: crossfit trees ~89%)
    mains = [r["secs"][r["main"]] for r in rows if r["secs"][r["main"]]["label"] is not None]
    pick = lambda sc: max(sc, key=sc.get)
    print(f"main sections: {len(mains)}, trees right {np.mean([pick(s['sc']) == s['label'] for s in mains]):.1%}")
    # other sections: those the mapper has at x0.5/x1/x2 of ours, mostly covered by one mapper tempo
    others = []
    for r in rows:
        m = r["secs"][r["main"]]
        f_main = pick(m["sc"])
        for i, s in enumerate(r["secs"]):
            if i == r["main"] or s["label"] is None or s["cover"] < 0.8:
                continue
            others.append((r, i, s, m["bpm"] * f_main, m["label"]))
    print(f"other sections: {len(others)} (>=32 beats: {sum(s['beats'] >= 32 for _, _, s, _, _ in others)})")
    print("labels:", Counter(s["label"] for _, _, s, _, _ in others))

    def rule(lam, min_beats=32, main_truth=False):
        right, wrong = 0.0, []
        for r, i, s, main_bpm, main_label in others:
            if main_truth:
                main_bpm = r["secs"][r["main"]]["bpm"] * (main_label or 1.0)
            if s["beats"] >= min_beats:
                f = max(s["sc"], key=lambda k: s["sc"][k] - lam * abs(np.log2(s["bpm"] * k / main_bpm)))
            else:  # follow the main octave: nearest to it in log tempo
                f = min(s["sc"], key=lambda k: abs(np.log2(s["bpm"] * k / main_bpm)))
            dur = s["hi"] - s["lo"]
            if f == s["label"]:
                right += dur
            else:
                wrong.append((r["id"], i, round(s["bpm"] * f), round(s["truth_bpm"]), dur))
        tot = sum(s["hi"] - s["lo"] for _, _, s, _, _ in others)
        return right / tot, wrong

    for mt in (False, True):
        print("main octave from", "mapper" if mt else "trees")
        for mb in (0, 16, 32, 64):
            print(f"  min_beats {mb:3d}:", "  ".join(f"lam {lam}: {rule(lam, mb, mt)[0]:.1%}" for lam in (0, 0.5, 1, 2, 4, 1e9)))
    acc, wrong = rule(1.0)
    print("current rule (lam 1, 32 beats), time right:", f"{acc:.1%}", "wrong sections:", len(wrong))
    for w in sorted(wrong, key=lambda x: -x[4])[:40]:
        print("  ", w[0], "sec", w[1], "ours", w[2], "mapper", w[3], f"{w[4] / 1000:.0f}s")


def cmd_cv():
    """Octave trees trained on main sections only (as now) vs main + other sections, grouped by song
    with the crossfit folds; accuracy on main sections and, separately, on the others."""
    from octave_trees_exp import TREES
    from sklearn.ensemble import HistGradientBoostingClassifier

    from autotiming.detect import OCTAVE_KS, octave_matrix

    rows = [r for r in json.loads(OUT.read_text()) if "error" not in r]
    assign = json.loads((CROSSFIT / "assign.json").read_text())
    data = []  # (fold, is_main, X, y, duration)
    for r in rows:
        if r["group"] not in assign:
            continue
        for i, s in enumerate(r["secs"]):
            if s["label"] is None or (i != r["main"] and (s["cover"] < 0.8 or s["beats"] < 16)):
                continue
            X = octave_matrix({k: s["feats"][str(k)] for k in OCTAVE_KS})
            y = np.array([s["label"] == k for k in OCTAVE_KS], int)
            data.append((assign[r["group"]], i == r["main"], X, y, s["hi"] - s["lo"]))
    fold = np.array([d[0] for d in data])
    is_main = np.array([d[1] for d in data])
    X = np.stack([d[2] for d in data])
    y = np.stack([d[3] for d in data])
    print(f"{is_main.sum()} main sections, {(~is_main).sum()} others")
    for name, use in (("main only", is_main), ("main + others", np.ones(len(data), bool))):
        for w_other in ((1.0,) if name == "main only" else (0.5, 1.0, 2.0)):
            ok = np.zeros(len(data), bool)
            for f in sorted(set(fold)):
                tr, te = use & (fold != f), fold == f
                wt = np.where(is_main[tr], 1.0, w_other).repeat(3)
                m = HistGradientBoostingClassifier(**TREES).fit(X[tr].reshape(-1, X.shape[2]), y[tr].ravel(), sample_weight=wt)
                p = m.predict_proba(X[te].reshape(-1, X.shape[2]))[:, 1].reshape(-1, 3)
                ok[te] = y[te][np.arange(te.sum()), p.argmax(1)] == 1
            print(f"  {name:14s} w_other {w_other}: main {ok[is_main].mean():.1%}   others {ok[~is_main].mean():.1%}")


def section_data(rows, w_other=1.0):
    """X (n, 3, features), y (n, 3), weights per row and song: main sections, plus the others the
    mapper keeps at one tempo (cover >= 0.8) for 16 beats or more."""
    from autotiming.detect import OCTAVE_KS, octave_matrix

    X, y, w, g = [], [], [], []
    for r in rows:
        for i, s in enumerate(r["secs"]):
            if s["label"] is None or (i != r["main"] and (s["cover"] < 0.8 or s["beats"] < 16)):
                continue
            X.append(octave_matrix({k: s["feats"][str(k)] for k in OCTAVE_KS}))
            y.append([s["label"] == k for k in OCTAVE_KS])
            w.append(1.0 if i == r["main"] else w_other)
            g.append(r["group"])
    return np.stack(X), np.array(y, int), np.array(w), np.array(g)


def cmd_export(dest):
    """Fold models (crossfit: same song assignment as data/octave_crossfit) or the final model, trained on
    main + other sections."""
    from octave_trees_exp import TREES, export
    from sklearn.ensemble import HistGradientBoostingClassifier

    from autotiming.detect import OCTAVE_TREES

    rows = [r for r in json.loads(OUT.read_text()) if "error" not in r]
    X, y, w, g = section_data(rows)
    fit = lambda sel: HistGradientBoostingClassifier(**TREES).fit(X[sel].reshape(-1, X.shape[2]), y[sel].ravel(),
                                                                 sample_weight=w[sel].repeat(3))
    if dest.startswith("final"):  # "final" or "final:<path>"
        path = Path(dest.split(":", 1)[1]) if ":" in dest else OCTAVE_TREES
        export(fit(np.ones(len(X), bool)), path)
        print(f"{len(X)} sections -> {path}")
        return
    assign = json.loads((CROSSFIT / "assign.json").read_text())
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    fold = np.array([assign.get(x, -1) for x in g])
    for f in sorted(set(assign.values())):
        export(fit(fold != f), out / f"fold{f}.npz")
    (out / "assign.json").write_text(json.dumps(assign))
    print(f"{len(set(assign.values()))} fold models from {len(X)} sections -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rows":
        cmd_rows(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "cv":
        cmd_cv()
    elif cmd == "export":
        cmd_export(sys.argv[2])
    else:
        cmd_eval()
