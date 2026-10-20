"""Learn which of two half-beat-shifted grids is the beat.

    uv run python bench/phase_exp.py

At the ranked tempo, the true grid and the grid shifted by half a beat are the
two candidates; features are octave_features (band-group mass on beats, halves,
quarters, odd beats). The current rule (kick/bass mass on the beat) is the
baseline. Folds are grouped by audio hash.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_onsets import load_onsets, path_for  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS, broadband_features, octave_features, weights  # noqa: E402

C_REG = float(os.environ.get("PHASE_C", "0.1"))


def rule_prefers_first(o, phi, L, lo, hi):
    """Current beat_phase criterion restricted to the two half-shifted candidates."""
    sel = (o.times >= lo) & (o.times < hi)
    t, wl, w = o.times[sel], (o.strength * (o.sub + 0.5 * o.low))[sel], weights(o)[sel]
    def score(p):
        d = np.abs((t - p + L / 2) % L - L / 2)
        hit = np.exp(-0.5 * (d / 6) ** 2)
        return (wl * hit).sum() + 0.3 * (w * hit).sum()
    return score(phi) >= score(phi + L / 2)


def build():
    rows = []
    for it in ranked_items():
        if not path_for(it["id"]).exists():
            continue
        o = load_onsets(it["id"])
        truth = sorted(it["truth"], key=lambda s: s.time)
        ends = [truth[k + 1].time if k + 1 < len(truth) else it["span"][1] for k in range(len(truth))]
        k = int(np.argmax([e - s.time for s, e in zip(truth, ends)]))
        seg = truth[k]
        lo, hi = max(seg.time, it["span"][0]) if k else it["span"][0], ends[k]
        if (o.times >= lo).sum() < 50 or hi - lo < 15000:
            continue
        phi = seg.time - OSU_OFFSET_MS  # ranked grid in the detector's clock
        L = seg.beat_length
        cands = [octave_features(o, phi, L, lo, hi) | broadband_features(o, phi, L, lo, hi),
                 octave_features(o, phi + L / 2, L, lo, hi) | broadband_features(o, phi + L / 2, L, lo, hi)]
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        rows.append({"id": it["id"], "group": md5, "cands": cands, "rule_ok": bool(rule_prefers_first(o, phi, L, lo, hi)),
                     "kind": it["tags"]["kind"]})
    return rows


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rows = build()
    keys = sorted(k for k in rows[0]["cands"][0] if not k.startswith("log2"))
    # the two candidates are mirror images, so learn on the difference (true - shifted), both signs
    D = np.array([[r["cands"][0][k] - r["cands"][1][k] for k in keys] for r in rows])
    X = np.concatenate([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    groups = np.array([r["group"] for r in rows] * 2)
    ok = np.zeros(len(rows), bool)
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = LogisticRegression(C=C_REG, fit_intercept=False, max_iter=5000).fit(X[tr], y[tr])
        te = te[te < len(rows)]
        ok[te] = clf.decision_function(D[te]) > 0
    rule = np.array([r["rule_ok"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    print(f"{len(rows)} maps; picking the right half-beat phase:")
    for name, m in (("all", kinds != ""), ("single", kinds == "single"), ("few", kinds == "few"), ("many", kinds == "many")):
        print(f"  {name:7} n={m.sum():4}  kick rule {rule[m].mean():.1%}   learned (CV) {ok[m].mean():.1%}")
    clf = LogisticRegression(C=C_REG, fit_intercept=False, max_iter=5000).fit(X, y)
    model = {"features": keys, "coef": clf.coef_[0].tolist()}
    (DATA / "phase_model.json").write_text(json.dumps(model, indent=1))
    for f, c in sorted(zip(keys, model["coef"]), key=lambda x: -abs(x[1]))[:8]:
        print(f"    {f:14} {c:+.2f}")
    wrong = [r["id"] for r, g in zip(rows, ok) if not g]
    print(f"  learned still wrong on {len(wrong)}: {wrong[:8]}")


if __name__ == "__main__":
    main()
