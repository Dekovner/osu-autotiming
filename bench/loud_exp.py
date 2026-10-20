"""Does loudness on the beat vs the "and" fix the half-beat phase?

    uv run python bench/loud_exp.py [workers]

Adds loudness features (mean power just after grid positions, broadband / low
/ high) to the phase_exp features and compares grouped-CV accuracy.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_exp  # noqa: E402
from run import DATA, ranked_items  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.detect import OSU_OFFSET_MS, loudness_on_beat  # noqa: E402

CACHE = DATA / "loud_features.json"


def one(args):
    item_id, audio, phi, L, lo, hi = args
    x, sr = decode_bass(audio)
    return item_id, loudness_on_beat(x, sr, phi, L, lo, hi)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rows = phase_exp.build()
    items = {it["id"]: it for it in ranked_items()}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = []
    for r in rows:
        if r["id"] in cache:
            continue
        it = items[r["id"]]
        truth = sorted(it["truth"], key=lambda s: s.time)
        ends = [truth[k + 1].time if k + 1 < len(truth) else it["span"][1] for k in range(len(truth))]
        k = int(np.argmax([e - s.time for s, e in zip(truth, ends)]))
        lo = max(truth[k].time, it["span"][0]) if k else it["span"][0]
        todo.append((r["id"], it["audio"], truth[k].time - OSU_OFFSET_MS, truth[k].beat_length, lo, ends[k]))
    print(f"{len(todo)} to compute", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, v in ex.map(one, todo, chunksize=4):
            cache[k] = v
    CACHE.write_text(json.dumps(cache))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    base_keys = sorted(k for k in rows[0]["cands"][0] if not k.startswith("log2"))
    loud_keys = sorted(cache[rows[0]["id"]])
    kinds = np.array([r["kind"] for r in rows])
    groups2 = np.array([r["group"] for r in rows] * 2)
    for label, use_base, use_loud in (("spectral only", True, False), ("loudness only", False, True), ("both", True, True)):
        cols = []
        if use_base:
            cols.append(np.array([[r["cands"][0][k] - r["cands"][1][k] for k in base_keys] for r in rows]))
        if use_loud:
            # shifted candidate is the mirror: its "beat minus and" is the negative
            cols.append(np.array([[2 * cache[r["id"]][k] for k in loud_keys] for r in rows]) / 10)
        D = np.concatenate(cols, axis=1)
        X, y = np.concatenate([D, -D]), np.concatenate([np.ones(len(D)), np.zeros(len(D))])
        ok = np.zeros(len(rows), bool)
        for tr, te in GroupKFold(5).split(X, y, groups2):
            clf = LogisticRegression(C=0.1, fit_intercept=False, max_iter=5000).fit(X[tr], y[tr])
            te = te[te < len(rows)]
            ok[te] = clf.decision_function(D[te]) > 0
        print(f"  {label:14} all {ok.mean():.1%}  single {ok[kinds == 'single'].mean():.1%}  "
              f"few {ok[kinds == 'few'].mean():.1%}  many {ok[kinds == 'many'].mean():.1%}")
    rule = np.array([r["rule_ok"] for r in rows])
    print(f"  {'kick rule':14} all {rule.mean():.1%}  single {rule[kinds == 'single'].mean():.1%}")
    # final "both" model on everything, as the detector's second opinion on phase
    clf = LogisticRegression(C=0.1, fit_intercept=False, max_iter=5000).fit(X, y)
    coef = clf.coef_[0]
    model = {"spectral": dict(zip(base_keys, coef[:len(base_keys)].tolist())),
             "loud": {k: float(c) * 2 / 10 for k, c in zip(loud_keys, coef[len(base_keys):])}}
    (DATA / "phase_model.json").write_text(json.dumps(model, indent=1))


if __name__ == "__main__":
    main()
