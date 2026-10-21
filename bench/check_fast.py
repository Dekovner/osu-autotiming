"""Do the fast tempo searches pick what the brute-force ones did? Run on every distinct song.

    uv run python bench/check_fast.py [workers]
    CHECK_PATH=0 uv run python bench/check_fast.py   # refine_period only (the slow part to check)

refine_period (coarse pass + full grid near the best peaks) and local_tempo_path
(windows as sums of parts) against their old forms, called on the very inputs
the detector gives them. Also fills the exact onset cache the bench uses.
"""

import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, background, by_audio  # noqa: E402

from autotiming import detect as D  # noqa: E402

NEW_REFINE, NEW_PATH = D.refine_period, D.local_tempo_path
CHECK_PATH = os.environ.get("CHECK_PATH", "1") == "1"  # CHECK_PATH=0: only refine_period against its old form


def old_refine(times, weights, L0, span=0.015):
    T = max(times[-1] - times[0], 1.0)
    step = max(L0 * L0 / (40 * T), L0 * 2e-6)
    Ls = np.arange(L0 * (1 - span), L0 * (1 + span), step)
    if len(Ls) > 4000:
        Ls = np.linspace(L0 * (1 - span), L0 * (1 + span), 4000)
    best_L, best_c = L0, -1.0
    for s in range(0, len(Ls), 256):
        chunk = Ls[s:s + 256]
        c = D.coherence(times, weights, chunk)
        i = int(np.argmax(c))
        if c[i] > best_c:
            best_L, best_c = float(chunk[i]), float(c[i])
    return best_L


def old_coherence_rows(t, w, centers, Ls, win_ms):
    E = np.zeros((len(centers), len(Ls)))
    for i, c in enumerate(centers):
        sel = np.abs(t - c) < win_ms / 2
        if sel.sum() < 6:
            continue
        E[i] = D.coherence(t[sel], w[sel], Ls) / (w[sel].sum() * 1.75)
    return E


def one(items):
    stats = Counter()
    worst = {"refine_rel": 0.0, "path_rel": 0.0}
    timing = Counter()

    def refine(times, weights, L0, span=0.015):
        t0 = time.perf_counter()
        a = NEW_REFINE(times, weights, L0, span)
        t1 = time.perf_counter()
        b = old_refine(times, weights, L0, span)
        timing["refine_new"] += t1 - t0
        timing["refine_old"] += time.perf_counter() - t1
        stats["refine_calls"] += 1
        if a != b:
            stats["refine_diff"] += 1
            worst["refine_rel"] = max(worst["refine_rel"], abs(a / b - 1))
        return a

    def path(t, w, L_lo=60000 / 400, L_hi=60000 / 60, win_ms=D.WIN_MS, hop_ms=D.WIN_HOP_MS, **kw):
        t0 = time.perf_counter()
        ca, La = NEW_PATH(t, w, L_lo, L_hi, win_ms, hop_ms, **kw)
        t1 = time.perf_counter()
        if not CHECK_PATH:
            timing["path_new"] += t1 - t0
            return ca, La
        # old: same Viterbi on brute-force window coherence, via a patched window_coherence
        keep = D.window_coherence
        D.window_coherence = lambda tt, ww, ranges, Ls: np.array(
            [D.coherence(tt[a:b], ww[a:b], Ls) if b > a else np.zeros(len(Ls)) for a, b in ranges])
        try:
            cb, Lb = NEW_PATH(t, w, L_lo, L_hi, win_ms, hop_ms, **kw)
        finally:
            D.window_coherence = keep
        timing["path_new"] += t1 - t0
        timing["path_old"] += time.perf_counter() - t1
        stats["path_calls"] += 1
        if not np.array_equal(La, Lb):
            stats["path_diff"] += 1
            worst["path_rel"] = max(worst["path_rel"], float(np.max(np.abs(La / Lb - 1))))
        return ca, La

    D.refine_period, D.local_tempo_path = refine, path
    from cache_onsets import exact_onsets

    it = items[0]
    t0 = time.perf_counter()
    pcm, sr = D.decode_bass(it["audio"])
    t1 = time.perf_counter()
    o = exact_onsets(it["audio"], pcm, sr)
    t2 = time.perf_counter()
    try:
        D.detect_full(o, pcm, sr)
        err = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    timing.update(decode=t1 - t0, onsets=t2 - t1, detect=time.perf_counter() - t2)
    return it["id"], dict(stats), worst, dict(timing), err


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    groups = []
    for ds in ("ranked", "nominated", "feedback", "site"):
        groups += by_audio(list(DATASETS[ds]()))
    seen, uniq = set(), []
    for g in groups:  # the same file can sit in several datasets
        key = Path(g[0]["audio"]).stat().st_size, Path(g[0]["audio"]).name, g[0]["id"].split("/")[0]
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    uniq.sort(key=lambda g: -Path(g[0]["audio"]).stat().st_size)
    print(f"{len(uniq)} songs", flush=True)
    tot, tim, bad = Counter(), Counter(), []
    with ProcessPoolExecutor(workers, initializer=background) as ex:
        futs = [ex.submit(one, g) for g in uniq]
        for i, f in enumerate(as_completed(futs), 1):
            sid, st, worst, tm, err = f.result()
            tot.update(st)
            tim.update(tm)
            if st.get("refine_diff") or st.get("path_diff") or err:
                bad.append((sid, st, worst, err))
            if i % 100 == 0:
                print(f"  {i}/{len(uniq)}", flush=True)
    print("calls/diffs:", dict(tot))
    print("seconds:", {k: round(v) for k, v in tim.items()})
    for b in bad:
        print("  ", *b)


if __name__ == "__main__":
    main()
