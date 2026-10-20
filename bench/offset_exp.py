"""Experiment: refine the offset from a beat-folded energy profile.

For each map where our cached prediction has the right tempo, fold the audio
energy on the predicted grid and find the steepest rise; report how that rise
sits relative to the *ranked* grid. A tight distribution = a good offset cue.

    uv run python bench/offset_exp.py [workers]
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RESULTS, ranked_items  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.grid import Segment, beats  # noqa: E402

OFFS = np.arange(-50, 50.5, 0.5)
BANDS = {"broad": None, "hp1k": 1000, "hp3k": 3000, "hp6k": 6000}


def rise(prof):
    """Position (ms) of the steepest rise of a folded log-energy profile, sub-step refined."""
    p = uniform_filter1d(prof, 5)
    d = np.gradient(p)
    i = int(np.argmax(d[10:-10])) + 10
    a, b, c = d[i - 1], d[i], d[i + 1]
    frac = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
    return float(OFFS[i] + np.clip(frac, -0.5, 0.5) * (OFFS[1] - OFFS[0])), float(p[i + 20:i + 60].mean() - p[i - 60:i - 20].mean())


def one(args):
    it, pred = args
    x, sr = decode_bass(it["audio"])
    out = {"id": it["id"]}
    segs = [Segment(t, L) for t, L in pred]
    for sub in (1, 2):
        grid = [Segment(s.time, s.beat_length / sub) for s in segs]
        bt = beats(grid, *it["span"])
        for name, hp in BANDS.items():
            sig = x if hp is None else sosfilt(butter(4, hp, "highpass", fs=sr, output="sos"), x)
            e = sig.astype(np.float64) ** 2
            k = max(1, int(sr / 1000))
            c = np.cumsum(np.concatenate([[0], e]))
            env = np.log10((c[k:] - c[:-k]) / k + 1e-10)
            idx = ((bt[:, None] + OFFS[None, :]) * sr / 1000).astype(int)
            idx = idx[(idx[:, 0] > 0) & (idx[:, -1] < len(env))]
            if len(idx) < 16:
                continue
            r, contrast = rise(env[idx].mean(axis=0))
            # where our grid's rise sits, expressed against the ranked grid
            pred_t = segs[0].time
            true_t = it["truth"][0].time
            L = segs[0].beat_length
            delta = (pred_t - true_t + L / 2) % L - L / 2  # pred grid minus true grid
            out[f"{name}/{sub}"] = (r + delta, contrast)
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    preds = {r["id"]: r for r in map(json.loads, (RESULTS / "ranked_ours.jsonl").open())}
    todo = []
    for it in ranked_items():
        r = preds.get(it["id"])
        if r and r.get("ok") and it["tags"]["kind"] == "single" and r["octave"] == 1 and r["p95_ms"] < 30 and len(r["pred"]) == 1:
            todo.append((it, r["pred"]))
    print(f"{len(todo)} maps", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo, chunksize=2))
    (RESULTS.parent / "offset_exp.json").write_text(json.dumps(res))
    base = np.array([preds[r["id"]]["bias_ms"] for r in res])
    print(f"current method: IQR {np.subtract(*np.percentile(base, [75, 25])):.2f}  within 5ms of median {np.mean(np.abs(base - np.median(base)) < 5):.0%}")
    for key in [f"{b}/{s}" for s in (1, 2) for b in BANDS]:
        v = np.array([r[key][0] for r in res if key in r])
        med = np.median(v)
        print(f"{key:10} rise vs ranked line: median {med:+6.2f} ms  IQR {np.subtract(*np.percentile(v, [75, 25])):5.2f}"
              f"  within 3ms {np.mean(np.abs(v - med) < 3):.0%}  5ms {np.mean(np.abs(v - med) < 5):.0%}")


if __name__ == "__main__":
    main()
