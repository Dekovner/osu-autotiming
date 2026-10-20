"""BPM rounding on cached predictions: how round are mappers' BPMs, and what does snapping cost?

    uv run python bench/snap_eval.py ranked ours
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def worker(dataset, method):
    import numpy as np

    sys.path.insert(0, str(HERE))
    from run import DATASETS, load

    from autotiming.detect import connect_lines, snap_bpm
    from autotiming.grid import Segment, compare

    items = {it["id"]: it for it in DATASETS[dataset]()}
    rows = [r for r in load(dataset, method) if r.get("ok") and r["id"] in items]
    cov0, cov1, c5_0, c5_1, round_ok, round_n = [], [], [], [], 0, 0
    for r in rows:
        it = items[r["id"]]
        base = connect_lines([Segment(t, L) for t, L in r["pred"]])
        snapped = snap_bpm(base, it["span"][1] + 2000)
        a, b = compare(base, it["truth"], *it["span"]), compare(snapped, it["truth"], *it["span"])
        cov0.append(a["cover10"]); cov1.append(b["cover10"]); c5_0.append(a["cover5"]); c5_1.append(b["cover5"])
        if len(it["truth"]) == 1 and r["octave"] == 1:  # single-BPM maps with the right octave: exact BPM match?
            round_n += 1
            round_ok += abs(snapped[0].bpm - it["truth"][0].bpm) < 1e-6
    print(json.dumps({"n": len(rows), "cov10": [float(np.mean(cov0)), float(np.mean(cov1))],
                      "cov5": [float(np.mean(c5_0)), float(np.mean(c5_1))], "exact_bpm": round_ok / max(1, round_n), "singles": round_n}))


def main():
    if sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3])
    dataset, method = sys.argv[1], sys.argv[2]
    sys.path.insert(0, str(HERE))
    from run import DATASETS

    import numpy as np

    bpms = np.array([s.bpm for it in DATASETS[dataset]() for s in it["truth"]])
    for step in (1.0, 0.5, 0.25, 0.1):
        print(f"ranked red lines with BPM a multiple of {step:g}: {np.mean(np.abs(bpms / step - np.round(bpms / step)) < 1e-3):.0%}")
    for tol in ("1", "2", "3", "5"):
        out = subprocess.run([sys.executable, __file__, "--worker", dataset, method], env=dict(os.environ, SNAP_MAX_MS=tol),
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        r = json.loads(out)
        print(f"SNAP_MAX_MS={tol}: maps {r['n']}  cov10 {r['cov10'][0]:.1%} -> {r['cov10'][1]:.1%}  cov5 {r['cov5'][0]:.1%} -> {r['cov5'][1]:.1%}  "
              f"exact mapper BPM on single-BPM maps {r['exact_bpm']:.0%} (n={r['singles']})")


if __name__ == "__main__":
    main()
