"""Head-to-head table on the maps every listed method has a result for.

    uv run python bench/table.py ranked ours timing_analyzer librosa essentia
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import load  # noqa: E402


def stats(rows):
    ok = [r for r in rows if r.get("ok")]
    n = len(rows)
    octs = Counter(r["octave"] for r in ok)
    tempo_ok = [r for r in ok if r["octave"] in (0.5, 1, 2)]
    p95 = np.array([r["p95_ms"] for r in tempo_ok] + [np.inf] * (n - len(tempo_ok)))
    # offbeat: right tempo, grid sits a half beat off
    half = sum(1 for r in tempo_ok if r["octave"] == 1 and 0.35 < abs(r["bias_ms"]) / (r["truth"][0][1]) < 0.65)
    return {
        "n": n,
        "x1": octs[1] / n,
        "x2": (octs[2] + octs[0.5]) / n,
        "wrong": (octs[0] + n - len(ok)) / n,
        "cov5": sum(r.get("cover5", 0) for r in ok) / n,
        "cov10": sum(r.get("cover10", 0) for r in ok) / n,
        "all5": float(np.mean(p95 < 5)),
        "all10": float(np.mean(p95 < 10)),
        "offbeat": half / n,
    }


def main():
    dataset, methods = sys.argv[1], sys.argv[2:]
    data = {m: {r["id"]: r for r in load(dataset, m)} for m in methods}
    common = set.intersection(*(set(d) for d in data.values()))
    kinds = sorted({data[methods[0]][i]["tags"].get("kind", "-") for i in common})
    for kind in ["all", *kinds]:
        ids = [i for i in common if kind == "all" or data[methods[0]][i]["tags"].get("kind") == kind]
        if not ids:
            continue
        print(f"\n[{kind}] {len(ids)} maps")
        print(f"  {'method':16} {'covered±5':>9} {'covered±10':>10} {'whole±5':>8} {'whole±10':>9} {'tempo x1':>8} {'x2/half':>7} {'wrong':>6} {'offbeat':>7}")
        for m in methods:
            s = stats([data[m][i] for i in ids])
            print(f"  {m:16} {s['cov5']:9.0%} {s['cov10']:10.0%} {s['all5']:8.0%} {s['all10']:9.0%} {s['x1']:8.0%} {s['x2']:7.0%} {s['wrong']:6.0%} {s['offbeat']:7.0%}")


if __name__ == "__main__":
    main()
