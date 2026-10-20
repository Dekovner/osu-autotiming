"""Per-map listing for a small dataset: ranked timing vs each method.

    uv run python bench/per_map.py nominated ours timing_analyzer
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATA, load  # noqa: E402


def cell(r):
    if not r or not r.get("ok"):
        return f"{'fail':>26}"
    oct_ = {1: "", 2: " x2", 0.5: " x½", 0: " BPM?"}[r["octave"]]
    return f"{r['bias_ms']:+7.1f} {r['p95_ms']:6.1f} {r['cover5']:4.0%}{oct_:5}"


def main():
    dataset, methods = sys.argv[1], sys.argv[2:]
    data = {m: {r["id"]: r for r in load(dataset, m)} for m in methods}
    names = {}
    for d in (DATA / dataset).glob("*"):
        osu = next(d.glob("*.osu"), None)
        if osu:
            names[d.name] = osu.name.split(" (")[0][:34]
    ids = sorted(set().union(*data.values()), key=lambda i: -max(abs(data[m].get(i, {}).get("bias_ms", 0)) for m in methods))
    print(f"{'map':40} {'ranked':>15} | " + " | ".join(f"{m + ': bias p95 cov5':>26}" for m in methods))
    for i in ids:
        r0 = next(data[m][i] for m in methods if i in data[m])
        t = r0["truth"]
        ranked = f"{60000 / t[0][1]:7.2f}" + (f" +{len(t) - 1}ln" if len(t) > 1 else "     ")
        print(f"{names.get(i.split('/')[0], i)[:40]:40} {ranked:>15} | " + " | ".join(cell(data[m].get(i)) for m in methods))


if __name__ == "__main__":
    main()
