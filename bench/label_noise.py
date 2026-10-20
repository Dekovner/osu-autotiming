"""How much do ranked timings disagree with each other?

Sets that ship byte-identical audio were timed independently by different
mappers; their grid-vs-grid difference is the noise floor of ranked labels.

    uv run python bench/label_noise.py
"""

import hashlib
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ranked_items  # noqa: E402

from autotiming.grid import compare  # noqa: E402


def main():
    by_md5 = defaultdict(list)
    for it in ranked_items():
        md5 = hashlib.md5(Path(it["audio"]).read_bytes()).hexdigest()
        by_md5[md5].append(it)
    rows = []
    for items in by_md5.values():
        for a, b in combinations(items, 2):
            if a["tags"]["set"] == b["tags"]["set"]:
                continue
            lo, hi = max(a["span"][0], b["span"][0]), min(a["span"][1], b["span"][1])
            if hi - lo < 30000:
                continue
            r = compare(a["truth"], b["truth"], lo, hi)
            if r.get("ok"):
                rows.append((a, b, r))
    print(f"{len(rows)} pairs of independently timed identical audio")
    if not rows:
        return
    oct_ok = [r for *_, r in rows if r["octave"] in (0.5, 1, 2)]
    print(f"  same tempo family: {len(oct_ok)}/{len(rows)}, of those x2/half: {sum(r['octave'] != 1 for r in oct_ok)}")
    bias = np.abs([r["bias_ms"] for r in oct_ok])
    p95 = np.array([r["p95_ms"] for r in oct_ok])
    for q in (50, 75, 90):
        print(f"  |offset difference| p{q}: {np.percentile(bias, q):5.1f} ms    grid p95 diff p{q}: {np.percentile(p95, q):5.1f} ms")
    print("  worst pairs:")
    for a, b, r in sorted(rows, key=lambda x: -x[2]["p95_ms"])[:8]:
        print(f"    {a['id'][:40]:40} vs {b['id'][:40]:40} oct={r['octave']} bias={r['bias_ms']:+.1f} p95={r['p95_ms']:.1f}"
              f"  red lines {len(a['truth'])}/{len(b['truth'])}")


if __name__ == "__main__":
    main()
