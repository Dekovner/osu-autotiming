"""Maps whose result changed between two bench tags.

    uv run python bench/flips.py ranked _fast1 _rt1 [--kind single]

Lists maps that left / joined "whole map p95 < 10 ms" and the biggest coverage moves.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RESULTS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--kind")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    load = lambda t: {r["id"]: r for r in map(json.loads, (RESULTS / f"{args.dataset}_ours{t}.jsonl").open())}
    A, B = load(args.a), load(args.b)
    good = lambda r: r.get("ok") and r["octave"] in (0.5, 1, 2) and r["p95_ms"] < 10
    rows = [(i, A[i], B[i]) for i in A if i in B and (not args.kind or A[i]["tags"]["kind"] == args.kind)]
    lost = [x for x in rows if good(x[1]) and not good(x[2])]
    won = [x for x in rows if good(x[2]) and not good(x[1])]
    print(f"{len(rows)} maps: whole<10 lost {len(lost)}, won {len(won)}")
    fmt = lambda r: f"p95 {r.get('p95_ms', 0):6.1f} cov {r.get('cover10', 0):.2f} lines {len(r.get('pred') or [])}"
    for name, xs in (("lost", lost), ("won", won)):
        for i, a, b in xs[: args.n]:
            print(f"  {name} {i[:40]:40} {fmt(a)}  ->  {fmt(b)}")
    moves = sorted(rows, key=lambda x: x[2].get("cover10", 0) - x[1].get("cover10", 0))
    print("coverage moves:")
    for i, a, b in moves[:5] + moves[-5:]:
        print(f"  {i[:40]:40} {a.get('cover10', 0):.2f} -> {b.get('cover10', 0):.2f}  lines {len(a.get('pred') or [])} -> {len(b.get('pred') or [])}")


if __name__ == "__main__":
    main()
