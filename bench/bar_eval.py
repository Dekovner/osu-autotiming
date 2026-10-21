"""Is "1" where the mapper put it? Bar agreement from cached bench predictions.

    uv run python bench/bar_eval.py ranked ours_perf ours_db1 [--ids data/subset.txt]

For each mapper downbeat (a red line plus whole bars, inside the mapped span) where
our grid has the same tempo (0.5%) and a beat within 15 ms, it counts whether that
beat is "1" of our bar too. Grid errors are the main bench's business: they only
shrink the count here. "bars" pools every such downbeat, "maps" averages per map,
"maps 90%" is the share of maps with at least 90% of their bars right.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

TOL_MS, TEMPO_TOL = 15.0, 0.005


def bars(row, span) -> tuple[int, int]:
    """(downbeats compared, of them on our "1"), mapper downbeats inside span (first, last object)."""
    truth = sorted(row["truth"])
    pred = sorted(row.get("pred") or [])
    if not pred:
        return 0, 0
    lo, hi = span
    starts = np.array([p[0] for p in pred])
    n = right = 0
    for i, (t, L) in enumerate(truth):
        end = min(truth[i + 1][0] if i + 1 < len(truth) else hi, hi)
        d = t + 4 * L * np.arange(0, max(0, int((end - t) / (4 * L))) + 1)
        d = d[(d < end) & (d >= lo - 4 * L)]
        for x in d:
            k = max(0, int(np.searchsorted(starts, x + TOL_MS, side="right")) - 1)
            p, pL = pred[k]
            if abs(np.log(pL / L)) > TEMPO_TOL:
                continue
            b = (x - p) / pL
            if abs(b - round(b)) * pL > TOL_MS:
                continue
            n += 1
            right += round(b) % 4 == 0
    return n, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tags", nargs="+", help="result files: <dataset>_<tag>.jsonl, e.g. ours_perf")
    ap.add_argument("--ids")
    args = ap.parse_args()
    keep = set(Path(args.ids).read_text().split()) if args.ids else None
    rows = {}
    for tag in args.tags:
        f = RESULTS / f"{args.dataset}_{tag}.jsonl"
        rows[tag] = {r["id"]: r for r in map(json.loads, f.open()) if keep is None or r["id"] in keep}
    common = set.intersection(*(set(r) for r in rows.values()))
    spans = {it["id"]: it["span"] for it in DATASETS[args.dataset]() if it["id"] in common}
    common &= set(spans)
    print(f"{len(common)} maps in common")
    per = {}
    for tag in args.tags:
        res = {i: bars(rows[tag][i], spans[i]) for i in sorted(common)}
        per[tag] = res
        n = sum(a for a, _ in res.values())
        ok = sum(b for _, b in res.values())
        maps = [b / a for a, b in res.values() if a]
        print(f"  {tag:16} bars {ok / max(n, 1):6.1%} of {n:6}   maps {np.mean(maps):6.1%}   "
              f"maps 90% {np.mean(np.array(maps) >= 0.9):6.1%}   (maps compared {len(maps)})")
    if len(args.tags) == 2:
        a, b = args.tags
        diff = []
        for i in common:
            (na, ra), (nb, rb) = per[a][i], per[b][i]
            fa, fb = (ra / na if na else None), (rb / nb if nb else None)
            if fa is not None and fb is not None and abs(fa - fb) > 0.05:
                diff.append((fb - fa, i, fa, fb))
        diff.sort()
        print(f"  {b} vs {a}: better on {sum(d > 0 for d, *_ in diff)}, worse on {sum(d < 0 for d, *_ in diff)} maps (>5% of bars)")
        for d, i, fa, fb in diff[:8] + diff[-8:]:
            print(f"    {d:+.0%}  {fa:.0%} -> {fb:.0%}  {i}")


if __name__ == "__main__":
    main()
