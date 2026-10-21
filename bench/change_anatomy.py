"""Where do multi-red-line maps lose their beats?

    uv run python bench/change_anatomy.py ranked _fast1 [--kind many] [--list]

Every true beat more than 10 ms from our grid (at the map's octave) is put in one
bucket, by the true section it sits in and our local tempo there:
  octave   our beat length there is x2 / x0.5 of the true one (when the map isn't)
  tempo    off by more than 1% (and no clean octave)
  near     within 4 beats of a true tempo jump (boundary in the wrong place)
  phase    tempo right, beats shifted
and by the length of its true section (short: < 8 beats, i.e. a drift/rubato line).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATASETS, RESULTS  # noqa: E402

from autotiming.grid import Segment, _nearest_dist, beats, subdivide  # noqa: E402


def local_bl(segs, t):
    """Beat length of the red line in force at each time t."""
    times = np.array([s.time for s in segs])
    idx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(segs) - 1)
    return np.array([segs[i].beat_length for i in idx]), idx


def anatomy(item, row):
    truth = sorted(item["truth"], key=lambda s: s.time)
    pred = sorted((Segment(t, L) for t, L in row["pred"]), key=lambda s: s.time)
    start, end = item["span"]
    oct_ = row.get("octave") or 1
    tf = subdivide(truth, 2) if oct_ == 2 else truth
    pf = subdivide(pred, 2) if oct_ == 0.5 else pred
    tb = beats(tf, start, end)
    if len(tb) < 2:
        return None
    margin = 4 * max(s.beat_length for s in tf + pf)
    err = np.abs(_nearest_dist(tb, beats(pf, start - margin, end + margin)))
    tbl, tidx = local_bl(tf, tb)
    pbl, _ = local_bl(pf, tb)
    r = np.log(tbl / pbl)
    # true tempo jumps (>2%) and the section lengths in beats
    jumps = [tf[i].time for i in range(1, len(tf)) if abs(np.log(tf[i].beat_length / tf[i - 1].beat_length)) > 0.02]
    ends = [tf[i + 1].time if i + 1 < len(tf) else end for i in range(len(tf))]
    sec_beats = np.array([(min(ends[i], end) - max(tf[i].time, start)) / tf[i].beat_length for i in range(len(tf))])
    short = sec_beats[tidx] < 8
    jt = np.array(jumps) if jumps else np.array([np.inf])
    near = np.min(np.abs(tb[:, None] - jt[None, :]), axis=1) < 4 * tbl
    bad = err >= 10
    cls = np.where(np.abs(np.abs(r) - np.log(2)) < 0.02, "octave",
                   np.where(np.abs(r) > 0.01, np.where(near, "near", "tempo"), np.where(near, "near", "phase")))
    out = Counter()
    out["beats"] = len(tb)
    out["good"] = int(np.sum(~bad))
    for c, s in zip(cls[bad], short[bad]):
        out[f"{c}/{'short' if s else 'long'}"] += 1
    out["short_beats"] = int(np.sum(short))
    # how far off the phase errors are: a few ms (offset), a half beat, a quarter
    ph = bad & (cls == "phase")
    frac = err[ph] / tbl[ph]
    for lo, hi, name in ((0, 0.06, "ms"), (0.06, 0.2, "1/8"), (0.2, 0.4, "1/4"), (0.4, 0.6, "1/2")):
        out[f"~phase {name}"] += int(np.sum((frac >= lo) & (frac < hi)))
    return out


def stable_jumps(segs, start, end, min_beats=8):
    """(time, old L, new L) of tempo changes > 2% into a section that holds >= min_beats beats."""
    segs = sorted(segs, key=lambda s: s.time)
    out = []
    for i in range(1, len(segs)):
        a, b = segs[i - 1], segs[i]
        if abs(np.log(b.beat_length / a.beat_length)) < 0.02 or not start <= b.time <= end:
            continue
        nxt = segs[i + 1].time if i + 1 < len(segs) else end
        if (nxt - b.time) / b.beat_length >= min_beats:
            out.append((b.time, a.beat_length, b.beat_length))
    return out


def changes(item, row, tol_beats=2):
    """Matched / missed / extra stable tempo changes, ours vs the mapper's."""
    start, end = item["span"]
    tj = stable_jumps(item["truth"], start, end)
    pj = stable_jumps([Segment(t, L) for t, L in row["pred"]], start, end)
    used, hit, close = set(), 0, 0
    out = Counter()
    for t, a, L in tj:
        cand = [(abs(p - t), k) for k, (p, _, _) in enumerate(pj) if k not in used and abs(p - t) < tol_beats * L]
        if cand:
            d, k = min(cand)
            used.add(k)
            hit += 1
            close += d < 10
        else:
            out[f"miss {ratio_class(a, L)}"] += 1
    for k, (p, a, L) in enumerate(pj):
        if k not in used:
            out[f"extra {ratio_class(a, L)}"] += 1
    return {"true": len(tj), "pred": len(pj), "hit": hit, "close": close, **out}


def ratio_class(a, b):
    r = abs(np.log(b / a))
    for k, name in ((2, "x2"), (1.5, "2:3"), (4 / 3, "3:4"), (1.25, "4:5")):
        if abs(r - np.log(k)) < 0.01:
            return name
    return "<10%" if r < np.log(1.1) else "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("tag")
    p.add_argument("--kind", default="many")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    items = {it["id"]: it for it in DATASETS[args.dataset]()}
    rows = [json.loads(l) for l in (RESULTS / f"{args.dataset}_ours{args.tag}.jsonl").open()]
    tot, per = Counter(), []
    for row in rows:
        it = items.get(row["id"])
        if not it or not row.get("ok") or (args.kind != "all" and row["tags"]["kind"] != args.kind):
            continue
        a = anatomy(it, row)
        if a:
            a.update({f"chg_{k}": v for k, v in changes(it, row).items()})
            tot += a
            per.append((row["id"], a, row))
    n = tot["beats"]
    print(f"stable tempo changes (>2%, new tempo holds >= 8 beats): true {tot['chg_true']}, ours {tot['chg_pred']}, "
          f"found {tot['chg_hit']} ({tot['chg_hit'] / max(1, tot['chg_true']):.0%}; {tot['chg_close']} within 10 ms), "
          f"extra {tot['chg_pred'] - tot['chg_hit']}")
    print("  " + ", ".join(f"{k[4:]} {tot[k]}" for k in sorted(tot) if k.startswith("chg_miss") or k.startswith("chg_extra")))
    print(f"{len(per)} maps, {n} true beats, {tot['good'] / n:.1%} within 10 ms; {tot['short_beats'] / n:.1%} of beats in sections < 8 beats")
    for k in sorted(k for k in tot if "/" in k):
        print(f"  {k:14} {tot[k] / n:6.1%}")
    # maps by how many beats they lose
    if args.list:
        per.sort(key=lambda x: -(x[1]["beats"] - x[1]["good"]))
        for i, a, row in per[:40]:
            lost = a["beats"] - a["good"]
            top = ", ".join(f"{k} {v}" for k, v in a.most_common() if "/" in k)[:80]
            print(f"  {i[:45]:45} lost {lost:5}/{a['beats']:5} oct {row['octave']}  {top}")


if __name__ == "__main__":
    main()
