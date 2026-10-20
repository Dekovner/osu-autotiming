"""What do multi-red-line maps actually contain?

    uv run python bench/multi_anatomy.py

Classifies every red-line transition in ranked maps (same BPM = phase/meter
reset, <2% = drift, else jump), section lengths in beats, and how much of the
playable span each kind of map covers.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ranked_items  # noqa: E402


def transition(a, b):
    r = abs(np.log(b.beat_length / a.beat_length))
    if r < 1e-4:
        # same tempo: where does the new line sit relative to the old grid?
        frac = ((b.time - a.time) / a.beat_length) % 1.0
        frac = min(frac, 1 - frac)
        return "reset_on_beat" if frac < 0.02 else ("reset_half" if abs(frac - 0.5) < 0.02 else "reset_other")
    if r < 0.02:
        return "drift<2%"
    for k, name in ((2, "x2"), (1.5, "x1.5"), (4 / 3, "x4/3")):
        if abs(r - np.log(k)) < 0.01:
            return f"ratio {name}"
    return "jump"


def main():
    kinds, sec_beats, per_map = Counter(), [], []
    for it in ranked_items():
        t = sorted(it["truth"], key=lambda s: s.time)
        if len(t) < 2:
            continue
        span_end = it["span"][1]
        types = Counter()
        for a, b in zip(t, t[1:]):
            k = transition(a, b)
            kinds[k] += 1
            types[k] += 1
        for i, s in enumerate(t):
            end = t[i + 1].time if i + 1 < len(t) else span_end
            if end > s.time:
                sec_beats.append((end - s.time) / s.beat_length)
        per_map.append((it["id"], len(t), types))
    total = sum(kinds.values())
    print(f"{len(per_map)} multi-line maps, {total} transitions")
    for k, n in kinds.most_common():
        print(f"  {k:14} {n:5} ({n / total:.0%})")
    sb = np.array(sec_beats)
    print("section length in beats: " + "  ".join(f"p{q}={np.percentile(sb, q):.0f}" for q in (10, 25, 50, 75, 90)))
    print(f"  sections shorter than 8 beats: {np.mean(sb < 8):.0%},  shorter than 16: {np.mean(sb < 16):.0%}")
    # map archetypes by dominant transition type
    arche = Counter()
    for _, n, types in per_map:
        dom = types.most_common(1)[0][0]
        arche[("live-ish drift" if dom == "drift<2%" else "resets" if dom.startswith("reset") else "jumps") + (" (many lines)" if n > 5 else " (few lines)")] += 1
    print("maps by dominant transition:")
    for k, n in arche.most_common():
        print(f"  {k:28} {n}")


if __name__ == "__main__":
    main()
