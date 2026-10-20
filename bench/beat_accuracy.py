"""How close are the cached tracked beats to ranked beats, per style (honours DEV and TRACK_* settings).

    uv run python bench/beat_accuracy.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_eval import beats_file, items_and_current  # noqa: E402

from autotiming.detect import OSU_OFFSET_MS  # noqa: E402
from autotiming.grid import beats as grid_beats  # noqa: E402


def main():
    items, _ = items_and_current()
    by = defaultdict(list)
    for it in items:
        f = beats_file(it["id"])
        if not f.exists():
            continue
        mine = np.load(f)["times"] + OSU_OFFSET_MS
        if len(mine) < 2:
            continue
        tb = grid_beats(it["truth"], *it["span"])
        j = np.clip(np.searchsorted(mine, tb), 1, len(mine) - 1)
        d = np.minimum(np.abs(tb - mine[j - 1]), np.abs(tb - mine[j]))
        by[it["style"]].append((np.mean(d < 5), np.mean(d < 10), np.mean(d < 25)))
    print("tracked beats within 5/10/25 ms of ranked beats:")
    for s in ("single", "sections", "bar-by-bar", "beat-by-beat"):
        if by[s]:
            a = np.array(by[s])
            print(f"  {s:13} n={len(a):3}  {a[:, 0].mean():4.0%} {a[:, 1].mean():4.0%} {a[:, 2].mean():4.0%}")


if __name__ == "__main__":
    main()
