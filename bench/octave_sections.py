"""Per-section octave choice vs one octave for the whole song, on cached raw segments.

    uv run python bench/octave_sections.py '[{"OCT_PER_SECTION": "0"}, {"OCT_CONT_LAMBDA": "1"}]'

For multi-section songs: share of playable time whose output BPM matches the
ranked BPM there (same octave, within 2%), and coverage within 10 ms.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def worker():
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    sys.path.insert(0, str(HERE))
    from run import DATA, ranked_items

    raw = json.loads((DATA / "raw_segments.json").read_text())
    items = [it for it in ranked_items() if isinstance(raw.get(it["id"]), list) and len(raw[it["id"]]) >= 2]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "12"))) as ex:
        res = list(ex.map(_one, [(it, raw[it["id"]]) for it in items], chunksize=4))
    res = [r for r in res if r]
    print(json.dumps({"n": len(res), "bpm_right": float(np.mean([r[0] for r in res])), "cov10": float(np.mean([r[1] for r in res]))}))


def _one(args):
    it, raw = args
    import numpy as np

    sys.path.insert(0, str(HERE))
    from cache_onsets import load_onsets

    from autotiming.detect import OSU_OFFSET_MS, choose_octave
    from autotiming.grid import Segment, compare

    try:
        o = load_onsets(it["id"])
        segs = choose_octave(o, [Segment(t, L) for t, L in raw])
        out = [Segment(s.time + OSU_OFFSET_MS, s.beat_length) for s in segs]
        truth = sorted(it["truth"], key=lambda s: s.time)
        lo, hi = it["span"]
        grid = np.arange(lo, hi, 250.0)

        def bpm_at(ss, x):
            starts = [s.time for s in ss]
            k = max(0, np.searchsorted(starts, x, side="right") - 1)
            return ss[k].bpm

        ok = np.mean([abs(np.log(bpm_at(out, x) / bpm_at(truth, x))) < 0.02 for x in grid])
        return ok, compare(out, it["truth"], lo, hi).get("cover10", 0)
    except Exception:
        return None


def main():
    if sys.argv[1] == "--worker":
        return worker()
    for c in json.loads(sys.argv[1]):
        env = dict(os.environ, **{k: str(v) for k, v in c.items()})
        out = subprocess.run([sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"{json.dumps(c):48} maps {r['n']}  time with the mapper's BPM {r['bpm_right']:.1%}  coverage ±10 {r['cov10']:.1%}", flush=True)


if __name__ == "__main__":
    main()
