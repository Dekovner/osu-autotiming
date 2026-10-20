"""Where does our grid sit relative to ranked timing, per file?

    uv run python bench/bias.py [method]

Prints the bias distribution (is it one constant or several clusters?), splits
it by MP3 header type, and re-scores everything with the median bias removed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RESULTS, ranked_items  # noqa: E402

from autotiming.grid import Segment, compare  # noqa: E402


def mp3_kind(path: Path) -> str:
    if path.suffix.lower() != ".mp3":
        return path.suffix.lower()
    head = path.read_bytes()[:64 * 1024]
    if b"LAME" in head or b"Lavc" in head or b"Lavf" in head:
        tag = "lame" if b"LAME" in head else "lavc"
        return f"mp3+{tag}" + ("+xing" if b"Xing" in head or b"Info" in head else "")
    if b"Xing" in head or b"Info" in head:
        return "mp3+xing"
    if b"VBRI" in head:
        return "mp3+vbri"
    return "mp3 bare"


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "ours"
    items = {it["id"]: it for it in ranked_items()}
    rows = [json.loads(l) for l in (RESULTS / f"ranked_{method}.jsonl").open()]
    rows = [r for r in rows if r.get("ok") and r["octave"] in (0.5, 1, 2) and r["id"] in items]
    good = [r for r in rows if r["p95_ms"] < 40]
    b = np.array([r["bias_ms"] for r in good])
    print(f"{len(good)} files with the right tempo family and p95<40ms")
    print("bias histogram (ms):")
    hist, edges = np.histogram(b, bins=np.arange(-30, 46, 3))
    for h, e in zip(hist, edges):
        print(f"  {e:+4.0f}..{e + 3:+4.0f} {'#' * h}")
    by_kind = defaultdict(list)
    for r in good:
        by_kind[mp3_kind(Path(items[r["id"]]["audio"]))].append(r["bias_ms"])
    print("by audio header:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:18} n={len(v):3}  median {np.median(v):+6.1f}  IQR {np.percentile(v, 25):+6.1f}..{np.percentile(v, 75):+6.1f}")
    shift = float(np.median(b))
    p95 = []
    for r in rows:
        it = items[r["id"]]
        pred = [Segment(t - shift, L) for t, L in r["pred"]]
        res = compare(pred, it["truth"], *it["span"])
        p95.append(res["p95_ms"])
    p95 = np.array(p95)
    print(f"after removing {shift:+.1f} ms (of {len(rows)} right-tempo files): "
          f"p95<5ms {np.mean(p95 < 5):.0%}  <10ms {np.mean(p95 < 10):.0%}  <20ms {np.mean(p95 < 20):.0%}")


if __name__ == "__main__":
    main()
