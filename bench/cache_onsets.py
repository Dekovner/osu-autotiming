"""Cache onsets of every ranked item so tuning experiments skip audio work.

    uv run python bench/cache_onsets.py [workers]
    from cache_onsets import load_onsets; o = load_onsets(item_id)
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATA, ranked_items  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.detect import Onsets, onsets  # noqa: E402

CACHE = DATA / "onsets"


def path_for(item_id: str) -> Path:
    return CACHE / (item_id.replace("/", "__") + ".npz")


def load_onsets(item_id: str) -> Onsets:
    z = np.load(path_for(item_id))
    return Onsets(z["times"], z["strength"], z["low"], z["env"].astype(np.float32), z["env_low"].astype(np.float32),
                  z["sub"], float(z["hop_ms"]), z["profile"].astype(np.float32))


def one(it):
    f = path_for(it["id"])
    if f.exists():
        return
    pcm, sr = decode_bass(it["audio"])
    o = onsets(pcm, sr)
    np.savez_compressed(f, times=o.times, strength=o.strength, low=o.low, sub=o.sub, hop_ms=o.hop_ms,
                        env=o.env.astype(np.float16), env_low=o.env_low.astype(np.float16),
                        profile=o.profile.astype(np.float16))


def main():
    CACHE.mkdir(exist_ok=True)
    items = [it for it in ranked_items() if not path_for(it["id"]).exists()]
    print(f"{len(items)} to cache", flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as ex:
        for i, _ in enumerate(ex.map(one, items, chunksize=2), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
