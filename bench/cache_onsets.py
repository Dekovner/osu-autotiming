"""Cache onsets of every ranked item so tuning experiments skip audio work.

    uv run python bench/cache_onsets.py [workers]
    from cache_onsets import load_onsets; o = load_onsets(item_id)

load_onsets keeps the envelopes in float16 (fine for tuning experiments). The bench
itself goes through exact_onsets: full precision, keyed by the audio's md5 and by
a hash of the code that makes onsets, so it gives what the site would.
"""

import hashlib
import inspect
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DATA, ranked_items  # noqa: E402

from autotiming import decode as _decode  # noqa: E402
from autotiming import detect as _detect  # noqa: E402
from autotiming.decode import decode_bass  # noqa: E402
from autotiming.detect import Onsets, onsets  # noqa: E402

CACHE = DATA / "onsets"
EXACT = DATA / "onsets_exact"
FIELDS = ("times", "strength", "low", "env", "env_low", "sub", "hop_ms", "profile")


@cache
def onsets_version() -> str:
    """Hash of decoding + onset code and its constants: changing either starts a fresh cache."""
    D = _detect
    src = (inspect.getsource(_decode) + inspect.getsource(D.onsets) + inspect.getsource(D._bands)
           + repr((D.HOP_MS, D.N_FFT, D.N_BANDS, D.FLUX_LAG, D.ONSET_BIAS_MS)))
    return hashlib.md5(src.encode()).hexdigest()[:12]


def exact_onsets(path, pcm, sr) -> Onsets:
    """onsets(pcm, sr), from disk when this audio file was seen with the same onset code."""
    f = EXACT / onsets_version() / (hashlib.md5(Path(path).read_bytes()).hexdigest() + f"_{sr}.npz")
    if f.exists():
        z = np.load(f)
        return Onsets(*(float(z[k]) if k == "hop_ms" else z[k] for k in FIELDS))
    o = onsets(pcm, sr)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **{k: getattr(o, k) for k in FIELDS})
    tmp.replace(f)
    return o


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
