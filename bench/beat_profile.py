"""Average sound energy around ranked beats: where is the attack relative to the red line?

    uv run python bench/beat_profile.py [max_files]

Uses no onset detector at all, so it separates "osu!/mapper convention" from
"our detector is late". Prints the mean profile from -60 to +60 ms around every
true beat (single red line maps), broadband and above 2 kHz.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ranked_items  # noqa: E402

from autotiming.decode import decode_bass  # noqa: E402
from autotiming.grid import beats  # noqa: E402


def envelope(x, sr):
    e = x.astype(np.float64) ** 2
    k = max(1, int(sr / 1000))  # 1 ms
    c = np.cumsum(np.concatenate([[0], e]))
    return (c[k:] - c[:-k]) / k


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    offs = np.arange(-60, 61)
    acc = {"broad": np.zeros(len(offs)), "hf": np.zeros(len(offs))}
    n = 0
    for it in ranked_items():
        if it["tags"]["segments"] != 1:
            continue
        x, sr = decode_bass(it["audio"])
        hf = sosfilt(butter(4, 2000, "highpass", fs=sr, output="sos"), x)
        bt = beats(it["truth"], *it["span"])
        for name, sig in (("broad", x), ("hf", hf)):
            env = envelope(sig, sr)
            idx = (bt[:, None] + offs[None, :]) * sr / 1000
            idx = idx[(idx[:, 0] > 0) & (idx[:, -1] < len(env) - 1)].astype(int)
            prof = np.log10(env[idx] + 1e-10).mean(axis=0)
            acc[name] += prof - prof.mean()
        n += 1
        if n >= limit:
            break
    print(f"{n} single-red-line maps; mean log-energy around ranked beats (0 = red line grid)")
    for name, prof in acc.items():
        prof = prof / n
        d = np.diff(prof)
        print(f"\n[{name}] steepest rise at {offs[np.argmax(d)] + 0.5:+.1f} ms, peak at {offs[np.argmax(prof)]:+d} ms")
        lo, hi = prof.min(), prof.max()
        for o, v in zip(offs, prof):
            if o % 3 == 0:
                print(f"  {o:+4d} ms {'#' * int(40 * (v - lo) / (hi - lo + 1e-12))}")


if __name__ == "__main__":
    main()
