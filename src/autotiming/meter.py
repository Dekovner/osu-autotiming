"""Bars of 3 or of 4: the song's meter from how its bass, chords and hits repeat.

Every other bar decision in the pipeline counts 4 beats; 7.5% of ranked maps are mostly
3/4 or 6/4 (bars of 6 beats are two bars of 3 at the bar level), and in 4/4 their
barlines are wrong in game whatever the grid. On the beat grid, music repeats in bars and
4-bar phrases: every 3 / 6 / 12 beats in 3/4, every 4 / 8 / 16 in 4/4. Boosted trees over
those periodicities (bench/meter_exp.py, crossfit) found 92% of the 88 songs in 3 at
p >= 0.9 with no false call on 1599 songs in 4. apply_meter() then puts each line on "1" of
a bar of 3 along its own grid: the grid itself does not move.
"""

from pathlib import Path

import numpy as np

from .downbeat import CHROMA_HOP_MS, EDGES, GROUPS, _beat_means, _boundary_change, bar_features

PERIODS = (2, 3, 4, 6, 8, 12, 16)
PAIRS = ((3, 4), (6, 8), (12, 16), (3, 2), (6, 4))  # the 3-ish period against its 4-ish twin
MIN_BEATS = 48  # 16 bars of 3: enough phrases to tell
METER_TREES = Path(__file__).resolve().parent / "models" / "meter_trees.npz"
METER_P = 0.9  # p(bars of 3) to switch: no false call on 4/4 in crossfit even at 0.5
# "1" of a bar of 3: z-scored sum of these bar_features over the 3 candidates (88 songs in 3:
# 100% of the mappers' "1"; each alone 92-95%)
PHASE_KEYS = ("ch_change2", "chroma_change", "bass_change2", "bass_phrase", "strength_on_one")
PHASE_MIN_BEATS = 12  # a line shorter than 4 bars keeps its start


def sequences(o, h, beats: np.ndarray) -> dict:
    """Per-beat signals on beats[:-1] (onset clock)."""
    B = _beat_means(h.bass, h.bass_hop_ms, beats)
    C = _beat_means(h.chroma, CHROMA_HOP_MS, beats)
    seq = {
        "bass1": _boundary_change(B, 1)[:-1], "bass2": _boundary_change(B, 2)[:-1],
        "chord1": _boundary_change(C, 1)[:-1], "chord2": _boundary_change(C, 2)[:-1],
    }
    L = float(np.median(np.diff(beats)))
    tol = min(25.0, L / 6)
    k = np.searchsorted(beats, o.times)
    near = np.full(len(o.times), -1)
    for j in (k - 1, k):
        jj = np.clip(j, 0, len(beats) - 1)
        ok = np.abs(o.times - beats[jj]) < tol
        near[ok] = jj[ok]
    sel = (near >= 0) & (near < len(beats) - 1)
    prof = o.profile.astype(np.float64)
    for g, (a, b) in GROUPS.items():
        m = prof[:, (EDGES[:-1] >= a) & (EDGES[:-1] < b)].sum(1)
        seq[f"on_{g}"] = np.bincount(near[sel], m[sel], len(beats) - 1)
    seq["on_strength"] = np.bincount(near[sel], o.strength[sel], len(beats) - 1)
    idx = np.clip((beats[:-1] / o.hop_ms).astype(int), 0, len(o.env) - 1)
    seq["env"] = o.env[idx].astype(np.float64)
    fr = np.clip((beats[:-1] / h.bass_hop_ms).astype(int), 2, len(h.bass_energy) - 3)
    E = h.bass_energy
    seq["bass_rise"] = np.array([E[i:i + 2].max() - E[i - 2:i].mean() for i in fr], np.float64)
    return seq


def periodicity(x: np.ndarray) -> dict:
    """Phase contrast (best phase's mean, in std of x) and autocorrelation per period."""
    x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    f = {}
    for P in PERIODS:
        n = len(z) // P * P
        f[f"c{P}"] = float(z[:n].reshape(-1, P).mean(0).max()) if n >= 3 * P else 0.0
        f[f"r{P}"] = float(np.mean(z[P:] * z[:-P])) if len(z) > P else 0.0
    return f


def features(o, h, beats: np.ndarray) -> dict:
    """Periodicity of every per-beat signal, plus each 3-ish period minus its 4-ish twin."""
    out = {}
    for name, x in sequences(o, h, beats).items():
        p = periodicity(x)
        for k, v in p.items():
            out[f"{name}_{k}"] = v
        for kind in ("c", "r"):
            for a, b in PAIRS:
                out[f"{name}_{kind}{a}-{b}"] = p[f"{kind}{a}"] - p[f"{kind}{b}"]
    return out


def song_meter(o, h, segs, t_end: float, offset_ms: float) -> float:
    """p(bars of 3) from the longest line's grid (osu! clock lines, offset_ms to the onset clock)."""
    from .beatprob import load_model, predict_proba

    if h is None or not segs or not METER_TREES.exists():
        return 0.0
    segs = sorted(segs, key=lambda s: s.time)
    ends = [s.time for s in segs[1:]] + [t_end]
    i = int(np.argmax([e - s.time for s, e in zip(segs, ends)]))
    s = segs[i]
    lo = max(s.time, o.times[0] + offset_ms)
    first = s.time + np.ceil((lo - s.time) / s.beat_length) * s.beat_length
    n = int((ends[i] - first) // s.beat_length)
    if n < MIN_BEATS:
        return 0.0
    beats = first + np.arange(n) * s.beat_length - offset_ms
    m = load_model(METER_TREES)
    f = features(o, h, beats)
    x = np.array([[f[str(k)] for k in m["names"]]], float)
    return float(predict_proba(m, x)[0])


def bar3_shift(o, h, s, end: float, offset_ms: float) -> int:
    """Beats (0, 1 or 2) after s.time where bars of 3 start, from PHASE_KEYS on the line's own span."""
    L = s.beat_length
    lo, hi = s.time - offset_ms, end - offset_ms
    F = np.array([[bar_features(o, lo + j * L, L, lo, hi, meter=3, ch=h).get(k, 0.0) for k in PHASE_KEYS]
                  for j in range(3)])
    F = np.nan_to_num(F)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return int(np.argmax(Z.sum(1)))


def apply_meter(o, h, segs, t_end: float, offset_ms: float, info: dict | None = None) -> list:
    """Bars of 3 when the song is in 3 (METER_P): every line gets meter 3, and each line of
    PHASE_MIN_BEATS or more moves along its own grid to "1" (the first one back, so the song
    start stays covered; the others to the nearest "1", after the line before)."""
    from .grid import Segment

    p = song_meter(o, h, segs, t_end, offset_ms)
    if info is not None:
        info["meter_p"] = p
    if p < METER_P:
        return segs
    segs = sorted(segs, key=lambda s: s.time)
    ends = [s.time for s in segs[1:]] + [t_end]
    out = []
    for i, (s, end) in enumerate(zip(segs, ends)):
        L, t = s.beat_length, s.time
        if (end - t) / L >= PHASE_MIN_BEATS:
            j = bar3_shift(o, h, s, end, offset_ms)
            if i == 0:
                t -= ((3 - j) % 3) * L
            else:
                shift = j if j < 2 else -1
                if out[-1].time + L < t + shift * L < end - L:
                    t += shift * L
        out.append(Segment(float(t), L, 3))
    if info is not None:
        info["meter"] = 3
    return out
