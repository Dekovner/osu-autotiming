"""Timing grids (a list of red lines) and how far apart two of them are."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    time: float  # ms where this red line starts
    beat_length: float  # ms
    meter: int = 4

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length


def beats(segments: list[Segment], start: float, end: float) -> np.ndarray:
    """All beat times (ms) of a piecewise grid within [start, end]."""
    segs = sorted(segments, key=lambda s: s.time)
    out = []
    for i, s in enumerate(segs):
        lo = start if i == 0 else max(start, s.time)  # first red line extends backwards
        hi = min(end, segs[i + 1].time) if i + 1 < len(segs) else end
        if hi < lo:
            continue
        k0 = np.ceil((lo - s.time) / s.beat_length - 1e-9)
        k1 = np.floor((hi - s.time) / s.beat_length - 1e-9) if i + 1 < len(segs) else np.floor((hi - s.time) / s.beat_length)
        if k1 >= k0:
            out.append(s.time + np.arange(k0, k1 + 1) * s.beat_length)
    return np.concatenate(out) if out else np.zeros(0)


def _nearest_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed distance from each a to the nearest b (a - b)."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    left, right = b[idx - 1] if len(b) > 1 else b[idx], b[idx]
    dl, dr = a - left, a - right
    return np.where(np.abs(dl) < np.abs(dr), dl, dr)


def subdivide(segments: list[Segment], k: int) -> list[Segment]:
    return [Segment(s.time, s.beat_length / k, s.meter) for s in segments]


def compare(pred: list[Segment], truth: list[Segment], start: float, end: float) -> dict:
    """Grid-vs-grid error over the playable span.

    Distances go both ways: pred->truth catches offset and drift, truth->pred
    catches missed tempo sections. A 2x/0.5x tempo pick is a convention question,
    not an accuracy one, so it's reported as `octave` and both grids are compared
    at the finer of the two densities.
    """
    tb, pb = beats(truth, start, end), beats(pred, start, end)
    if len(tb) < 2 or len(pb) < 2:
        return {"ok": False}
    ratio = float(np.median(np.diff(tb)) / np.median(np.diff(pb)))  # >1: pred faster
    octave = min((0.5, 1, 2), key=lambda k: abs(np.log(ratio / k)))
    if abs(np.log(ratio / octave)) > 0.02:
        octave = 0  # not a clean multiple: plain wrong tempo
    tf = subdivide(truth, 2) if octave == 2 else truth
    pf = subdivide(pred, 2) if octave == 0.5 else pred
    tb, pb = beats(tf, start, end), beats(pf, start, end)
    margin = 4 * max(s.beat_length for s in tf + pf)  # targets extend past the span: no edge artefacts
    p2t = _nearest_dist(pb, beats(tf, start - margin, end + margin))
    t2p = _nearest_dist(tb, beats(pf, start - margin, end + margin))
    err = np.abs(np.concatenate([p2t, t2p]))
    return {
        "ok": True,
        "octave": octave,
        "tempo_ratio": ratio,
        "median_ms": float(np.median(err)),
        "p95_ms": float(np.percentile(err, 95)),
        "max_ms": float(err.max()),
        "bias_ms": float(np.median(p2t)),  # + = predicted grid is late
    }
