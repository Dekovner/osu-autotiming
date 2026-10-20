"""Which beat of the bar is "1": features per candidate bar phase.

Mappers place red lines on downbeats. Cues for "1": the kick lands there, the
harmony changes there (new chord per bar), the bar's loudest hits start there.
"""

import numpy as np

from .detect import N_BANDS, Onsets

EDGES = np.geomspace(30, 16000, N_BANDS + 1)
GROUPS = {"sub": (0, 90), "low": (90, 250), "mid": (250, 2500), "high": (2500, 16000)}


def _near(t, grid_times, tol):
    j = np.clip(np.searchsorted(grid_times, t), 1, max(1, len(grid_times) - 1))
    return np.minimum(np.abs(t - grid_times[j - 1]), np.abs(t - grid_times[np.minimum(j, len(grid_times) - 1)])) < tol


def bar_features(o: Onsets, phi: float, L: float, lo: float, hi: float, meter: int = 4) -> dict:
    """Features for "bars start at phi" (phi on a beat, bars of `meter` beats), over [lo, hi)."""
    bar = meter * L
    starts = phi + np.arange(np.ceil((lo - phi) / bar), np.floor((hi - phi) / bar)) * bar
    sel = (o.times >= lo) & (o.times < hi)
    t, prof, s = o.times[sel], o.profile[sel].astype(np.float64), o.strength[sel]
    tol = min(25.0, L / 6)
    on_one = _near(t, starts, tol)
    beats = phi + np.arange(np.ceil((lo - phi) / L), np.floor((hi - phi) / L)) * L
    on_beat = _near(t, beats, tol)
    f = {}
    for g, (a, b) in GROUPS.items():
        m = prof[:, (EDGES[:-1] >= a) & (EDGES[:-1] < b)].sum(1)
        f[f"{g}_on_one"] = float(m[on_one].sum() / (m[on_beat].sum() + 1e-9))  # share of on-beat mass on "1"
    f["strength_on_one"] = float(s[on_one].sum() / (s[on_beat].sum() + 1e-9))
    f["count_on_one"] = float(on_one.sum() / (on_beat.sum() + 1e-9))
    # harmonic change: spectral profile difference between the bar before and after each "1"
    env_idx = lambda tm: np.clip((tm / o.hop_ms).astype(int), 0, len(o.env) - 1)
    if len(starts) > 2 and prof.size:
        mids = prof[:, (EDGES[:-1] >= 120) & (EDGES[:-1] < 2500)]
        diffs = []
        for x in starts[1:-1]:
            before = (t >= x - bar) & (t < x)
            after = (t >= x) & (t < x + bar)
            if before.sum() < 2 or after.sum() < 2:
                continue
            a_, b_ = mids[before].mean(0), mids[after].mean(0)
            diffs.append(np.abs(np.log1p(b_) - np.log1p(a_)).mean())
        f["harm_change"] = float(np.mean(diffs)) if diffs else 0.0
    else:
        f["harm_change"] = 0.0
    f["flux_on_one"] = float(o.env[env_idx(starts)].mean() / (o.env[env_idx(beats)].mean() + 1e-9)) if len(starts) else 0.0
    return f
