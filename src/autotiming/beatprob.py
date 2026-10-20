"""Per-onset features for "is this onset a beat?".

The tracker and the phase choice used to follow raw onset strength, which can't
tell a kick on the beat from a loud stab on the "and". A model trained on
ranked timing (bench/beatprob_exp.py) turns these features into a beat
probability per onset.
"""

import numpy as np

from .detect import N_BANDS, Onsets

BAND_EDGES = np.geomspace(30, 16000, N_BANDS + 1)
GROUP_EDGES = [30, 60, 120, 250, 500, 1000, 2500, 6000, 16000]  # 8 timbre groups


def _window_stat(t, s, half_ms):
    """For each onset: max and mean strength of onsets within ±half_ms (itself included)."""
    lo, hi = np.searchsorted(t, t - half_ms), np.searchsorted(t, t + half_ms, side="right")
    cs = np.cumsum(np.r_[0, s])
    mean = (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)
    # running max over 10 ms bins
    from scipy.ndimage import maximum_filter1d

    b = ((t - t[0]) / 10).astype(int)
    grid = np.zeros(b[-1] + 1)
    np.maximum.at(grid, b, s)
    mx = maximum_filter1d(grid, size=2 * int(half_ms / 10) + 1)[b]
    return np.maximum(mx, s), mean


def _mass_at(t, s, offsets, tol=8.0):
    """Onset strength found at t + offset (±tol), per onset, per offset."""
    cs = np.cumsum(np.r_[0, s])
    out = []
    for off in offsets:
        lo, hi = np.searchsorted(t, t + off - tol), np.searchsorted(t, t + off + tol, side="right")
        out.append(cs[hi] - cs[lo])
    return np.stack(out, 1)


def onset_features(o: Onsets, L_est: float) -> tuple[np.ndarray, list[str]]:
    t, s = o.times, np.maximum(o.strength, 1e-6)
    prof = np.maximum(o.profile.astype(np.float64), 0)
    groups = np.stack([prof[:, (BAND_EDGES[:-1] >= a) & (BAND_EDGES[:-1] < b)].sum(1)
                       for a, b in zip(GROUP_EDGES[:-1], GROUP_EDGES[1:])], 1)
    tot = groups.sum(1, keepdims=True) + 1e-9
    share = groups / tot
    feats, names = [share], [f"band{i}" for i in range(share.shape[1])]
    ls = np.log(s)
    feats.append(ls[:, None]); names.append("log_strength")
    feats.append(np.log(tot)); names.append("log_flux")
    for w in (250.0, 1000.0, 4000.0):
        mx, mean = _window_stat(t, s, w)
        feats.append(np.stack([ls - np.log(mx), ls - np.log(mean + 1e-9)], 1))
        names += [f"rel_max_{int(w)}", f"rel_mean_{int(w)}"]
    d_prev = np.diff(t, prepend=t[0] - 1000)
    d_next = np.diff(t, append=t[-1] + 1000)
    feats.append(np.stack([np.log1p(d_prev), np.log1p(d_next), np.log1p(d_prev) / np.log1p(L_est),
                           np.log1p(d_next) / np.log1p(L_est)], 1))
    names += ["log_dprev", "log_dnext", "dprev_rel", "dnext_rel"]
    # rhythmic context at the estimated beat length: beats echo at ±1, ±2 beats
    offs = [-2 * L_est, -L_est, -L_est / 2, L_est / 2, L_est, 2 * L_est, -L_est / 4, L_est / 4]
    m = _mass_at(t, s, offs)
    feats.append(np.log1p(m / s[:, None])); names += ["ctx_m2", "ctx_m1", "ctx_mh", "ctx_ph", "ctx_p1", "ctx_p2", "ctx_mq", "ctx_pq"]
    # the same echoes, but kick-weighted (low groups)
    sl = s * (share[:, 0] + share[:, 1] + share[:, 2])
    ml = _mass_at(t, sl, [-L_est, L_est, -L_est / 2, L_est / 2])
    feats.append(np.log1p(ml / (sl[:, None] + 1e-6))); names += ["kctx_m1", "kctx_p1", "kctx_mh", "kctx_ph"]
    feats.append(np.full((len(t), 1), np.log(60000 / L_est))); names.append("log_bpm")
    return np.concatenate(feats, 1).astype(np.float32), names
