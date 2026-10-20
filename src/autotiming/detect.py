"""BPM / offset detection.

Pipeline: decode like osu! -> multi-band spectral-flux onsets (2 ms hop) ->
tempo candidates from autocorrelation -> for each candidate, snap onsets to the
beat / 1/2 / 1/4 grid and fit time = offset + k * beat_length by robust weighted
least squares over the whole song. Hundreds of onsets spread over minutes pin the
beat length far tighter than any frame-based tempo estimate.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

from .decode import decode_bass
from .grid import Segment

HOP_MS = 2.0
N_FFT = 1024
N_BANDS = 64
FLUX_LAG = 3  # frames: compare against 6 ms ago, robust to slow attacks / vibrato
ONSET_BIAS_MS = 0.0  # calibrated on synthetic audio (see bench)
# Ranked red lines sit ~20 ms before the physical attack in BASS-decoded audio,
# for mp3 and ogg alike (bench/beat_profile.py, bench/bias.py). That's osu!'s
# clock convention, so it's applied to the output, not to onset detection.
OSU_OFFSET_MS = -20.4
# centre of the tempo prior in the octave model (ranked BPM median is 170)
OCTAVE_CENTER_BPM = 175.0


@dataclass
class Onsets:
    times: np.ndarray  # ms
    strength: np.ndarray
    low: np.ndarray  # share of the onset's energy below ~200 Hz (kick/bass)
    env: np.ndarray  # full flux envelope, one value per hop_ms
    env_low: np.ndarray
    sub: np.ndarray  # share below ~90 Hz: kick fundamentals, the most reliable beat cue
    hop_ms: float  # exact frame step (HOP_MS rounded to whole samples)
    profile: np.ndarray | None = None  # per-onset flux in each of the N_BANDS bands (what kind of hit it is)


def _bands(sr: int, n_fft: int) -> np.ndarray:
    """Bin edges of log-spaced bands, 30 Hz .. 16 kHz."""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(30, min(16000, sr / 2 - 1), N_BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    return np.maximum.accumulate(np.maximum(idx, np.arange(len(idx)) + 1))


def onsets(pcm: np.ndarray, sr: int, n_fft: int = N_FFT, hop_target_ms: float = HOP_MS) -> Onsets:
    hop = int(round(sr * hop_target_ms / 1000))
    hop_ms = hop * 1000 / sr
    flux_lag = max(1, int(round(FLUX_LAG * HOP_MS / hop_ms)))  # same 6 ms whatever the hop
    # frames are centred on k * hop; the signal is zero-padded by n_fft // 2 in front and n_fft behind,
    # but only chunk by chunk: a padded copy of the whole song is another 100+ MB on long tracks
    pcm = np.asarray(pcm, np.float32)
    front = n_fft // 2
    n_frames = (len(pcm) + front + n_fft - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    edges = _bands(sr, n_fft)
    low_bands = np.searchsorted(np.geomspace(30, 16000, N_BANDS + 1), 200)
    sub_bands = np.searchsorted(np.geomspace(30, 16000, N_BANDS + 1), 90)
    bands = np.empty((n_frames, N_BANDS), np.float32)
    step = 2048  # frames per chunk: ~70 MB of temporaries, flat whatever the song length
    for s in range(0, n_frames, step):
        e = min(n_frames, s + step)
        a0 = s * hop - front  # first sample of this chunk in the signal's own indexing
        seg = np.zeros((e - 1 - s) * hop + n_fft, np.float32)
        lo, hi = max(0, a0), min(len(pcm), a0 + len(seg))
        if hi > lo:
            seg[lo - a0:hi - a0] = pcm[lo:hi]
        idx = (np.arange(e - s) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(seg[idx] * win, axis=1))
        cs = np.concatenate([np.zeros((e - s, 1), np.float32), np.cumsum(mag, axis=1, dtype=np.float32)], axis=1)
        bands[s:e] = (cs[:, edges[1:]] - cs[:, edges[:-1]]) / np.diff(edges)
    logb = np.log1p(np.multiply(bands, 1000, out=bands), out=bands)  # in place: one frames x bands array less
    diff = np.zeros_like(logb)
    for a in range(flux_lag, len(logb), 65536):
        b = min(len(logb), a + 65536)
        diff[a:b] = np.maximum(0, logb[a:b] - logb[a - flux_lag:b - flux_lag])
    env = diff.sum(axis=1)
    env_low = diff[:, :low_bands].sum(axis=1)
    env_sub = diff[:, :sub_bands].sum(axis=1)

    # peaks above a local adaptive threshold, at least 25 ms apart
    thresh = median_filter(env, size=int(400 / hop_ms)) + 0.3 * uniform_filter1d(env, int(2000 / hop_ms)) + 1e-6
    is_peak = (env == maximum_filter1d(env, int(25 / hop_ms) * 2 + 1)) & (env > thresh)
    k = np.flatnonzero(is_peak)
    k = k[(k > 0) & (k < len(env) - 1)]
    # parabolic interpolation for sub-frame peak position
    a, b, c = env[k - 1], env[k], env[k + 1]
    frac = np.where(a - 2 * b + c != 0, 0.5 * (a - c) / (a - 2 * b + c), 0).clip(-0.5, 0.5)
    times = (k + frac) * hop_ms - ONSET_BIAS_MS
    strength = env[k] - thresh[k]
    low = env_low[k] / (env[k] + 1e-9)
    sub = env_sub[k] / (env[k] + 1e-9)
    return Onsets(times, strength, low, env, env_low, sub, hop_ms, diff[k])


def tempo_candidates(env: np.ndarray, hop_ms: float, n: int = 6, lo_bpm=60, hi_bpm=400) -> list[float]:
    """Beat lengths (ms) whose autocorrelation of the onset envelope peaks."""
    x = env - uniform_filter1d(env, int(1000 / hop_ms))
    m = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, m)
    acf = np.fft.irfft(f * np.conj(f))[: len(x)]
    acf /= acf[0] + 1e-9
    lags = np.arange(int(60000 / hi_bpm / hop_ms), int(60000 / lo_bpm / hop_ms))
    # a real beat period also lines up at 2x and 4x
    score = acf[lags] + 0.5 * acf[np.minimum(2 * lags, len(acf) - 1)] + 0.25 * acf[np.minimum(4 * lags, len(acf) - 1)]
    peaks = np.flatnonzero((score == maximum_filter1d(score, 9)) & (score > 0))
    peaks = peaks[np.argsort(score[peaks])[::-1]][:n]
    return [float(lags[p] * hop_ms) for p in peaks]


HARMONICS = ((1, 1.0), (2, 0.5), (4, 0.25))  # beat, 1/2, 1/4 lines


def _phase(times, weights, L, sigma=8.0, res=0.5):
    """Beat phase maximizing kernel-weighted onset mass on the beats.

    Onset phases go into a res-ms histogram, which is circularly convolved with
    the Gaussian kernel by FFT: O(n + bins log bins) instead of n x bins.
    """
    nb = max(1, int(np.ceil(L / res)))
    hist = np.bincount((np.floor((times % L) / res).astype(int)) % nb, weights=weights, minlength=nb)
    k = np.arange(nb)
    d = np.minimum(k, nb - k) * res
    score = np.fft.irfft(np.fft.rfft(hist) * np.fft.rfft(np.exp(-0.5 * (d / sigma) ** 2)), nb)
    return float((np.argmax(score) + 0.5) * res)


def coherence(times, weights, Ls) -> np.ndarray:
    """Sum over HARMONICS of the weighted resultant length of onset phases, per beat length."""
    return _coherence(np.ascontiguousarray(times, np.float64), np.ascontiguousarray(weights, np.float64),
                      np.ascontiguousarray(Ls, np.float64))


@njit(cache=True)
def _coherence(times, weights, Ls):
    """One cos/sin per (onset, L); the 1/2 and 1/4 harmonics are its square and fourth
    power. Compiled, it runs without the onset x L temporaries numpy would allocate."""
    out = np.empty(len(Ls))
    for j in range(len(Ls)):
        k = 2 * np.pi / Ls[j]
        r1 = i1 = r2 = i2 = r4 = i4 = 0.0
        for n in range(len(times)):
            a = k * times[n]
            c, s = np.cos(a), np.sin(a)
            c2, s2 = c * c - s * s, 2 * c * s
            c4, s4 = c2 * c2 - s2 * s2, 2 * c2 * s2
            w = weights[n]
            r1 += w * c; i1 += w * s
            r2 += w * c2; i2 += w * s2
            r4 += w * c4; i4 += w * s4
        out[j] = np.sqrt(r1 * r1 + i1 * i1) + 0.5 * np.sqrt(r2 * r2 + i2 * i2) + 0.25 * np.sqrt(r4 * r4 + i4 * i4)
    return out


def refine_period(times, weights, L0, span=0.015):
    """Beat length near L0 at which all onsets of the song are most coherent.

    Coherence = weighted resultant length of onset phases at the beat and its
    1/2, 1/4 subdivisions. Its peak is ~1/duration wide, so stepping finely over
    ±1.5% lands within a small fraction of a millisecond per beat.
    """
    T = max(times[-1] - times[0], 1.0)
    step = max(L0 * L0 / (40 * T), L0 * 2e-6)
    Ls = np.arange(L0 * (1 - span), L0 * (1 + span), step)
    if len(Ls) > 4000:
        Ls = np.linspace(L0 * (1 - span), L0 * (1 + span), 4000)
    best_L, best_c = L0, -1.0
    for s in range(0, len(Ls), 256):
        chunk = Ls[s:s + 256]
        c = coherence(times, weights, chunk)
        i = int(np.argmax(c))
        if c[i] > best_c:
            best_L, best_c = float(chunk[i]), float(c[i])
    return best_L


def grid_excess(times, weights, phi, L, subdiv=4, tol=8.0):
    """Onset mass within tol of the grid, minus what a random grid of the same density would catch."""
    q = L / subdiv
    d = np.abs((times - phi + q / 2) % q - q / 2)
    hit = float((weights * (d < tol)).sum() / (weights.sum() + 1e-9))
    return hit - min(1.0, 2 * tol / q)


def weights(o: Onsets) -> np.ndarray:
    return o.strength * (1 + 2 * o.low)  # kick/bass hits carry the beat


def beat_phase(t, w_low, w_all, phi, L, subdiv=4):
    """Which of the `subdiv` grid shifts is the beat.

    The loudest position is often an offbeat (snare + chord on the "and"), so
    loudness alone picks the wrong phase on syncopated music. Kick/bass onsets
    mark the beat far more reliably; overall mass only breaks ties.
    """
    q = L / subdiv
    best, best_s = phi, -np.inf
    for j in range(subdiv):
        p = phi + j * q
        d = np.abs((t - p + L / 2) % L - L / 2)
        hit = np.exp(-0.5 * (d / 6) ** 2)
        s = (w_low * hit).sum() + 0.3 * (w_all * hit).sum()
        if s > best_s:
            best, best_s = p, s
    return best


def fit_grid(t, w, L0: float, subdiv: int = 4, span=0.015, w_low=None) -> tuple[float, float, float]:
    """Refine (offset, beat_length) around a rough beat length. Returns (offset, L, score)."""
    L = refine_period(t, w, L0, span)
    phi = _phase(t, w, L)
    level_w = np.array([1.0, 0.25, 0.5, 0.25])  # beat, 1/4, 1/2, 3/4
    for tol in (20, 12, 8, 6):
        q = L / subdiv
        m = np.round((t - phi) / q)
        resid = t - (phi + m * q)
        keep = np.abs(resid) < min(tol, q / 3)
        if keep.sum() < 8:
            break
        mm, tt = m[keep], t[keep]
        ww = w[keep] * level_w[(mm % subdiv).astype(int)] / (1 + (resid[keep] / 4) ** 2)
        A = np.stack([np.ones_like(mm), mm], axis=1)
        sw = np.sqrt(ww)
        (phi, q), *_ = np.linalg.lstsq(A * sw[:, None], tt * sw, rcond=None)
        L = q * subdiv
    if w_low is not None:
        phi = beat_phase(t, w_low, w, phi, L, subdiv)
    return float(phi % L), float(L), grid_excess(t, w, phi, L, subdiv)


WIN_MS, WIN_HOP_MS = 10000.0, 2500.0
L_STEP = 1.001  # log spacing of the local tempo search (window peaks are ~5% wide)
JUMP_COST = 0.15  # Viterbi penalty for a tempo jump, in units of normalized coherence


def local_tempo_path(t, w, L_lo=60000 / 400, L_hi=60000 / 60, win_ms=WIN_MS, hop_ms=WIN_HOP_MS, jump_cost=None,
                     prior_L=None, prior_w=0.0, drift=2):
    """Most likely beat length per window: free to drift by up to `drift` grid steps (0.1% each) per window, penalized to jump."""
    Ls = L_lo * L_STEP ** np.arange(int(np.log(L_hi / L_lo) / np.log(L_STEP)) + 1)
    centers = np.arange(t[0] + win_ms / 2, max(t[-1] - win_ms / 2, t[0] + win_ms / 2) + 1, hop_ms)
    E = np.zeros((len(centers), len(Ls)))
    for i, c in enumerate(centers):
        sel = np.abs(t - c) < win_ms / 2
        if sel.sum() < 6:
            continue
        tt, ww = t[sel], w[sel]
        E[i] = coherence(tt, ww, Ls) / (ww.sum() * sum(a for _, a in HARMONICS))
    if prior_L is not None:  # soft preference for the main tempo, against x1.5 / x0.67 slides
        E -= prior_w * np.log(Ls / prior_L)[None, :] ** 2
    # Viterbi: moving to a neighbouring L (drift) is free, anything else costs JUMP_COST
    jump_cost = JUMP_COST if jump_cost is None else jump_cost
    score = E[0].copy()
    back = np.zeros(E.shape, int)
    n = len(score)
    for i in range(1, len(E)):
        # best predecessor among the 5 neighbouring beat lengths (drift), vectorized
        padded = np.pad(score, drift, constant_values=-np.inf)
        win = np.lib.stride_tricks.sliding_window_view(padded, 2 * drift + 1)
        near_arg = np.arange(n) - drift + win.argmax(axis=1)
        near = win.max(axis=1)
        jump_arg = int(np.argmax(score))
        jump = score[jump_arg] - jump_cost
        use_jump = jump > near
        back[i] = np.where(use_jump, jump_arg, near_arg)
        score = np.where(use_jump, jump, near) + E[i]
    path = np.zeros(len(E), int)
    path[-1] = int(np.argmax(score))
    for i in range(len(E) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return centers, Ls[path]


def _segment_windows(centers, Lpath, tol=0.004):
    """Group consecutive windows whose beat length stays within tol of the group's median."""
    groups = [[0]]
    for i in range(1, len(centers)):
        ref = np.median(Lpath[groups[-1]])
        if abs(np.log(Lpath[i] / ref)) < tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _boundary(t, w, a: Segment, b: Segment, lo, hi):
    """Time in [lo, hi] (on a beat of b) where onsets switch from grid a to grid b."""
    ks = np.arange(np.ceil((lo - b.time) / b.beat_length), np.floor((hi - b.time) / b.beat_length) + 1)
    if len(ks) == 0:
        return b.time + np.round(((lo + hi) / 2 - b.time) / b.beat_length) * b.beat_length

    def fit(seg):
        q = seg.beat_length / 4
        d = np.abs((t - seg.time + q / 2) % q - q / 2)
        return w * np.exp(-0.5 * (d / 6) ** 2)

    fa, fb = fit(a), fit(b)
    sel = (t >= lo - WIN_MS) & (t <= hi + WIN_MS)
    best, best_s = None, -np.inf
    for k in ks:
        tb = b.time + k * b.beat_length
        s = fa[sel & (t < tb - 2)].sum() + fb[sel & (t >= tb - 2)].sum()
        if s > best_s:
            best, best_s = tb, s
    return best


def _detect_raw(o: Onsets) -> list[Segment]:
    t, w = o.times, weights(o)
    wl = o.strength * (o.sub + 0.5 * o.low)
    # global pass: the beat octave of the strongest candidate anchors the local search
    best = None
    # no autocorrelation peak at all (near-silent or arrhythmic audio): fall back to the prior's tempo
    for L0 in tempo_candidates(o.env, o.hop_ms) or [60000 / OCTAVE_CENTER_BPM]:
        phi, L, score = fit_grid(t, w, L0, w_low=wl)
        if best is None or score > best[2]:
            best = (phi, L, score)
    phi, L, _ = best
    if t[-1] - t[0] < 2 * WIN_MS:
        return [Segment(phi, L)]
    centers, Lpath = local_tempo_path(t, w, L / 1.9, L * 1.9)
    groups = _segment_windows(centers, Lpath)
    if len(groups) == 1:
        return [Segment(phi, L)]
    segs, spans = [], []
    for g in groups:
        lo, hi = centers[g[0]] - WIN_MS / 2, centers[g[-1]] + WIN_MS / 2
        sel = (t >= lo) & (t <= hi)
        if sel.sum() < 8:
            continue
        p, l, _ = fit_grid(t[sel], w[sel], float(np.median(Lpath[g])), span=0.006, w_low=wl[sel])
        segs.append(Segment(p, l))
        spans.append((lo, hi))
    if not segs:
        return [Segment(phi, L)]
    # merge neighbours that ended up on the same grid
    merged, mspans = [segs[0]], [spans[0]]
    for s, sp in zip(segs[1:], spans[1:]):
        prev = merged[-1]
        same_L = abs(np.log(s.beat_length / prev.beat_length)) < 0.0015
        dphi = (s.time - prev.time) % prev.beat_length
        if same_L and min(dphi, prev.beat_length - dphi) < 8:
            sel = (t >= mspans[-1][0]) & (t <= sp[1])
            p, l, _ = fit_grid(t[sel], w[sel], prev.beat_length, span=0.002, w_low=wl[sel])
            merged[-1], mspans[-1] = Segment(p, l), (mspans[-1][0], sp[1])
        else:
            merged.append(s)
            mspans.append(sp)
    out = [merged[0]]
    for i in range(1, len(merged)):
        a, b = out[-1], merged[i]
        # the switch happens where the two analysed spans overlap
        lo, hi = mspans[i][0], mspans[i - 1][1]
        if hi < lo:
            lo, hi = hi, lo
        out.append(Segment(_boundary(t, w, a, b, lo - WIN_HOP_MS, hi + WIN_HOP_MS), b.beat_length))
    return prune_segments(t, w, wl, out, Segment(phi, L))


def piecewise_excess(t, w, segs: list[Segment], subdiv=4, tol=8.0) -> float:
    """grid_excess for a whole timing: each onset is judged against the red line it falls under."""
    starts = np.array([s.time for s in segs])
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(segs) - 1)
    ph = np.array([s.time for s in segs])[idx]
    q = np.array([s.beat_length for s in segs])[idx] / subdiv
    d = np.abs((t - ph + q / 2) % q - q / 2)
    return float((w * ((d < tol) - np.minimum(1.0, 2 * tol / q))).sum() / (w.sum() + 1e-9))


PRUNE_SLACK = 0.01  # a red line must explain at least this much more onset mass to stay


def prune_segments(t, w, wl, segs: list[Segment], global_seg: Segment) -> list[Segment]:
    """Drop red lines that don't earn their keep.

    Spurious tempo changes come from noisy windows; a real one makes the onsets
    clearly fit better. Greedily merge the neighbour pair whose merge costs the
    least, while that cost stays under PRUNE_SLACK.
    """
    if len(segs) > 1 and piecewise_excess(t, w, [global_seg]) >= piecewise_excess(t, w, segs) - PRUNE_SLACK:
        return [global_seg]
    memo = {}  # a merge of an unchanged neighbour pair is refit once, not on every pass

    def merged_fit(lo, hi, L):
        key = (round(lo, 3), round(hi, 3), round(L, 6))
        if key not in memo:
            sel = (t >= lo) & (t < hi)
            memo[key] = fit_grid(t[sel], w[sel], L, span=0.003, w_low=wl[sel])
        return memo[key]

    while len(segs) > 1:
        base = piecewise_excess(t, w, segs)
        best = None
        for i in range(len(segs) - 1):
            lo = segs[i].time if i > 0 else t[0]
            hi = segs[i + 2].time if i + 2 < len(segs) else t[-1] + 1
            sel = (t >= lo) & (t < hi)
            if sel.sum() < 8:
                cand_segs = segs[:i + 1] + segs[i + 2:]
            else:
                fits = [merged_fit(lo, hi, s.beat_length) for s in segs[i:i + 2]]
                p, l, _ = max(fits, key=lambda f: f[2])
                start = p if i == 0 else p + np.ceil((segs[i].time - p) / l - 0.5) * l
                cand_segs = segs[:i] + [Segment(start, l)] + segs[i + 2:]
            score = piecewise_excess(t, w, cand_segs)
            if best is None or score > best[0]:
                best = (score, cand_segs)
        if best[0] < base - PRUNE_SLACK:
            break
        segs = best[1]
    return segs


OCTAVE_EDGES = np.geomspace(30, 16000, N_BANDS + 1)
OCTAVE_GROUPS = {"sub": (0, 90), "low": (90, 250), "mid": (250, 2500), "high": (2500, 16000)}
# Logistic scores for "this candidate tempo is the one mappers pick", fitted on
# 733 ranked maps by bench/octave_exp.py (grouped 5-fold CV: 85.5% vs 81.6% for
# nearest-to-175-BPM). Only differences between candidates matter, so no intercept.
OCTAVE_COEF = {
    "density": -0.105367,
    "high_beat": -1.659233,
    "high_half": 10.106496,
    "high_odd": 1.782723,
    "high_quarter": -10.981509,
    "log2_rel": -0.328215,
    "low_beat": -0.625796,
    "low_half": 5.125806,
    "low_odd": -2.200679,
    "low_quarter": 2.720595,
    "mid_beat": -1.254305,
    "mid_half": -6.617605,
    "mid_odd": 6.763075,
    "mid_quarter": 7.192880,
    "sub_beat": -3.033623,
    "sub_half": -1.239800,
    "sub_odd": 0.520464,
    "sub_quarter": -3.064269,
    "log2_rel_sq": -2.354271,
}


def octave_features(o: Onsets, phi: float, L: float, lo: float, hi: float) -> dict:
    """Where each band group's onset mass lands on the grid (phi, L): beats, halves, quarters, odd beats."""
    sel = (o.times >= lo) & (o.times < hi)
    t, prof = o.times[sel], o.profile[sel]
    pos = ((t - phi) / L) % 1.0
    beat_idx = np.floor((t - phi) / L + 0.5).astype(int)
    tol = min(0.08, 12 / L)
    near = lambda p: np.abs((pos - p + 0.5) % 1.0 - 0.5) < tol
    on_b, on_h, on_q = near(0), near(0.5), near(0.25) | near(0.75)
    odd = (beat_idx % 2) == 1
    f = {"log2_rel": np.log2(60000 / L / OCTAVE_CENTER_BPM)}
    f["log2_rel_sq"] = f["log2_rel"] ** 2
    eps = 1e-6
    for g, (a, b) in OCTAVE_GROUPS.items():
        m = prof[:, (OCTAVE_EDGES[:-1] >= a) & (OCTAVE_EDGES[:-1] < b)].sum(axis=1)
        tot = m.sum() + eps
        f[f"{g}_beat"] = m[on_b].sum() / tot
        f[f"{g}_half"] = m[on_h].sum() / tot
        f[f"{g}_quarter"] = m[on_q].sum() / tot
        f[f"{g}_odd"] = m[on_b & odd].sum() / (m[on_b].sum() + eps)  # backbeat: snare on every other beat
    f["density"] = sel.sum() / max(1.0, (hi - lo) / L)
    return f


def octave_candidates(o: Onsets, phi: float, L: float, lo: float, hi: float) -> dict:
    """The main red line at x0.5 / x1 / x2 tempo: factor -> (phase, beat length)."""
    wl, w = o.strength * (o.sub + 0.5 * o.low), weights(o)
    sel = (o.times >= lo) & (o.times < hi)
    half_phi = beat_phase(o.times[sel], wl[sel], w[sel], phi, 2 * L, subdiv=2) if sel.any() else phi
    return {1.0: (phi, L), 2.0: (phi, L / 2), 0.5: (half_phi, 2 * L)}


def octave_score(f: dict) -> float:
    return sum(c * f[k] for k, c in OCTAVE_COEF.items())


# Boosted trees over the same features (bench/octave_trees_exp.py): 89.5% grouped CV on 733
# ranked maps vs 85.7% for the logistic scores above, which remain the fallback.
OCTAVE_TREES = Path(__file__).resolve().parent / "models" / "octave_trees.npz"
OCTAVE_KEYS = sorted(OCTAVE_COEF)
OCTAVE_KS = (0.5, 1.0, 2.0)


def octave_matrix(feats: dict) -> np.ndarray:
    """Model input for the x0.5 / x1 / x2 candidates: features, relative to the three's mean, rank among them."""
    from scipy.stats import rankdata

    F = np.nan_to_num(np.array([[feats[k][f] for f in OCTAVE_KEYS] for k in OCTAVE_KS]))
    return np.concatenate([F, F - F.mean(0, keepdims=True), rankdata(F, axis=0)], axis=1)


def octave_scores(feats: dict) -> dict:
    """Score per candidate factor (log-odds that mappers pick it)."""
    if not OCTAVE_TREES.exists():
        return {k: octave_score(feats[k]) for k in feats}
    from .beatprob import load_model, predict_proba

    p = np.clip(predict_proba(load_model(OCTAVE_TREES), octave_matrix(feats)), 1e-6, 1 - 1e-6)
    return {k: float(np.log(p[i] / (1 - p[i]))) for i, k in enumerate(OCTAVE_KS)}


OCT_PER_SECTION = os.environ.get("OCT_PER_SECTION", "1") == "1"
OCT_SECTION_MIN_BEATS = 32  # shorter sections follow the main section's octave
OCT_CONT_LAMBDA = float(os.environ.get("OCT_CONT_LAMBDA", "1.0"))  # score cost per |log2(bpm / main bpm)|


def choose_octave(o: Onsets, segs: list[Segment]) -> list[Segment]:
    """Rescale red lines by x0.5/x1/x2 so each tempo is what a mapper would pick.

    The main (longest) section decides by the octave model. Other long sections
    decide on their own, with a cost for straying from the main tempo: a slow
    middle section at 150 in a 200 BPM song would otherwise come out as 300
    (the double grid always fits the onsets as well), yet genuine jumps like
    154 -> 250 must survive.
    """
    ends = [segs[i + 1].time if i + 1 < len(segs) else o.times[-1] + 1 for i in range(len(segs))]
    main = max(range(len(segs)), key=lambda i: ends[i] - segs[i].time)

    def scored(i):
        lo = segs[i].time if i > 0 else o.times[0]
        cands = octave_candidates(o, segs[i].time, segs[i].beat_length, lo, ends[i])
        if o.profile is None:  # no band profile: fall back to the plain tempo prior
            return {k: -abs(np.log(segs[i].bpm * k / OCTAVE_CENTER_BPM)) for k in cands}, cands
        return octave_scores({k: octave_features(o, *cands[k], lo, ends[i]) for k in cands}), cands

    s_main, c_main = scored(main)
    f_main = max(s_main, key=s_main.get)
    main_bpm = segs[main].bpm * f_main
    out = []
    for i, s in enumerate(segs):
        if i == main:
            f, cands = f_main, c_main
        elif OCT_PER_SECTION and (ends[i] - s.time) / s.beat_length >= OCT_SECTION_MIN_BEATS:
            sc, cands = scored(i)
            f = max(sc, key=lambda k: sc[k] - OCT_CONT_LAMBDA * abs(np.log2(s.bpm * k / main_bpm)))
        else:
            f, cands = f_main, None
        if f == 1.0:
            out.append(s)
        elif f == 2.0:
            out.append(Segment(s.time, s.beat_length / 2, s.meter))
        else:  # halving: one of every two old beats becomes the beat; kick/bass decides which
            L = 2 * s.beat_length
            if cands is not None:
                phi = cands[0.5][0]
            else:
                wl, w = o.strength * (o.sub + 0.5 * o.low), weights(o)
                sel = (o.times >= s.time - 2 * s.beat_length) & (o.times < ends[i])
                phi = beat_phase(o.times[sel], wl[sel], w[sel], s.time, L, subdiv=2) if sel.any() else s.time
            out.append(Segment(phi if i == 0 else s.time + ((phi - s.time) % L), L, s.meter))
    return out


class NoRhythmError(ValueError):
    """The audio has (almost) no onsets: silence, a drone, a broken file."""


MIN_ONSETS = 16


def detect_onsets(o: Onsets) -> list[Segment]:
    if len(o.times) < MIN_ONSETS:
        raise NoRhythmError(f"only {len(o.times)} onsets found: nothing to time")
    segs = choose_octave(o, _detect_raw(o))
    return [Segment(s.time + OSU_OFFSET_MS, s.beat_length, s.meter) for s in segs]


FINE_N_FFT, FINE_HOP_MS, FINE_MATCH_MS = 512, 1.0, 8.0


def sharpen(o: Onsets, fine: Onsets) -> Onsets:
    """Swap each onset time for the short-window one found within FINE_MATCH_MS.

    The 23 ms analysis window is good for tempo and phase cues (bass needs it)
    but smears where a soft attack starts; an 11.6 ms window localizes it better.
    """
    if len(fine.times) == 0:
        return o
    j = np.clip(np.searchsorted(fine.times, o.times), 1, len(fine.times) - 1)
    near = np.where(np.abs(fine.times[j - 1] - o.times) < np.abs(fine.times[j] - o.times), j - 1, j)
    d = fine.times[near] - o.times
    times = np.where(np.abs(d) < FINE_MATCH_MS, fine.times[near], o.times)
    return Onsets(times, o.strength, o.low, o.env, o.env_low, o.sub, o.hop_ms, o.profile)


TRACK_MIN_MS = 20000.0  # shorter audio: the steady detector alone
_env = os.environ.get  # bench sweeps override these; defaults are the chosen values
TRACK_MIN_SHARE = float(_env("TRACK_MIN_SHARE", "0.0"))  # tracked share below this: keep steady lines
# accept the hybrid only if its onset fit beats the steady one by this (at 8 ms) and at all (at 4 ms);
# -1 disables the check (per-run checks inside hybrid() may make it unnecessary)
HYBRID_ACCEPT_GAIN = float(_env("HYBRID_ACCEPT_GAIN", "-1"))
LINE_COST = float(_env("LINE_COST", "30"))  # red-line compression of hybrid beats
LINE_SIGMA_MS = float(_env("LINE_SIGMA_MS", "3"))


def downbeat_anchor(o: Onsets, steady: list[Segment], ch) -> float | None:
    """A downbeat (onset clock) on the longest steady section, chosen once by harmonic change."""
    if ch is None or not steady:
        return None
    from .downbeat import downbeat_phase
    from .grid import beats as grid_beats

    phys = sorted([Segment(s.time - OSU_OFFSET_MS, s.beat_length, s.meter) for s in steady], key=lambda s: s.time)
    ends = [s.time for s in phys[1:]] + [o.times[-1]]
    k = int(np.argmax([e - s.time for s, e in zip(phys, ends)]))
    b = grid_beats([phys[k]], max(phys[k].time, o.times[0]), ends[k])
    if len(b) < 12:
        return None
    return float(b[downbeat_phase(o, ch, b[0], phys[k].beat_length, b[0], b[-1])])


def combine(o: Onsets, steady: list[Segment], tracked, anchor: float | None = None) -> list[Segment]:
    """Final red lines from the steady lines (osu! clock) and tracked beats (onset clock).

    The hybrid keeps steady beats wherever a steady grid holds the onsets. If it
    never needed the tracker, the steady lines are returned untouched; otherwise
    the merged beats are compressed into red lines, which must also buy a better
    onset fit than the steady lines.
    """
    from .track import hybrid, lines

    if tracked is None or len(tracked.times) < 4:
        return steady
    phys = sorted([Segment(s.time - OSU_OFFSET_MS, s.beat_length, s.meter) for s in steady], key=lambda s: s.time)
    merged = hybrid(o, phys, tracked)
    if merged.tracked_share <= TRACK_MIN_SHARE:  # a steady song: keep its lines exactly
        return steady
    allowed = None
    if anchor is not None:  # new red lines only on downbeats, like a mapper places them
        # bars counted along the merged beats from the steady downbeat they share
        main = max(range(len(phys)), key=lambda i: (phys[i + 1].time if i + 1 < len(phys) else o.times[-1]) - phys[i].time)
        bar = 4 * phys[main].beat_length
        d = np.abs((merged.times - anchor + bar / 2) % bar - bar / 2)
        hits = np.flatnonzero(d < 10)
        if len(hits):
            allowed = (np.arange(len(merged.times)) - hits[0]) % 4 == 0
    segs = sorted(lines(o, merged, line_cost=LINE_COST, sigma_ms=LINE_SIGMA_MS, allowed=allowed), key=lambda s: s.time)
    segs = [steady_twin(s, phys) for s in segs]
    if HYBRID_ACCEPT_GAIN >= 0:
        t, w = o.times, weights(o)
        gain4 = piecewise_excess(t, w, segs, tol=4.0) - piecewise_excess(t, w, phys, tol=4.0)
        gain8 = piecewise_excess(t, w, segs, tol=8.0) - piecewise_excess(t, w, phys, tol=8.0)
        if gain4 <= 0 or gain8 <= HYBRID_ACCEPT_GAIN:
            return steady
    return [Segment(s.time + OSU_OFFSET_MS, s.beat_length, s.meter) for s in segs]


TWIN_TEMPO, TWIN_MS = float(_env("TWIN_TEMPO", "0.002")), 5.0


def steady_twin(s: Segment, phys: list[Segment]) -> Segment:
    """A hybrid line that is a steady section's grid (same tempo within 0.2%, on its beats) gets that
    grid's exact values: the steady fit over the whole section is the tighter one (184.997, not 185.031)."""
    starts = [p.time for p in phys]
    p = phys[max(0, int(np.searchsorted(starts, s.time + TWIN_MS, side="right")) - 1)]
    if abs(np.log(s.beat_length / p.beat_length)) > TWIN_TEMPO:
        return s
    t = p.time + np.round((s.time - p.time) / p.beat_length) * p.beat_length
    return Segment(float(t), p.beat_length, s.meter) if abs(t - s.time) <= TWIN_MS else s


ALIGN_PREV_W = float(_env("ALIGN_PREV_W", "0.5"))


def align_octave(secs: list[Segment], steady: list[Segment]) -> list[Segment]:
    """Scale each section (onset clock) by x0.5/x1/x2 to sit closest to the steady tempo at its start,
    and (weight ALIGN_PREV_W) to the section before it.

    The second term is for ritardandos the steady lines don't see: 165 -> 150 after
    a steady 220 must not become 165 -> 300, though 300 is nearer 220 than 150 is.
    """
    st = sorted(steady, key=lambda s: s.time)
    starts = np.array([s.time - OSU_OFFSET_MS for s in st])
    out = []
    for s in secs:
        ref = st[max(0, int(np.searchsorted(starts, s.time, side="right")) - 1)].beat_length
        prev = out[-1].beat_length if out else ref
        k = min((0.5, 1.0, 2.0), key=lambda k: abs(np.log(s.beat_length / k / ref))
                + ALIGN_PREV_W * abs(np.log(s.beat_length / k / prev)))
        out.append(Segment(s.time, s.beat_length / k, s.meter))
    return out


def main_beat_length(o: Onsets, steady: list[Segment]) -> float:
    starts = sorted(s.time - OSU_OFFSET_MS for s in steady)
    ends = starts[1:] + [o.times[-1]]
    longest = int(np.argmax([e - s for s, e in zip(starts, ends)]))
    return sorted(steady, key=lambda s: s.time)[longest].beat_length


PHASE_MIN_BEATS = 32  # shorter steady sections take the main section's phase decision
PHASE_UNSURE = (0.25, 0.75)  # phase model probabilities in here get the "check the and" note (half the errors, 4% of maps)


def steady_phase(o: Onsets, pcm, sr, ch, steady: list[Segment]) -> tuple[list[Segment], float | None]:
    """Move each steady section by half a beat where the phase trees say the beat is on its "and".

    Returns the sections and the main section's P(phase kept), None without the model.
    """
    if pcm is None or not steady or not PHASE_TREES.exists():
        return steady, None
    st = sorted(steady, key=lambda s: s.time)
    starts = [s.time - OSU_OFFSET_MS for s in st]
    ends = starts[1:] + [o.times[-1] + 1]
    main = int(np.argmax([e - a for a, e in zip(starts, ends)]))
    probs = {}
    for i, s in enumerate(st):
        if i == main or (ends[i] - starts[i]) / s.beat_length >= PHASE_MIN_BEATS:
            lo = starts[i] if i > 0 else o.times[0]
            probs[i] = phase_keep_prob(phase_features(o, pcm, sr, ch, starts[i], s.beat_length, lo, ends[i]))
    out = [Segment(s.time + s.beat_length / 2, s.beat_length, s.meter) if probs.get(i, probs[main]) < 0.5 else s
           for i, s in enumerate(st)]
    return out, probs[main]


def detect_full(o: Onsets, pcm=None, sr=None, downbeat_ms: float | None = None, info: dict | None = None) -> list[Segment]:
    """Steady detector, then the beat tracker wherever the tempo really moves.

    Steady red lines come first (precise and phase-consistent), each on the beat
    the phase trees pick. The tracker runs on beat probabilities around their
    main tempo and its beats are smoothed through all onsets; combine() decides
    where they replace the steady grid. info, if given, gets "phase_keep": the
    phase model's confidence for the main section.
    """
    from .beatprob import beat_prob
    from .track import fine_sections, track_beats

    ch = None
    if pcm is not None:
        from .downbeat import chroma

        ch = chroma(pcm, sr)
    steady, phase_keep = steady_phase(o, pcm, sr, ch, detect_onsets(o))
    if info is not None:
        info["phase_keep"] = phase_keep
    # a mapper who knows where "1" is (osu! time) overrides the guess
    anchor = downbeat_anchor(o, steady, ch) if downbeat_ms is None else downbeat_ms - OSU_OFFSET_MS
    if o.profile is None or o.times[-1] - o.times[0] < TRACK_MIN_MS:
        return connect_lines(snap_bpm(first_on_downbeat(steady, anchor, ch, o), o.times[-1] + OSU_OFFSET_MS))
    main_L = main_beat_length(o, steady)
    # the tracker expects each section at the octave the steady lines chose there
    secs = align_octave(fine_sections(o, main_L), steady)
    tracked = track_beats(o, secs, probs=beat_prob(o, main_L))
    return connect_lines(snap_bpm(first_on_downbeat(combine(o, steady, tracked, anchor), anchor, ch, o), o.times[-1] + OSU_OFFSET_MS))


CONNECT_MAX_MS = 15.0  # a line further off the previous grid is a real reset, not a fitting gap
CONNECT_MAX_MS_PER_BEAT = 2.0  # and closing the gap may not bend the section's grid more than this per beat


def connect_lines(segs: list[Segment]) -> list[Segment]:
    """Make every red line land exactly on the previous section's grid.

    osu! stores red lines in whole ms and restarts barlines at each one; if the
    previous grid reaches the next line 1-3 ms off, the editor shows a double
    barline and mappers call the lines "not connected". Times are rounded to ms
    and each section's beat length becomes (next line - this line) / beats
    between them, so the grids meet exactly. A line more than CONNECT_MAX_MS off
    the previous grid is a real phase reset and is left alone.
    """
    segs = sorted(segs, key=lambda s: s.time)
    m = len(segs)
    if m < 2:
        return [Segment(float(round(s.time)), s.beat_length, s.meter) for s in segs]
    # which joints to close, and the whole number of beats each section spans
    n = [0] * m
    joined = [False] * m
    for i in range(m - 1):
        x = (segs[i + 1].time - segs[i].time) / segs[i].beat_length
        gap = abs(x - round(x)) * segs[i].beat_length
        if gap < 1.0 and is_round(segs[i].bpm):
            continue  # snap_bpm already put the next line where this rounded grid lands: keep the round BPM
        if round(x) >= 1 and gap <= CONNECT_MAX_MS and gap / round(x) <= CONNECT_MAX_MS_PER_BEAT:
            n[i], joined[i] = int(round(x)), True
    # joint least squares per chain of joined sections: the knots move so both neighbours'
    # grids stay on their own beats (a long steady section barely moves; a short one bends)
    T = [s.time for s in segs]
    i = 0
    while i < m:
        j = i
        while j < m - 1 and joined[j]:
            j += 1
        if j > i:  # sections i..j-1 are joined; knots i..j-1 move, knot j (start of a section
            # whose own grid runs on unchecked) stays where it is
            rows, rhs = [], []
            for a in range(i, j):
                for k in range(n[a] + 1):  # section a's own beats, from its knot to the next
                    r = np.zeros(j - i)
                    target = segs[a].time + k * segs[a].beat_length
                    r[a - i] += 1 - k / n[a]
                    if a + 1 < j:
                        r[a + 1 - i] += k / n[a]
                    else:
                        target -= k / n[a] * segs[j].time
                    rows.append(r)
                    rhs.append(target)
            sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
            T[i:j] = list(sol)
        i = j + 1
    t = [float(round(x)) for x in T]  # osu! stores whole ms
    return [Segment(t[i], (t[i + 1] - t[i]) / n[i] if joined[i] else segs[i].beat_length, segs[i].meter)
            for i in range(m)]


SNAP_STEPS = (1.0, 0.5, 0.25, 0.1)  # BPM roundings mappers use, roundest first
SNAP_MAX_MS = float(os.environ.get("SNAP_MAX_MS", "5"))  # how far a rounded grid may drift from the fitted one
# (ranked: exact mapper BPM on 92% of single-BPM maps at 5 ms, coverage -1 point; a fitted 150.009 is noise)


def is_round(bpm: float) -> bool:
    return abs(bpm / SNAP_STEPS[-1] - round(bpm / SNAP_STEPS[-1])) < 1e-6


def snap_bpm(segs: list[Segment], t_end: float) -> list[Segment]:
    """Round each section's BPM to the roundest value the audio allows (150.009 -> 150).

    Songs are produced at round tempos, so the last decimals are measurement
    noise, and mappers would retype them anyway. A rounding is taken when the
    rounded grid stays within SNAP_MAX_MS of the fitted one over the section (a
    line whose position is still free, the first one or one after a phase reset,
    may move to centre that error). The next line then moves to where the rounded
    grid lands (whole ms), so sections stay connected without bending the round
    BPM, unless that moves a longer section by more than 2 ms: then this
    one keeps its fitted BPM and connect_lines, which runs after and closes all
    joints between unrounded lines, bends it instead.
    """
    segs = sorted(segs, key=lambda s: s.time)
    out, moved_to = [], None
    for i, s in enumerate(segs):
        free = moved_to is None
        st = s.time if free else moved_to
        nxt = segs[i + 1].time if i + 1 < len(segs) else None
        n = round((nxt - s.time) / s.beat_length) if nxt is not None else max(1, int((t_end - st) / s.beat_length))
        L, shift, moved_to = s.beat_length, 0.0, None
        for step in SNAP_STEPS:
            c = round(s.bpm / step) * step
            Lc = 60000.0 / c
            drift = n * (Lc - L)  # rounded grid minus fitted grid at the section's end
            if not free:
                if abs(drift) <= SNAP_MAX_MS:
                    L = Lc
                    break
                continue
            lo, hi = max(-SNAP_MAX_MS, -SNAP_MAX_MS - drift), min(SNAP_MAX_MS, SNAP_MAX_MS - drift)
            if lo > hi:
                continue
            L, shift = Lc, -drift / 2
            break
        if nxt is not None and is_round(60000.0 / L):
            land = round(st + shift) + n * L
            nn = (segs[i + 2].time if i + 2 < len(segs) else t_end) - nxt  # the next section's length
            if abs(land - nxt) > CONNECT_MAX_MS:
                pass  # a real reset: the next line keeps its own phase
            elif nn <= n * L or abs(land - nxt) <= 2.0:  # a longer grid may only move a hair
                moved_to = float(round(land))
            else:  # moving a longer grid that far costs more than bending this one: connect_lines bends it
                L, shift = s.beat_length, 0.0
        out.append(Segment(float(round(st + shift)), L, s.meter))
    return out


def first_on_downbeat(segs: list[Segment], anchor: float | None, ch, o: Onsets) -> list[Segment]:
    """Move the first red line to its section's first downbeat at or after 0 ms (same grid)."""
    if not segs:
        return segs
    from .downbeat import downbeat_phase
    from .grid import beats as grid_beats

    segs = sorted(segs, key=lambda s: s.time)
    s0 = segs[0]
    end = segs[1].time if len(segs) > 1 else None
    bar = s0.meter * s0.beat_length
    on_grid = lambda x: abs((x - s0.time + s0.beat_length / 2) % s0.beat_length - s0.beat_length / 2) < 10
    a = None if anchor is None else anchor + OSU_OFFSET_MS
    if a is not None and on_grid(a) and (end is None or a < end + bar):
        first = a  # the song's anchor downbeat lies on the first section's grid
    elif ch is not None and end is not None:
        b = grid_beats([s0], s0.time, end)
        if len(b) < 12:
            return segs
        bp = b - OSU_OFFSET_MS  # the audio's own clock, like onsets and chroma
        first = b[downbeat_phase(o, ch, bp[0], s0.beat_length, bp[0], bp[-1])]
    else:
        return segs
    first -= np.floor(first / bar) * bar  # earliest downbeat >= 0 on this grid
    if end is not None and first >= end:
        return segs
    return [Segment(float(first), s0.beat_length, s0.meter)] + segs[1:]


def detect(path, fine: bool | None = None, steady_only: bool = False) -> list[Segment]:
    if fine is None:
        fine = os.environ.get("AUTOTIMING_FINE", "0") == "1"
    pcm, sr = decode_bass(path)
    o = onsets(pcm, sr)
    if fine:
        o = sharpen(o, onsets(pcm, sr, FINE_N_FFT, FINE_HOP_MS))
    return detect_onsets(o) if steady_only else detect_full(o, pcm, sr)


# Second opinion on the half-beat phase: logistic model over spectral placement,
# broadband hits and loudness on the beat vs the "and", fitted on 725 ranked maps
# (bench/phase_exp.py, bench/loud_exp.py). It is not more accurate than the kick
# rule on its own (94% vs 93% CV, a tie on the user's nominated maps), but where
# the two disagree the phase really is in doubt: flagging those 8% of maps
# catches 2/3 of phase errors, leaving 2.5% silent ones instead of 7%.
PHASE_COEF_SPECTRAL = {
    "bb_beat": 0.847612,
    "bb_half": -0.847612,
    "high_beat": 1.745920,
    "high_half": -1.745920,
    "high_odd": -0.042046,
    "low_beat": 0.916184,
    "low_half": -0.916184,
    "low_odd": -0.503244,
    "mid_beat": 1.087576,
    "mid_half": -1.087576,
    "mid_odd": -0.285190,
    "sub_beat": 0.633981,
    "sub_half": -0.633981,
    "sub_odd": -0.501005,
}
PHASE_COEF_LOUD = {
    "loud_all": 0.494976,
    "loud_hi": 0.443093,
    "loud_lo": 0.063156,
}


def broadband_features(o: Onsets, phi: float, L: float, lo: float, hi: float) -> dict:
    """Kick/snare hits are broadband; offbeat bass is low-only and hats are high-only."""
    sel = (o.times >= lo) & (o.times < hi)
    t, prof = o.times[sel], o.profile[sel]
    g = np.stack([prof[:, (OCTAVE_EDGES[:-1] >= a) & (OCTAVE_EDGES[:-1] < b)].sum(axis=1) for a, b in OCTAVE_GROUPS.values()], 1)
    share = g / (g.sum(1, keepdims=True) + 1e-9)
    bb = np.minimum(share[:, 0] + share[:, 1], share[:, 2] + share[:, 3]) * 2 * g.sum(1)
    pos = ((t - phi) / L) % 1.0
    tol = min(0.08, 12 / L)
    near = lambda q: np.abs((pos - q + 0.5) % 1.0 - 0.5) < tol
    tot = bb.sum() + 1e-9
    return {"bb_beat": bb[near(0)].sum() / tot, "bb_half": bb[near(0.5)].sum() / tot,
            "bb_quarter": bb[near(0.25) | near(0.75)].sum() / tot}


def _energy_ms(pcm, sr, sos=None, chunk=1 << 20) -> np.ndarray:
    """Mean power per 1 ms of the (optionally filtered) signal, filtering in chunks with carried
    state: a whole-song float64 filter output and its cumsum cost ~1 GB on a long track."""
    from scipy.signal import sosfilt, sosfilt_zi

    per_ms = sr / 1000.0
    n_ms = int(len(pcm) / per_ms)
    acc = np.zeros(n_ms + 2)
    zi = None if sos is None else sosfilt_zi(sos) * 0.0
    for a in range(0, len(pcm), chunk):
        x = pcm[a:a + chunk].astype(np.float64)
        if sos is not None:
            x, zi = sosfilt(sos, x, zi=zi)
        ms = np.minimum(((a + np.arange(len(x))) / per_ms).astype(np.int64), n_ms + 1)
        lo_ms = ms[0]
        acc[lo_ms:ms[-1] + 1] += np.bincount(ms - lo_ms, weights=x * x)
    return acc[:n_ms] / per_ms


def loudness_on_beat(pcm, sr, phi: float, L: float, lo: float, hi: float) -> dict:
    """How many dB louder the 40 ms after each beat are than after each "and" (all / <150 Hz / >2 kHz)."""
    from scipy.signal import butter

    beats = phi + np.arange(np.ceil((lo - phi) / L), np.floor((hi - phi) / L)) * L

    def folded(env, times):
        idx = np.round(times[:, None] + np.arange(0, 40)[None, :]).astype(int)
        idx = idx[(idx[:, 0] > 0) & (idx[:, -1] < len(env))]
        return float(10 * np.log10(env[idx].max(axis=1).mean() + 1e-12)) if len(idx) else 0.0

    out = {}
    for name, sos in (("all", None),
                      ("lo", butter(4, 150, "lowpass", fs=sr, output="sos")),
                      ("hi", butter(4, 2000, "highpass", fs=sr, output="sos"))):
        e = _energy_ms(pcm, sr, sos)
        c = np.cumsum(np.concatenate([[0.0], e]))
        env = (c[5:] - c[:-5]) / 5  # 5 ms boxcar, as before but at 1 ms resolution
        out[f"loud_{name}"] = folded(env, beats) - folded(env, beats + L / 2)
    return out


def phase_opinions(o: Onsets, pcm, sr, phi: float, L: float, lo: float, hi: float) -> tuple[bool, bool]:
    """(kick rule keeps phi, model keeps phi) against the grid shifted by half a beat."""
    sel = (o.times >= lo) & (o.times < hi)
    t, wl, w = o.times[sel], (o.strength * (o.sub + 0.5 * o.low))[sel], weights(o)[sel]

    def kick(p):
        d = np.abs((t - p + L / 2) % L - L / 2)
        hit = np.exp(-0.5 * (d / 6) ** 2)
        return (wl * hit).sum() + 0.3 * (w * hit).sum()

    a = octave_features(o, phi, L, lo, hi) | broadband_features(o, phi, L, lo, hi)
    b = octave_features(o, phi + L / 2, L, lo, hi) | broadband_features(o, phi + L / 2, L, lo, hi)
    loud = loudness_on_beat(pcm, sr, phi, L, lo, hi)
    score = sum(c * (a[k] - b[k]) for k, c in PHASE_COEF_SPECTRAL.items()) + sum(c * loud[k] for k, c in PHASE_COEF_LOUD.items())
    return kick(phi) >= kick(phi + L / 2), score > 0


PHASE_TREES = Path(__file__).resolve().parent / "models" / "phase_trees.npz"


def phase_features(o: Onsets, pcm, sr, ch, phi: float, L: float, lo: float, hi: float) -> dict:
    """Evidence that the beat is at phi rather than phi + L/2: each feature is "on the beat" minus
    "on the and" (band-group onset mass, broadband hits, loudness, harmonic change, the loudest
    hits and the song's first hit, the kick rule)."""
    from .downbeat import _chroma_change

    h = phi + L / 2
    a = octave_features(o, phi, L, lo, hi) | broadband_features(o, phi, L, lo, hi)
    b = octave_features(o, h, L, lo, hi) | broadband_features(o, h, L, lo, hi)
    f = {f"d_{k}": a[k] - b[k] for k in a if k not in ("log2_rel", "log2_rel_sq", "density")}
    if pcm is not None:
        f |= loudness_on_beat(pcm, sr, phi, L, lo, hi)
    grid = lambda p: p + np.arange(np.ceil((lo - p) / L), np.floor((hi - p) / L)) * L
    f["d_chroma"] = _chroma_change(ch, grid(phi), L) - _chroma_change(ch, grid(h), L) if ch is not None else 0.0
    sel = (o.times >= lo) & (o.times < hi)
    t, s, wl, w = o.times[sel], o.strength[sel], (o.strength * (o.sub + 0.5 * o.low))[sel], weights(o)[sel]
    on = lambda p, x: np.abs((x - p + L / 2) % L - L / 2) < min(20.0, L / 8)
    top = s >= np.percentile(s, 98) if len(s) >= 50 else np.zeros(len(s), bool)
    f["d_top"] = float(on(phi, t[top]).mean() - on(h, t[top]).mean()) if top.any() else 0.0
    strong = np.flatnonzero(o.strength >= np.median(o.strength))
    t0 = o.times[strong[0]] if len(strong) else None
    f["d_first"] = 0.0 if t0 is None else float(on(phi, np.array([t0]))[0]) - float(on(h, np.array([t0]))[0])

    def kick(p):
        d = np.abs((t - p + L / 2) % L - L / 2)
        hit = np.exp(-0.5 * (d / 6) ** 2)
        return (wl * hit).sum() + 0.3 * (w * hit).sum()

    f["d_kick"] = float(np.log((kick(phi) + 1e-6) / (kick(h) + 1e-6)))
    return f


def phase_keep_prob(f: dict) -> float | None:
    """P(the beat is at phi, not phi + L/2) from the phase trees (bench/phase_trees_exp.py); a feature
    vector and its negation are both scored so the answer can't depend on which candidate came first."""
    if not PHASE_TREES.exists():
        return None
    from .beatprob import load_model, predict_proba

    m = load_model(PHASE_TREES)
    x = np.array([[f[k] for k in m["names"]]])
    p = predict_proba(m, np.nan_to_num(np.concatenate([x, -x])))
    return float((p[0] + 1 - p[1]) / 2)


@dataclass
class Timing:
    segments: list[Segment]  # red lines in osu! time
    phase_uncertain: bool = False  # kick rule and phase model disagree: check the "and" too
    alt_segments: list[Segment] | None = None  # the same timing shifted by half a beat


def analyze(path, downbeat_ms: float | None = None) -> Timing:
    """Full analysis for users: red lines plus an honest flag when the beat phase is in doubt."""
    pcm, sr = decode_bass(path)
    o = onsets(pcm, sr)
    info = {}
    segs = detect_full(o, pcm, sr, downbeat_ms, info)
    alt = [Segment(s.time + s.beat_length / 2, s.beat_length, s.meter) for s in segs]
    if info.get("phase_keep") is not None:
        p = info["phase_keep"]
        return Timing(segs, phase_uncertain=PHASE_UNSURE[0] < p < PHASE_UNSURE[1], alt_segments=alt)
    starts = [s.time - OSU_OFFSET_MS for s in segs]  # back to the audio's own clock, like onset times
    ends = starts[1:] + [o.times[-1] + 1]
    main = max(range(len(segs)), key=lambda i: ends[i] - starts[i])
    lo = starts[main] if main > 0 else o.times[0]
    rule_keeps, model_keeps = phase_opinions(o, pcm, sr, starts[main], segs[main].beat_length, lo, ends[main])
    return Timing(segs, phase_uncertain=rule_keeps != model_keeps, alt_segments=alt)
