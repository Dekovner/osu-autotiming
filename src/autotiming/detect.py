"""BPM / offset detection.

Pipeline: decode like osu! -> multi-band spectral-flux onsets (2 ms hop) ->
tempo candidates from autocorrelation -> for each candidate, snap onsets to the
beat / 1/2 / 1/4 grid and fit time = offset + k * beat_length by robust weighted
least squares over the whole song. Hundreds of onsets spread over minutes pin the
beat length far tighter than any frame-based tempo estimate.
"""

from dataclasses import dataclass

import numpy as np
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
# mappers' default tempo octave: ranked BPM median is 170; picking the x0.5/x1/x2
# closest to this matches ranked 86% of the time vs 57% for raw strongest tempo
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


def _bands(sr: int) -> np.ndarray:
    """Bin edges of log-spaced bands, 30 Hz .. 16 kHz."""
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    edges = np.geomspace(30, min(16000, sr / 2 - 1), N_BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    return np.maximum.accumulate(np.maximum(idx, np.arange(len(idx)) + 1))


def onsets(pcm: np.ndarray, sr: int) -> Onsets:
    hop = int(round(sr * HOP_MS / 1000))
    hop_ms = hop * 1000 / sr
    pad = np.concatenate([np.zeros(N_FFT // 2, np.float32), pcm, np.zeros(N_FFT, np.float32)])
    n_frames = (len(pad) - N_FFT) // hop
    win = np.hanning(N_FFT).astype(np.float32)
    edges = _bands(sr)
    low_bands = np.searchsorted(np.geomspace(30, 16000, N_BANDS + 1), 200)
    sub_bands = np.searchsorted(np.geomspace(30, 16000, N_BANDS + 1), 90)
    bands = np.empty((n_frames, N_BANDS), np.float32)
    step = 8192  # frames per chunk keeps memory flat on long songs
    for s in range(0, n_frames, step):
        e = min(n_frames, s + step)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(N_FFT)[None, :]
        mag = np.abs(np.fft.rfft(pad[idx] * win, axis=1))
        cs = np.concatenate([np.zeros((e - s, 1), np.float32), np.cumsum(mag, axis=1, dtype=np.float32)], axis=1)
        bands[s:e] = (cs[:, edges[1:]] - cs[:, edges[:-1]]) / np.diff(edges)
    logb = np.log1p(1000 * bands)
    diff = np.zeros_like(logb)
    diff[FLUX_LAG:] = np.maximum(0, logb[FLUX_LAG:] - logb[:-FLUX_LAG])
    env = diff.sum(axis=1)
    env_low = diff[:, :low_bands].sum(axis=1)
    env_sub = diff[:, :sub_bands].sum(axis=1)

    # peaks above a local adaptive threshold, at least 25 ms apart
    thresh = median_filter(env, size=int(400 / HOP_MS)) + 0.3 * uniform_filter1d(env, int(2000 / HOP_MS)) + 1e-6
    is_peak = (env == maximum_filter1d(env, int(25 / HOP_MS) * 2 + 1)) & (env > thresh)
    k = np.flatnonzero(is_peak)
    k = k[(k > 0) & (k < len(env) - 1)]
    # parabolic interpolation for sub-frame peak position
    a, b, c = env[k - 1], env[k], env[k + 1]
    frac = np.where(a - 2 * b + c != 0, 0.5 * (a - c) / (a - 2 * b + c), 0).clip(-0.5, 0.5)
    times = (k + frac) * hop_ms - ONSET_BIAS_MS
    strength = env[k] - thresh[k]
    low = env_low[k] / (env[k] + 1e-9)
    sub = env_sub[k] / (env[k] + 1e-9)
    return Onsets(times, strength, low, env, env_low, sub, hop_ms)


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


def _phase(times, weights, L, sigma=8.0):
    """Beat phase maximizing kernel-weighted onset mass on the beats."""
    phis = np.arange(0, L, 0.5)
    d = (times[None, :] - phis[:, None]) % L
    d = np.minimum(d, L - d)
    score = (weights[None, :] * np.exp(-0.5 * (d / sigma) ** 2)).sum(axis=1)
    return float(phis[np.argmax(score)])


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
        ph = 2j * np.pi * times[:, None] / chunk[None, :]
        c = sum(a * np.abs((weights[:, None] * np.exp(h * ph)).sum(axis=0)) for h, a in HARMONICS)
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


def local_tempo_path(t, w, L_lo=60000 / 400, L_hi=60000 / 60):
    """Most likely beat length per window: free to drift slowly, penalized to jump."""
    Ls = L_lo * L_STEP ** np.arange(int(np.log(L_hi / L_lo) / np.log(L_STEP)) + 1)
    centers = np.arange(t[0] + WIN_MS / 2, max(t[-1] - WIN_MS / 2, t[0] + WIN_MS / 2) + 1, WIN_HOP_MS)
    E = np.zeros((len(centers), len(Ls)))
    for i, c in enumerate(centers):
        sel = np.abs(t - c) < WIN_MS / 2
        if sel.sum() < 6:
            continue
        tt, ww = t[sel], w[sel]
        ph = 2j * np.pi * tt[:, None] / Ls[None, :]
        coh = sum(a * np.abs((ww[:, None] * np.exp(h * ph)).sum(axis=0)) for h, a in HARMONICS)
        E[i] = coh / (ww.sum() * sum(a for _, a in HARMONICS))
    # Viterbi: moving to a neighbouring L (drift) is free, anything else costs JUMP_COST
    score = E[0].copy()
    back = np.zeros(E.shape, int)
    for i in range(1, len(E)):
        near = maximum_filter1d(score, 5)
        near_arg = np.array([j - 2 + int(np.argmax(score[max(0, j - 2):j + 3])) + (2 - min(j, 2)) for j in range(len(score))])
        jump_arg = int(np.argmax(score))
        jump = score[jump_arg] - JUMP_COST
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
    for L0 in tempo_candidates(o.env, o.hop_ms):
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
    return out


def choose_octave(o: Onsets, segs: list[Segment]) -> list[Segment]:
    """Rescale every red line by the same x0.5/x1/x2 so the main tempo looks like a mapper's pick."""
    main = max(range(len(segs)), key=lambda i: (segs[i + 1].time if i + 1 < len(segs) else o.times[-1]) - segs[i].time)
    bpm = segs[main].bpm
    f = min((0.5, 1.0, 2.0), key=lambda k: abs(np.log(bpm * k / OCTAVE_CENTER_BPM)))
    if f == 1.0:
        return segs
    if f == 2.0:
        return [Segment(s.time, s.beat_length / 2, s.meter) for s in segs]
    # halving: one of every two old beats becomes the beat; kick/bass decides which
    wl, w = o.strength * (o.sub + 0.5 * o.low), weights(o)
    out = []
    for i, s in enumerate(segs):
        end = segs[i + 1].time if i + 1 < len(segs) else o.times[-1] + 1
        sel = (o.times >= s.time - 2 * s.beat_length) & (o.times < end)
        L = 2 * s.beat_length
        phi = beat_phase(o.times[sel], wl[sel], w[sel], s.time, L, subdiv=2) if sel.any() else s.time
        out.append(Segment(phi if i == 0 else s.time + ((phi - s.time) % L), L, s.meter))
    return out


def detect_onsets(o: Onsets) -> list[Segment]:
    segs = choose_octave(o, _detect_raw(o))
    return [Segment(s.time + OSU_OFFSET_MS, s.beat_length, s.meter) for s in segs]


def detect(path) -> list[Segment]:
    pcm, sr = decode_bass(path)
    return detect_onsets(onsets(pcm, sr))
