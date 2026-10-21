"""Beat-level tracking for songs whose tempo moves: live drift, rubato, jumps.

1. A bar-pointer style HMM (state = beat interval x position inside the beat,
   tempo may change only on a beat, cheap for small changes, costly for big
   ones) finds the most likely beat sequence over 10 ms frames by Viterbi.
2. Every beat snaps to the onset that supports it, to sub-ms precision.
3. Optimal segmentation (DP with a cost per red line) turns the beats into the
   fewest red lines that keep the grid on the beats: one line for a steady
   song, one every few bars for drift, one per beat for rubato, as mappers do.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .grid import Segment

_env = __import__("os").environ.get  # tunables below can be overridden for bench sweeps
FPS = int(_env("TRACK_FPS", "100"))  # tracker frame rate; beats are refined to onsets after
OBS_LAMBDA = 16.0  # how much a beat frame prefers onset activity (madmom's observation lambda)
TRANSITION_LAMBDA = float(__import__("os").environ.get("TRACK_TLAMBDA", "60"))  # tempo-change stiffness: exp(-lambda * |ratio - 1|)
TEMPO_PRIOR = 100.0  # per-beat cost of log(interval / section interval)^2: 10% off ~ 1 nat, 30% ~ 7


@dataclass
class Beats:
    times: np.ndarray  # ms, in the onset clock
    supported: np.ndarray  # bool: an onset backs this beat (else it is interpolated)
    tracked_share: float = 1.0  # after hybrid(): share of steady beats replaced by tracked ones


# share of low-band (kick/bass) flux; 0.7 made tracked beats worse in every style
# (bench: 71% -> 67% of true beats within 10 ms on steady songs), so plain flux
LOW_MIX = float(__import__("os").environ.get("TRACK_LOW_MIX", "0.0"))


ACT_SOURCE = __import__("os").environ.get("TRACK_ACT", "flux")  # "flux" or "prob" (beat probability per onset)
PROB_STRENGTH_POW = 0.3  # beat probability x (relative strength)^this: loud beats still count a bit more


def prob_activation(o, probs: np.ndarray, fps=FPS) -> np.ndarray:
    """Activation from per-onset beat probabilities: a spike at each onset, max-pooled per frame."""
    frame_ms = 1000.0 / fps
    n = int(len(o.env) * o.hop_ms / frame_ms)
    rel = np.clip(o.strength / (np.percentile(o.strength, 99) + 1e-9), 0, 1) ** PROB_STRENGTH_POW
    a = np.zeros(n)
    f = np.clip((o.times / frame_ms).astype(int), 0, n - 1)
    np.maximum.at(a, f, probs * rel)
    return a * 0.98 + 0.01


def activation(o, fps=FPS, low_mix=None) -> np.ndarray:
    """Beat-ish activity per tracker frame in [0, 1].

    Plain onset flux can't tell the beat from the "and" (hats and offbeat stabs
    are as sharp as kicks), so a free-running tracker drifts onto whichever is
    louder for a stretch. Low-band flux (kick, bass hits) marks the beat far more
    often; it is mixed in at low_mix.
    """
    low_mix = LOW_MIX if low_mix is None else low_mix
    step = (1000.0 / fps) / o.hop_ms
    n = int(len(o.env) / step)
    idx = (np.arange(n)[:, None] * step + np.arange(int(np.ceil(step)))[None, :]).astype(int).clip(0, len(o.env) - 1)

    def pooled(env):
        a = env[idx].max(axis=1)
        return np.clip(a / (np.percentile(a, 99) + 1e-9), 0, 1)

    a = (1 - low_mix) * pooled(o.env) + low_mix * pooled(o.env_low)
    return a * 0.98 + 0.01


def viterbi_beats(act: np.ndarray, min_interval: int, max_interval: int, prior_interval=None,
                  fps=FPS) -> np.ndarray:
    """Beat frames of the best path through the bar-pointer HMM.

    prior_interval: expected beat interval in frames, a scalar or one value per
    frame; every beat pays TEMPO_PRIOR * log(interval / expected)^2.
    """
    intervals = np.arange(min_interval, max_interval + 1)
    n_i = len(intervals)
    starts = np.concatenate([[0], np.cumsum(intervals)[:-1]])
    ends = starts + intervals - 1
    n_states = int(intervals.sum())
    # tempo transitions between intervals, only at beats
    ratio = intervals[None, :] / intervals[:, None]
    log_t = -TRANSITION_LAMBDA * np.abs(ratio - 1)
    log_t -= np.logaddexp.reduce(log_t, axis=1, keepdims=True)
    # every beat pays for straying from the expected tempo: rubato survives when
    # onsets back it, a slide onto a triplet or dotted level does not
    log_i = np.log(intervals)
    log_p = np.log(np.broadcast_to(np.asarray(prior_interval, float), len(act))) if prior_interval is not None else None
    # observation: beat state vs any other state
    log_beat = np.log(act)
    log_non = np.log((1 - act) / (OBS_LAMBDA - 1))
    init = np.zeros(n_i) if log_p is None else -0.5 * ((log_i - log_p[0]) / 0.15) ** 2
    if log_p is None:
        log_p = np.full(len(act), np.nan)  # nan disables the per-beat tempo prior in the core
    return _viterbi_core(log_beat, log_non, intervals, starts, ends, log_t, log_i, np.ascontiguousarray(log_p),
                         init, TEMPO_PRIOR)


@njit(cache=True)
def _viterbi_core(log_beat, log_non, intervals, starts, ends, log_t, log_i, log_p, init, tempo_prior):
    """Forward pass and backtrack of the bar-pointer HMM (compiled: ~50x the numpy loop)."""
    n_t, n_i = len(log_beat), len(intervals)
    n_states = ends[-1] + 1
    v = np.empty(n_states)
    nv = np.empty(n_states)
    for i in range(n_i):  # any phase is possible at the start
        for st in range(starts[i], ends[i] + 1):
            v[st] = init[i] - np.log(intervals[i]) + log_non[0]
        v[starts[i]] += log_beat[0] - log_non[0]
    back = np.zeros((n_t, n_i), np.int16)
    best = np.empty(n_i)
    for t in range(1, n_t):
        for to in range(n_i):
            m, arg = -np.inf, 0
            for fr in range(n_i):
                val = v[ends[fr]] + log_t[fr, to]
                if val > m:
                    m, arg = val, fr
            if not np.isnan(log_p[t]):
                d = log_i[to] - log_p[t]
                m -= tempo_prior * d * d
            best[to] = m
            back[t, to] = arg
        # advance every position by one frame; a beat starts each interval block
        for i in range(n_i):
            for st in range(starts[i] + 1, ends[i] + 1):
                nv[st] = v[st - 1] + log_non[t]
            nv[starts[i]] = best[i] + log_beat[t]
        v, nv = nv, v
    # backtrack
    state = 0
    for st in range(n_states):
        if v[st] > v[state]:
            state = st
    blk = 0
    while blk + 1 < n_i and starts[blk + 1] <= state:
        blk += 1
    pos = state - starts[blk]
    frames = np.empty(n_t, np.int64)
    n_beats = 0
    for t in range(n_t - 1, -1, -1):
        if pos == 0:
            frames[n_beats] = t
            n_beats += 1
            if t > 0:
                blk = back[t, blk]
                pos = intervals[blk] - 1
        else:
            pos -= 1
    return frames[:n_beats][::-1].copy()


def refine(beat_ms: np.ndarray, o, tol_frac=0.15, tol_ms=25.0, weights=None) -> Beats:
    """Snap each beat to the strongest nearby onset; beats with none stay where the tracker put them."""
    t, w = o.times, (o.strength if weights is None else weights)
    out, sup = beat_ms.astype(float).copy(), np.zeros(len(beat_ms), bool)
    ivl = np.diff(beat_ms, prepend=beat_ms[0] - (beat_ms[1] - beat_ms[0] if len(beat_ms) > 1 else 500))
    for k, b in enumerate(beat_ms):
        tol = min(tol_ms, tol_frac * ivl[k])
        lo, hi = np.searchsorted(t, b - tol - 5), np.searchsorted(t, b + tol + 5)
        if hi > lo:
            j = lo + int(np.argmax(w[lo:hi] * np.exp(-0.5 * ((t[lo:hi] - b) / tol) ** 2)))
            if abs(t[j] - b) <= tol + 5:
                out[k], sup[k] = t[j], True
    return Beats(out, sup)


SHORT_LINE_BEATS = 2  # a red line spanning fewer beats than this is "short"
SHORT_LINE_COST = float(_env("SHORT_LINE_COST", "3"))  # x line_cost extra for a short line


def segment(beats: Beats, line_cost=60.0, sigma_ms=2.5, max_len=4096, allowed=None) -> list[Segment]:
    """Fewest red lines that keep the grid on the beats (optimal partition by DP).

    Cost of a section = squared residuals of its supported beats around a
    straight line (time vs beat index), in units of sigma_ms, plus line_cost.
    Sections share their boundary beat, as a new red line starts on a beat.
    A short section pays SHORT_LINE_COST x line_cost more: in rubato it can
    still win, a one-beat blip inside a smooth tempo ramp does not.
    allowed: bool per beat, where a new red line may start (mappers: downbeats only).
    """
    x = np.arange(len(beats.times), dtype=float)
    y = beats.times
    w = beats.supported.astype(float) + 1e-3  # unsupported beats barely count
    cw, cx, cy = np.cumsum(np.r_[0, w]), np.cumsum(np.r_[0, w * x]), np.cumsum(np.r_[0, w * y])
    cxx, cxy, cyy = np.cumsum(np.r_[0, w * x * x]), np.cumsum(np.r_[0, w * x * y]), np.cumsum(np.r_[0, w * y * y])

    def sse(i, j):  # beats i..j inclusive, vectorized over i
        W = cw[j + 1] - cw[i]
        Sx, Sy = cx[j + 1] - cx[i], cy[j + 1] - cy[i]
        Sxx, Sxy, Syy = cxx[j + 1] - cxx[i], cxy[j + 1] - cxy[i], cyy[j + 1] - cyy[i]
        vx = Sxx - Sx * Sx / W
        cov = Sxy - Sx * Sy / W
        vy = Syy - Sy * Sy / W
        return np.maximum(0, vy - np.where(vx > 1e-9, cov * cov / np.maximum(vx, 1e-9), 0))

    n = len(y)
    best = np.full(n, np.inf)
    prev = np.zeros(n, int)
    best[0] = 0.0
    for j in range(1, n):
        i = np.arange(max(0, j - max_len), j)
        if allowed is not None and j < n - 1 and not allowed[j]:
            continue  # no red line can start on this beat, so no section may end on it
        cost = best[i] + sse(i, j) / sigma_ms ** 2 + line_cost * (1 + SHORT_LINE_COST * (j - i < SHORT_LINE_BEATS))
        k = int(np.argmin(cost))
        best[j], prev[j] = cost[k], i[k]
    bounds = [n - 1]
    while bounds[-1] > 0:
        bounds.append(prev[bounds[-1]])
    bounds = bounds[::-1]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        xs, ys, ws = x[a:b + 1], y[a:b + 1], w[a:b + 1]
        A = np.stack([np.ones_like(xs), xs - xs[0]], 1) * np.sqrt(ws)[:, None]
        (t0, L), *_ = np.linalg.lstsq(A, ys * np.sqrt(ws), rcond=None)
        segs.append(Segment(float(t0), float(L)))
    return segs


SPAN = 1.35  # tempo freedom inside a section: wide enough for rubato, too narrow to slip an octave
FINE_WIN_MS, FINE_HOP_MS = 6000.0, 1000.0  # short windows so a few bars at a new tempo get their own section
SECTION_TOL = 0.08  # windows within ±8% belong to one section; the tracker follows drift inside it
FINE_RANGE = 1.75  # section tempos within x0.57..x1.75 of the main one: jumps like 210 -> 335 fit, x2 doesn't
# grid steps (0.1% each) the tempo may slide per 1 s window for free: 4%/s follows a ritardando like
# 200 -> 150 in 7 s instead of jumping, where x2 would look as good as the true tempo
FINE_DRIFT = int(_env("FINE_DRIFT", "40"))


def fine_sections(o, main_L: float) -> list[Segment]:
    """Tempo sections from short windows, around the (octave-chosen) main beat length."""
    from .detect import local_tempo_path, weights

    t, w = o.times, weights(o)
    if t[-1] - t[0] < 2 * FINE_WIN_MS:
        return [Segment(t[0], main_L)]
    # x2 / x0.5 must stay out of reach: coherence at double tempo is never worse
    # (it has the beats and the halves), so a wider range slides onto eighth notes
    centers, Lpath = local_tempo_path(t, w, main_L / FINE_RANGE, main_L * FINE_RANGE, FINE_WIN_MS, FINE_HOP_MS,
                                      jump_cost=0.1, prior_L=main_L, prior_w=0.3, drift=FINE_DRIFT)
    groups = [[0]]
    for i in range(1, len(centers)):
        if abs(np.log(Lpath[i] / np.median(Lpath[groups[-1]]))) < SECTION_TOL:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [Segment(t[0] if k == 0 else centers[g[0]] - FINE_HOP_MS / 2, float(np.median(Lpath[g])))
            for k, g in enumerate(groups)]


def polish(o, seg: Segment, lo: float, hi: float, subdiv=4) -> Segment:
    """Refit one red line on every onset of its span (beats and 1/2, 1/4 positions), keeping its phase.

    Beat times alone carry ~3 ms of jitter each; the onset-level regression the
    steady detector uses is far tighter over a long section.
    """
    from .detect import weights

    sel = (o.times >= lo) & (o.times < hi)
    t, w = o.times[sel], weights(o)[sel]
    phi, L = seg.time, seg.beat_length
    level_w = np.array([1.0, 0.25, 0.5, 0.25])
    for tol in (12, 8, 6):
        q = L / subdiv
        m = np.round((t - phi) / q)
        resid = t - (phi + m * q)
        keep = np.abs(resid) < min(tol, q / 3)
        if keep.sum() < 8:
            return seg
        ww = w[keep] * level_w[(m[keep] % subdiv).astype(int)] / (1 + (resid[keep] / 4) ** 2)
        A = np.stack([np.ones(keep.sum()), m[keep]], 1) * np.sqrt(ww)[:, None]
        (phi, q), *_ = np.linalg.lstsq(A, t[keep] * np.sqrt(ww), rcond=None)
        L = q * subdiv
    return Segment(float(phi), float(L))


def lines(o, beats: Beats, line_cost=60.0, sigma_ms=2.5, polish_beats=16, allowed=None) -> list[Segment]:
    """Red lines from tracked beats: optimal segmentation, then onset-level polish of long sections."""
    segs = segment(beats, line_cost, sigma_ms, allowed=allowed)
    out = []
    for k, s in enumerate(segs):
        end = segs[k + 1].time if k + 1 < len(segs) else beats.times[-1] + s.beat_length
        if (end - s.time) / s.beat_length >= polish_beats:
            p = polish(o, s, s.time - s.beat_length / 2, end - s.beat_length / 2)
            # keep the red line on the same beat it started on
            s = Segment(p.time + np.round((s.time - p.time) / p.beat_length) * p.beat_length, p.beat_length)
        out.append(s)
    return out


HYBRID_TOL_MS = 6.0  # an onset is "on" a grid if within this of one of its quarter positions
HYBRID_MARGIN = float(_env("HYBRID_MARGIN", "0.2"))  # tracker must put this much more onset mass on its grid, locally
HYBRID_MIN_RUN = 4  # beats: shorter switches to the tracker are noise
# a whole run of tracked beats must also win on aggregate, or it is wobble (e.g. a quiet outro), not tempo
RUN_MIN_GAIN = float(_env("RUN_MIN_GAIN", "0.4"))
RUN_MIN_BEATS = int(_env("RUN_MIN_BEATS", "16"))
RUN_SHIFT_MS = float(_env("RUN_SHIFT_MS", "24"))  # ...and beat the steady grid moved by up to this
RUN_MIN_SHIFT_GAIN = float(_env("RUN_MIN_SHIFT_GAIN", "0.1"))  # by this much (single-line maps given extra lines: 3.8% -> 0%)
SHIFTS = np.arange(-RUN_SHIFT_MS, RUN_SHIFT_MS + 0.5, 2.0)
RUN_SHORT_SHIFT_GAIN = float(_env("RUN_SHORT_SHIFT_GAIN", "0.15"))  # a shorter run must beat the shifted grid by this
RUN_SHORT_MIN_NET = float(_env("RUN_SHORT_MIN_NET", "0.05"))  # and bridge two steady tempos this far apart (log ratio): a ritardando into a new section
RUN_SHORT_MAX_DEV = float(_env("RUN_SHORT_MAX_DEV", "0.35"))  # beyond this the tracker jumped an octave (x1.5, x2), no ritardando does in a few beats
# a run ending this close to the song's end has no steady tempo after it to lead into: it may
# change tempo on its own (an accelerando to the end), smoothly, with no beat-to-beat octave jump
RUN_END_BEATS = int(_env("RUN_END_BEATS", "16"))
RUN_END_MAX_JUMP = float(_env("RUN_END_MAX_JUMP", "0.06"))  # log change of the 4-beat average per beat
SWITCH_TOL_MS = float(_env("SWITCH_TOL_MS", "10"))  # grids this close may hand over to each other
SWITCH_MAX_EXTEND = int(_env("SWITCH_MAX_EXTEND", "16"))  # beats a tracked run may grow to reach such a point
# a ramp run whose first beats sit on the steady grid's "and" (this share of them within 1/8 beat
# of it) took the other phase, e.g. slipping half a beat in the run-up: its beats move back onto the
# steady phase, so one phase holds through the song and a half-beat fix of it fixes it all. Only
# runs whose tempo leaves the steady one by RUN_PHASE_MIN_DEV: near it, a tracker on the "and" is a
# phase reset or drift the steady grid doesn't follow, and moving it back cost lines on ranked maps
RUN_PHASE_BEATS = 8
RUN_PHASE_SHARE = float(_env("RUN_PHASE_SHARE", "0.75"))
RUN_PHASE_MIN_DEV = float(_env("RUN_PHASE_MIN_DEV", "0.1"))


def _quarters(beat_times: np.ndarray) -> np.ndarray:
    """Beat and 1/4, 1/2, 3/4 positions between consecutive beats."""
    a, b = beat_times[:-1], beat_times[1:]
    q = a[:, None] + (b - a)[:, None] * np.array([0, 0.25, 0.5, 0.75])[None, :]
    return np.concatenate([q.ravel(), beat_times[-1:]])


def _on_grid(t: np.ndarray, grid: np.ndarray, tol: float) -> np.ndarray:
    j = np.clip(np.searchsorted(grid, t), 1, len(grid) - 1)
    return np.minimum(np.abs(t - grid[j - 1]), np.abs(t - grid[j])) < tol


def _runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Drop True runs shorter than min_run and fill False gaps shorter than min_run."""
    out = mask.copy()
    for val in (False, True):  # close short gaps, then open short runs
        i = 0
        while i < len(out):
            j = i
            while j < len(out) and out[j] == out[i]:
                j += 1
            if out[i] == val and j - i < min_run and i > 0 and j < len(out):
                out[i:j] = not val
            i = j
    return out


def hybrid(o, steady: list[Segment], tracked: Beats, margin=HYBRID_MARGIN) -> Beats:
    """Steady beats where a steady grid explains the onsets, tracked beats where it can't.

    The steady detector's phase is consistent over the whole song, the tracker's
    is not (it slides onto loud offbeats). Comparing both on quarter-note grids
    makes a half-beat slide neither help nor hurt the tracker; where it does win
    (rubato, drift, short tempo changes), its beats are used, moved back to the
    steady phase if the run starts on the "and".
    """
    from .detect import weights
    from .grid import beats as grid_beats

    t, w = o.times, weights(o)
    if len(tracked.times) < 4 or not steady:
        return tracked
    start, end = min(t[0], tracked.times[0]), max(t[-1], tracked.times[-1])
    sb = grid_beats(steady, start, end)
    if len(sb) < 4:
        return tracked
    q_s = _quarters(sb)
    on_s = _on_grid(t, q_s, HYBRID_TOL_MS)
    on_t = _on_grid(t, _quarters(tracked.times), HYBRID_TOL_MS)
    cw = np.cumsum(np.r_[0, w])
    cs, ct = np.cumsum(np.r_[0, w * on_s]), np.cumsum(np.r_[0, w * on_t])
    L = np.diff(sb, append=sb[-1] + (sb[-1] - sb[-2]))
    lo, hi = np.searchsorted(t, sb - 2 * L), np.searchsorted(t, sb + 2 * L)
    mass = cw[hi] - cw[lo] + 1e-9
    use_t = _runs((ct[hi] - ct[lo] - (cs[hi] - cs[lo])) / mass > margin, HYBRID_MIN_RUN)
    # judge each run of tracked beats as a whole
    devs = {}
    k = 0
    while k < len(use_t):
        if not use_t[k]:
            k += 1
            continue
        k2 = k
        while k2 < len(use_t) and use_t[k2]:
            k2 += 1
        a, b = lo[k], hi[k2 - 1]
        mass = cw[b] - cw[a] + 1e-9
        gain = (ct[b] - ct[a] - (cs[b] - cs[a])) / mass
        # the same grid a few ms later is not a tempo change: soft attacks (pads, a quiet
        # intro or break) peak 5-25 ms late, and a free tracker follows them
        shifted = max(np.sum(w[a:b] * _on_grid(t[a:b], q_s + d, HYBRID_TOL_MS)) for d in SHIFTS) / mass
        gain_shift = (ct[b] - ct[a]) / mass - shifted
        # how far the tracked tempo really leaves the steady one (4-beat average): a ritardando
        # moves 3-10%, a tracker wobbling on the same tempo stays within ~1%
        sel = (tracked.times >= sb[k] - L[k] / 2) & (tracked.times <= sb[k2 - 1] + L[k2 - 1] / 2)
        iv = np.diff(tracked.times[sel])
        dev = net = 0.0
        jump = np.inf
        if len(iv) >= 4:
            ma = np.convolve(iv, np.ones(4) / 4, mode="valid")
            dev = float(np.max(np.abs(np.log(ma / np.median(L[k:k2])))))
            # net change across the run: a ritardando leads into a new tempo, a tracker slipping onto
            # triplets for a few beats comes back to the one it left
            net = float(abs(np.log(iv[-2:].mean() / iv[:2].mean())))
            if len(ma) >= 2:
                jump = float(np.max(np.abs(np.diff(np.log(ma)))))
        net_steady = float(abs(np.log(L[min(k2 + 1, len(L) - 1)] / L[max(k - 2, 0)])))
        if _env("HYBRID_DEBUG"):
            print(f"run {sb[k] - 20.4:.0f}-{sb[k2 - 1] - 20.4:.0f} beats {k2 - k} gain {gain:.2f} vs shifted {gain_shift:.2f} "
                  f"steady {(cs[b] - cs[a]) / mass:.2f} shifted {shifted:.2f} tracked {(ct[b] - ct[a]) / mass:.2f} "
                  f"tempo dev {dev:.1%} net {net:.1%} net steady {net_steady:.1%} jump {jump:.1%}")
        # a few beats of a clear tempo change (a ritardando into a new section)
        short_ok = (gain_shift >= RUN_SHORT_SHIFT_GAIN and net_steady >= RUN_SHORT_MIN_NET
                    and dev <= RUN_SHORT_MAX_DEV)
        long_ok = k2 - k >= RUN_MIN_BEATS and gain >= RUN_MIN_GAIN and gain_shift >= RUN_MIN_SHIFT_GAIN
        end_ok = (k2 >= len(use_t) - RUN_END_BEATS and gain_shift >= RUN_SHORT_SHIFT_GAIN
                  and net >= RUN_SHORT_MIN_NET and jump <= RUN_END_MAX_JUMP)
        devs[k] = dev
        if not (long_ok or short_ok or end_ok):
            use_t[k:k2] = False
        elif k2 >= len(use_t) - RUN_END_BEATS:  # no steady tempo left to go back to: the run ends the song
            use_t[k2:] = True
            k2 = len(use_t)
        k = k2

    def nearest_gap(times):  # each steady beat to the nearest tracked one
        j = np.clip(np.searchsorted(times, sb), 1, len(times) - 1)
        return np.minimum(np.abs(sb - times[j - 1]), np.abs(sb - times[j]))

    gap = nearest_gap(tracked.times)
    mids = (tracked.times[:-1] + tracked.times[1:]) / 2
    times, sup = tracked.times, tracked.supported
    moved = set()  # starts of runs moved onto the steady phase: the seam there agrees by construction
    runs, k = [], 0
    while k < len(use_t):
        if not use_t[k]:
            k += 1
            continue
        k2 = k
        while k2 < len(use_t) and use_t[k2]:
            k2 += 1
        runs.append((k, k2))
        k = k2
    for r, (k, k2) in enumerate(runs):
        n = min(RUN_PHASE_BEATS, k2 - k)
        on_and = np.mean(np.abs(gap[k:k + n] / L[k:k + n] - 0.5) < 0.125)
        if devs.get(k, 0.0) >= RUN_PHASE_MIN_DEV and on_and >= RUN_PHASE_SHARE:
            # up to the next run: the seam at the end may move on with these beats (see below)
            a = sb[k] - L[k] / 2
            b = sb[runs[r + 1][0]] - L[runs[r + 1][0]] / 2 if r + 1 < len(runs) else np.inf
            keep, msel = (times < a) | (times >= b), (mids >= a) & (mids < b)
            times = np.r_[times[keep], mids[msel]]
            sup = np.r_[sup[keep], np.ones(msel.sum(), bool)]
            order = np.argsort(times)
            times, sup = times[order], sup[order]
            moved.add(k)
    tracked = Beats(times, sup, tracked.tracked_share)
    # switch sources only where the two grids agree: a run ending while the tracker and the
    # steady grid are far apart would put a phase jump mid-phrase (e.g. the last bars of a
    # tempo ramp, where onsets are too sparse to back either grid), so extend it until they meet
    gap = nearest_gap(tracked.times)
    k = 0
    while k < len(use_t):
        if not use_t[k]:
            k += 1
            continue
        k2 = k
        while k2 < len(use_t) and use_t[k2]:
            k2 += 1
        e = k2
        while e < len(use_t) and e - k2 < SWITCH_MAX_EXTEND and gap[e] > SWITCH_TOL_MS and not use_t[e]:
            e += 1
        b = k
        while (b > 0 and k - b < SWITCH_MAX_EXTEND and gap[b - 1] > SWITCH_TOL_MS and not use_t[b - 1]
               and not (k in moved and gap[k] <= SWITCH_TOL_MS)):
            b -= 1
        use_t[b:e] = True
        k = e

    out_t, out_s, k = [], [], 0
    while k < len(sb):
        if not use_t[k]:
            out_t.append(sb[k])
            out_s.append(True)
            k += 1
            continue
        k2 = k
        while k2 < len(sb) and use_t[k2]:
            k2 += 1
        a, b = sb[k] - L[k] / 2, sb[k2 - 1] + L[k2 - 1] / 2
        sel = (tracked.times >= a) & (tracked.times < b)
        cand, sup = tracked.times[sel], tracked.supported[sel]
        out_t.extend(cand)
        out_s.extend(sup)
        k = k2
    times, sup = np.array(out_t), np.array(out_s)
    order = np.argsort(times)
    times, sup = times[order], sup[order]
    keep = np.r_[True, np.diff(times) > 0.4 * np.median(np.diff(times))]  # seams can double a beat
    return Beats(times[keep], sup[keep], tracked_share=float(use_t.mean()))


SMOOTH_LAMBDA = float(__import__("os").environ.get("TRACK_SMOOTH", "30"))  # stiffness of the tempo curve
SMOOTH_HUBER_MS = float(_env("SMOOTH_HUBER_MS", "1"))  # beat-length changes beyond this count linearly (near L1):
# a step 180 -> 170 stays one step instead of a ramp spread over 8 beats, as mappers time it


def smooth_beats(o, skeleton: Beats, lam: float = SMOOTH_LAMBDA, weights=None) -> Beats:
    """Beat times from a smooth tempo curve fitted through every onset, not snapped to onsets.

    The skeleton (tracked beats) only says which beat each onset belongs to and
    at which quarter (beat, 1/4, 1/2, 3/4). The beat times T_k are then the least
    squares solution of: every such onset lies on its quarter position
    (linear interpolation between T_k and T_k+1), plus lam * (T_k+1 - 2 T_k + T_k-1)^2
    so tempo changes smoothly; that penalty is Huber-weighted, so a genuine jump
    costs little. A third of mappers' beats in rubato songs have no onset at
    all; this fills them from their neighbours, as a mapper would.
    """
    from scipy.sparse import coo_matrix, diags, vstack
    from scipy.sparse.linalg import spsolve

    from .detect import weights as onset_weights

    b = skeleton.times
    n = len(b)
    if n < 4:
        return skeleton
    t = o.times
    w_on = onset_weights(o) if weights is None else weights
    T = b.astype(float).copy()
    level_w = np.array([1.0, 0.3, 0.6, 0.3])
    huber = np.ones(n - 2)
    for tol in (20.0, 12.0, 8.0, 6.0):
        k = np.clip(np.searchsorted(T, t, side="right") - 1, 0, n - 2)
        frac = (t - T[k]) / (T[k + 1] - T[k])
        inside = (frac >= -0.125) & (frac < 1.125)
        q = np.round(frac * 4) / 4
        # quarters past the end of a beat belong to the next beat's start
        nxt = q >= 1
        k2 = np.where(nxt, np.minimum(k + 1, n - 2), k)
        q2 = np.where(nxt, q - 1, q)
        pred = T[k2] + q2 * (T[k2 + 1] - T[k2])
        resid = t - pred
        keep = inside & (np.abs(resid) < tol) & (q2 >= 0)
        if keep.sum() < 8:
            break
        kk, qq = k2[keep], q2[keep]
        ww = w_on[keep] * level_w[(np.round(qq * 4).astype(int)) % 4] / (1 + (resid[keep] / 3) ** 2)
        rows = np.arange(keep.sum())
        A = coo_matrix((np.r_[1 - qq, qq], (np.r_[rows, rows], np.r_[kk, kk + 1])), shape=(keep.sum(), n)).tocsr()
        Wd = diags(ww)
        D2 = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
        scale = np.median(ww) if len(ww) else 1.0
        H = diags(huber * lam * scale)
        lhs = (A.T @ Wd @ A + D2.T @ H @ D2 + diags(np.full(n, 1e-6 * scale))).tocsc()
        rhs = A.T @ (ww * t[keep]) + 1e-6 * scale * T
        T = spsolve(lhs, rhs)
        d2 = D2 @ T
        huber = np.minimum(1.0, SMOOTH_HUBER_MS / np.maximum(np.abs(d2), 1e-9))
        if np.any(np.diff(T) <= 0):  # degenerate solve: keep the skeleton
            return skeleton
    return Beats(T, np.ones(n, bool))


def track_beats(o, sections: list[Segment], probs: np.ndarray | None = None) -> Beats:
    """Beats tracked in one pass over the whole song, expecting each section's tempo in its span.

    One continuous path matters: tracking sections separately lets the phase be
    re-chosen at every seam, so steady songs flip between beat and offbeat.
    """
    act = activation(o) if probs is None else prob_activation(o, probs)
    frame_ms = 1000.0 / FPS
    secs = sorted(sections, key=lambda s: s.time)
    starts = np.array([0] + [int(s.time / frame_ms) for s in secs[1:]])
    expected = np.empty(len(act))
    for k, s in enumerate(secs):
        expected[starts[k]:(starts[k + 1] if k + 1 < len(secs) else len(act))] = s.beat_length / frame_ms
    lo = max(3, int(np.floor(expected.min() / SPAN)))
    hi = int(np.ceil(expected.max() * SPAN))
    frames = viterbi_beats(act, lo, hi, prior_interval=expected)
    if len(frames) < 2:
        return Beats(np.zeros(0), np.zeros(0, bool))
    skeleton = refine(frames * frame_ms + frame_ms / 2, o, weights=None if probs is None else probs * o.strength)
    return smooth_beats(o, skeleton) if SMOOTH_LAMBDA > 0 else skeleton


def track(o, sections: list[Segment], line_cost=60.0, sigma_ms=2.5) -> tuple[Beats, list[Segment]]:
    """Tracked beats and the red lines they compress to."""
    beats = track_beats(o, sections)
    return beats, (lines(o, beats, line_cost, sigma_ms) if len(beats.times) > 1 else [])
