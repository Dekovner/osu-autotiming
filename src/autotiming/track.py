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

from .grid import Segment

FPS = 100  # tracker frame rate; beat times are refined to onsets afterwards
OBS_LAMBDA = 16.0  # how much a beat frame prefers onset activity (madmom's observation lambda)
TRANSITION_LAMBDA = 60.0  # tempo-change stiffness: exp(-lambda * |ratio - 1|)
TEMPO_PRIOR = 100.0  # per-beat cost of log(interval / section interval)^2: 10% off ~ 1 nat, 30% ~ 7


@dataclass
class Beats:
    times: np.ndarray  # ms, in the onset clock
    supported: np.ndarray  # bool: an onset backs this beat (else it is interpolated)


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
    v = np.full(n_states, -np.inf)
    init = np.zeros(n_i) if log_p is None else -0.5 * ((log_i - log_p[0]) / 0.15) ** 2
    for i in range(n_i):  # any phase is possible at the start
        v[starts[i]:ends[i] + 1] = init[i] - np.log(intervals[i])
    v += log_non[0]
    v[starts] += log_beat[0] - log_non[0]
    back = np.zeros((len(act), n_i), np.int16)
    for t in range(1, len(act)):
        from_end = v[ends][:, None] + log_t  # [from interval, to interval]
        arg = from_end.argmax(axis=0)
        best = from_end[arg, np.arange(n_i)]
        if log_p is not None:
            best = best - TEMPO_PRIOR * (log_i - log_p[t]) ** 2
        v = np.roll(v, 1)
        v[starts] = best
        back[t] = arg
        v += log_non[t]
        v[starts] += log_beat[t] - log_non[t]
    # backtrack
    state = int(np.argmax(v))
    blk = int(np.searchsorted(starts, state, side="right") - 1)
    pos = state - starts[blk]
    beat_frames = []
    for t in range(len(act) - 1, -1, -1):
        if pos == 0:
            beat_frames.append(t)
            if t > 0:
                blk = int(back[t, blk])
                pos = intervals[blk] - 1
        else:
            pos -= 1
    return np.array(beat_frames[::-1])


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


def segment(beats: Beats, line_cost=60.0, sigma_ms=2.5, max_len=4096) -> list[Segment]:
    """Fewest red lines that keep the grid on the beats (optimal partition by DP).

    Cost of a section = squared residuals of its supported beats around a
    straight line (time vs beat index), in units of sigma_ms, plus line_cost.
    Sections share their boundary beat, as a new red line starts on a beat.
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
        cost = best[i] + sse(i, j) / sigma_ms ** 2 + line_cost
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


def fine_sections(o, main_L: float) -> list[Segment]:
    """Tempo sections from short windows, around the (octave-chosen) main beat length."""
    from .detect import local_tempo_path, weights

    t, w = o.times, weights(o)
    if t[-1] - t[0] < 2 * FINE_WIN_MS:
        return [Segment(t[0], main_L)]
    # x2 / x0.5 must stay out of reach: coherence at double tempo is never worse
    # (it has the beats and the halves), so a wider range slides onto eighth notes
    centers, Lpath = local_tempo_path(t, w, main_L / FINE_RANGE, main_L * FINE_RANGE, FINE_WIN_MS, FINE_HOP_MS,
                                      jump_cost=0.1, prior_L=main_L, prior_w=0.3)
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


def lines(o, beats: Beats, line_cost=60.0, sigma_ms=2.5, polish_beats=16) -> list[Segment]:
    """Red lines from tracked beats: optimal segmentation, then onset-level polish of long sections."""
    segs = segment(beats, line_cost, sigma_ms)
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
HYBRID_MARGIN = 0.10  # tracker must put this much more onset mass on its grid, locally
HYBRID_MIN_RUN = 4  # beats: shorter switches to the tracker are noise


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
    steady phase if it sits on the "and".
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
    on_s = _on_grid(t, _quarters(sb), HYBRID_TOL_MS)
    on_t = _on_grid(t, _quarters(tracked.times), HYBRID_TOL_MS)
    cw = np.cumsum(np.r_[0, w])
    cs, ct = np.cumsum(np.r_[0, w * on_s]), np.cumsum(np.r_[0, w * on_t])
    L = np.diff(sb, append=sb[-1] + (sb[-1] - sb[-2]))
    lo, hi = np.searchsorted(t, sb - 2 * L), np.searchsorted(t, sb + 2 * L)
    mass = cw[hi] - cw[lo] + 1e-9
    use_t = _runs((ct[hi] - ct[lo] - (cs[hi] - cs[lo])) / mass > margin, HYBRID_MIN_RUN)

    out_t, out_s, k = [], [], 0
    mids = (tracked.times[:-1] + tracked.times[1:]) / 2
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
        msel = (mids >= a) & (mids < b)
        # back to the steady phase if the tracker sits on the "and" here
        if len(cand) and msel.any() and abs(mids[msel][0] - sb[k]) < abs(cand[0] - sb[k]):
            cand, sup = mids[msel], np.ones(msel.sum(), bool)
        out_t.extend(cand)
        out_s.extend(sup)
        k = k2
    times, sup = np.array(out_t), np.array(out_s)
    order = np.argsort(times)
    times, sup = times[order], sup[order]
    keep = np.r_[True, np.diff(times) > 0.4 * np.median(np.diff(times))]  # seams can double a beat
    return Beats(times[keep], sup[keep])


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
    return refine(frames * frame_ms + frame_ms / 2, o, weights=None if probs is None else probs * o.strength)


def track(o, sections: list[Segment], line_cost=60.0, sigma_ms=2.5) -> tuple[Beats, list[Segment]]:
    """Tracked beats and the red lines they compress to."""
    beats = track_beats(o, sections)
    return beats, (lines(o, beats, line_cost, sigma_ms) if len(beats.times) > 1 else [])
