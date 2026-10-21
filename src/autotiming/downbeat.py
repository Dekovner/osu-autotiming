"""Which beat of the bar is "1": features per candidate bar phase.

Mappers place red lines on downbeats. Cues for "1": the bass moves to a new note
there, the harmony changes there (new chord per bar), the kick lands there, the
bar's loudest hits start there.
"""

from dataclasses import dataclass

import numpy as np

from .detect import N_BANDS, Onsets

EDGES = np.geomspace(30, 16000, N_BANDS + 1)
GROUPS = {"sub": (0, 90), "low": (90, 250), "mid": (250, 2500), "high": (2500, 16000)}


CHROMA_HOP_MS = 46.4  # ~2048 samples at 44.1 kHz


def chroma(pcm: np.ndarray, sr: int, n_fft: int = 8192) -> np.ndarray:
    """12-bin pitch-class energy per frame (55 Hz .. 2 kHz), L1-normalized. Frame k is centred at k * CHROMA_HOP_MS."""
    hop = int(round(sr * CHROMA_HOP_MS / 1000))
    pad = np.concatenate([np.zeros(n_fft // 2, np.float32), pcm.astype(np.float32), np.zeros(n_fft, np.float32)])
    n = (len(pad) - n_fft) // hop
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    use = (freqs >= 55) & (freqs <= 2000)
    pc = (np.round(12 * np.log2(freqs[use] / 440.0)) % 12).astype(int)
    M = np.zeros((use.sum(), 12), np.float32)
    M[np.arange(use.sum()), pc] = 1
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n, 12), np.float32)
    for s in range(0, n, 128):  # chunked: a few MB of temporaries
        e = min(n, s + 128)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(pad[idx] * win, axis=1))[:, use]
        out[s:e] = np.log1p(mag) @ M
    return out / (out.sum(1, keepdims=True) + 1e-9)


BASS_HZ = (32, 260)


def bass_chroma(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Pitch classes of the bass (32-260 Hz, L1-normalized) and its log energy per frame, and the
    frame step in ms (frame k centred at k * step). Run at a quarter of 44.1 kHz: the bass needs
    2.7 Hz bins (a semitone at 40 Hz is 2.4 Hz), 4x cheaper on the decimated signal."""
    from scipy.signal import resample_poly

    q = max(1, int(sr // 11025))
    x, sr_q = resample_poly(pcm.astype(np.float32), 1, q).astype(np.float32), sr / q
    n_fft, hop = int(2 ** np.ceil(np.log2(sr_q / 2.7))), int(round(sr_q * CHROMA_HOP_MS / 1000))
    pad = np.concatenate([np.zeros(n_fft // 2, np.float32), x, np.zeros(n_fft, np.float32)])
    n = (len(pad) - n_fft) // hop
    freqs = np.fft.rfftfreq(n_fft, 1 / sr_q)
    use = (freqs >= BASS_HZ[0]) & (freqs <= BASS_HZ[1])
    pc = (np.round(12 * np.log2(freqs[use] / 440.0)) % 12).astype(int)
    M = np.zeros((use.sum(), 12), np.float32)
    M[np.arange(use.sum()), pc] = 1
    win = np.hanning(n_fft).astype(np.float32)
    out, energy = np.empty((n, 12), np.float32), np.empty(n, np.float32)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        idx = (np.arange(s, e) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.log1p(np.abs(np.fft.rfft(pad[idx] * win, axis=1))[:, use])
        out[s:e], energy[s:e] = mag @ M, mag.sum(1)
    return out / (out.sum(1, keepdims=True) + 1e-9), energy, hop / sr_q * 1000


@dataclass
class Harmony:
    chroma: np.ndarray  # chroma() frames, CHROMA_HOP_MS apart
    bass: np.ndarray  # bass_chroma() frames, bass_hop_ms apart
    bass_energy: np.ndarray
    bass_hop_ms: float


def harmony(pcm: np.ndarray, sr: int) -> Harmony:
    return Harmony(chroma(pcm, sr), *bass_chroma(pcm, sr))


def _beat_means(X: np.ndarray, hop_ms: float, beats: np.ndarray) -> np.ndarray:
    """Mean frame of X over each beat [beats[k], beats[k + 1])."""
    fr = np.clip((beats / hop_ms).astype(int), 0, len(X) - 1)
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0, dtype=np.float64)])
    a = fr[:-1]
    b = np.minimum(np.maximum(fr[1:], a + 1), len(X))
    return (c[b] - c[a]) / (b - a)[:, None]


def _boundary_change(P: np.ndarray, w: int) -> np.ndarray:
    """Cosine distance between the mean of the w beats before and the w beats after each beat
    boundary k (the start of beat k); NaN where either side runs out."""
    d = np.full(len(P) + 1, np.nan)
    if len(P) < 2 * w:
        return d
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    k = np.arange(w, len(P) - w + 1)
    u, v = c[k] - c[k - w], c[k + w] - c[k]
    d[k] = 1 - (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return d


def _mean_at(d: np.ndarray, sel: np.ndarray) -> float:
    v = d[sel]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else 0.0


def bass_features(h: Harmony, beats: np.ndarray, first: int) -> dict:
    """Bar-start evidence on beats[k] for k = first (mod 4): the bass changes note (1 and 2 beats
    either side), bass and chords change between 2-bar phrases (on "1" of odd or even bars),
    and the bass gets louder. Bass change over 2 beats alone picks the mappers' "1" on 93%
    of ranked maps (bench/downbeat_exp.py), harmonic change over a bar 85%."""
    f = {k: 0.0 for k in ("bass_change1", "bass_change2", "bass_phrase", "bass_rise", "ch_change2", "ch_phrase")}
    if len(beats) < 12:
        return f
    k = np.arange(len(beats))
    one = (k - first) % 4 == 0
    B = _beat_means(h.bass, h.bass_hop_ms, beats)
    C = _beat_means(h.chroma, CHROMA_HOP_MS, beats)
    f["bass_change1"] = _mean_at(_boundary_change(B, 1), one)
    f["bass_change2"] = _mean_at(_boundary_change(B, 2), one)
    f["ch_change2"] = _mean_at(_boundary_change(C, 2), one)
    for name, P in (("bass_phrase", B), ("ch_phrase", C)):
        d = _boundary_change(P, 8)
        f[name] = max(_mean_at(d, (k - first) % 8 == p) for p in (0, 4))
    fr = (beats[one] / h.bass_hop_ms).astype(int)
    E = h.bass_energy
    rise = [E[i:i + 2].max() - E[i - 2:i].mean() for i in fr if 2 <= i < len(E) - 2]
    f["bass_rise"] = float(np.mean(rise)) if rise else 0.0
    return f


def _chroma_change(ch: np.ndarray, starts: np.ndarray, bar: float) -> float:
    """Mean cosine distance between the pitch-class profile of the bar before and after each "1"."""
    if ch is None or len(starts) < 3:
        return 0.0
    k = lambda tm: np.clip((tm / CHROMA_HOP_MS).astype(int), 0, len(ch) - 1)
    d = []
    for x in starts[1:-1]:
        a = ch[k(np.array([x - bar])) [0]:k(np.array([x]))[0]].mean(0)
        b = ch[k(np.array([x]))[0]:k(np.array([x + bar]))[0]].mean(0)
        if np.isfinite(a).all() and np.isfinite(b).all():
            d.append(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.mean(d)) if d else 0.0


def bar_phase(ch: np.ndarray, beat_times: np.ndarray, meter: int = 4) -> int:
    """Which beat index (mod meter) starts the bars: where the harmony changes most.

    Chroma change alone picks the mappers' downbeat on 85% of ranked maps
    (bench/downbeat_exp.py; kick, loudness and onset cues only added noise).
    Works on any beat sequence, so bars may change length through a tempo ramp.
    """
    if ch is None or len(beat_times) < 3 * meter:
        return 0
    k = lambda tm: int(np.clip(tm / CHROMA_HOP_MS, 0, len(ch) - 1))
    best, best_s = 0, -1.0
    for j in range(meter):
        starts = beat_times[j::meter]
        d = []
        for a, b, c in zip(starts[:-2], starts[1:-1], starts[2:]):
            x, y = ch[k(a):k(b)], ch[k(b):k(c)]
            if len(x) and len(y):
                u, v = x.mean(0), y.mean(0)
                d.append(1 - (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        s = float(np.mean(d)) if d else 0.0
        if s > best_s:
            best, best_s = j, s
    return best


MODEL_PATH = __import__("pathlib").Path(__file__).resolve().parent / "models" / "downbeat_trees.npz"
FEATURE_KEYS = ["chroma_change", "count_on_one", "first_on_one", "flux_on_one", "harm_change", "high_on_one", "low_on_one",
                "mid_on_one", "strength_on_one", "sub_on_one", "top_on_one",
                "bass_change1", "bass_change2", "bass_phrase", "bass_rise", "ch_change2", "ch_phrase"]


def candidate_matrix(feats: list[dict]) -> np.ndarray:
    """Model input for the 4 bar-phase candidates: each one's features relative to the
    four, its rank among them, the candidate two beats later ("its 3"), and the spread."""
    F = np.nan_to_num(np.array([[f[k] for k in FEATURE_KEYS] for f in feats]))
    rel = F - F.mean(0, keepdims=True)
    from scipy.stats import rankdata

    rank = rankdata(F, axis=0)  # ties share a rank: candidate order must carry no information
    other = np.roll(F, -2, axis=0) - F.mean(0, keepdims=True)
    spread = np.repeat((F.max(0) - F.min(0))[None, :], len(F), axis=0)
    return np.concatenate([rel, rank, other, spread], axis=1)


def downbeat_scores(o: Onsets, ch: Harmony, phi: float, L: float, lo: float, hi: float) -> np.ndarray:
    """Score per beat j = 0..3 from phi for "the bar starts here": the trees' probability, or
    chroma change without the model."""
    from .beatprob import load_model, predict_proba

    feats = [bar_features(o, phi + j * L, L, lo, hi, ch=ch) for j in range(4)]
    if MODEL_PATH.exists():
        return np.asarray(predict_proba(load_model(MODEL_PATH), candidate_matrix(feats)), float)
    return np.array([f["chroma_change"] for f in feats])


def downbeat_phase(o: Onsets, ch, phi: float, L: float, lo: float, hi: float) -> int:
    """Which of the 4 beats from phi starts the bar. Boosted trees over bar features
    (bench/downbeat_exp.py cv): 94.8% on 686 ranked maps out of fold; 88.3% before
    the bass features, 85% for chroma change alone."""
    return int(np.argmax(downbeat_scores(o, ch, phi, L, lo, hi)))


def _near(t, grid_times, tol):
    j = np.clip(np.searchsorted(grid_times, t), 1, max(1, len(grid_times) - 1))
    return np.minimum(np.abs(t - grid_times[j - 1]), np.abs(t - grid_times[np.minimum(j, len(grid_times) - 1)])) < tol


def bar_features(o: Onsets, phi: float, L: float, lo: float, hi: float, meter: int = 4, ch: Harmony | None = None) -> dict:
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
    f["chroma_change"] = _chroma_change(None if ch is None else ch.chroma, starts, bar)
    if ch is not None:
        m0 = int(np.ceil((lo - phi) / L))  # beats[0] is phi + m0 beats: "1" falls on beats[-m0 % 4]
        f |= bass_features(ch, beats, -m0 % 4)
    # song structure: crashes, drops and section entries (the loudest hits) land on "1",
    # and so does the first real hit of the song
    if len(s) >= 20 and len(starts):
        top = s >= np.percentile(s, 98)
        f["top_on_one"] = float(on_one[top].mean())
    else:
        f["top_on_one"] = 0.0
    # first real hit of the whole song (not of this window: its start must not leak the answer)
    strong = np.flatnonzero(o.strength >= np.median(o.strength))
    f["first_on_one"] = 0.5
    if len(strong):
        t0 = o.times[strong[0]]
        if lo - bar <= t0 < hi:  # this section's grid reaches the song start
            d = (t0 - phi) % bar
            f["first_on_one"] = float(min(d, bar - d) < tol)
    return f
