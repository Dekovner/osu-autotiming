"""Synthetic songs with exactly known timing.

    uv run python bench/synth.py --n 60

Each song gets a random fractional BPM, offset and style, optionally a tempo
change, and is encoded several ways. The truth is stored in osu!'s clock: the
encoded file is decoded with BASS and aligned to the source, so encoder delay
(e.g. +25 ms for an mp3 without a LAME tag) is measured, not assumed.
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from autotiming.decode import decode_bass

SR = 44100
OUT = Path(__file__).resolve().parents[1] / "data" / "synth"
STYLES = ("edm", "soft", "syncopated", "ambient_intro", "rock")


def env(n, attack, decay):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1) if attack > 0 else np.ones(n)
    return a * np.exp(-t / decay)


def kick(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.0005, 0.12)


def snare(rng):
    n = int(0.25 * SR)
    noise = sosfilt(butter(2, [1500, 9000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    tone = np.sin(2 * np.pi * 190 * np.arange(n) / SR)
    return (0.7 * noise * env(n, 0.0005, 0.06) + 0.5 * tone * env(n, 0.0005, 0.04))


def hat(rng):
    n = int(0.08 * SR)
    return 0.35 * sosfilt(butter(4, 7000, "highpass", fs=SR, output="sos"), rng.standard_normal(n)) * env(n, 0.0003, 0.02)


def pluck(freq, dur, attack=0.002):
    n = int(dur * SR)
    t = np.arange(n) / SR
    tone = sum(np.sin(2 * np.pi * freq * h * t) / h**1.5 for h in range(1, 7))
    return 0.25 * tone * env(n, attack, dur / 3)


def pad(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    tone = sum(np.sin(2 * np.pi * freq * m * t + m) for m in (1, 1.005, 2, 3))
    shape = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.3).clip(0)
    return 0.08 * tone * shape


def place(buf, sample, t_ms, gain=1.0):
    i = int(round(t_ms * SR / 1000))
    if i < 0 or i >= len(buf):
        return
    j = min(len(buf), i + len(sample))
    buf[i:j] += gain * sample[: j - i]


def make_song(rng, idx):
    style = STYLES[idx % len(STYLES)]
    duration = rng.uniform(80, 140) * 1000
    bpm1 = round(rng.uniform(80, 240), int(rng.integers(0, 3)))
    meter = 3 if rng.random() < 0.1 else 4
    offset = rng.uniform(100, 2500)
    segments = [{"time": offset, "beat_length": 60000 / bpm1, "meter": meter}]
    if rng.random() < 0.3:  # tempo change on a downbeat
        L = 60000 / bpm1
        bar = int(rng.integers(16, 40))
        bpm2 = round(bpm1 * rng.choice([0.75, 0.9, 1.1, 1.25, 1.5]) + rng.uniform(-3, 3), 1)
        segments.append({"time": offset + bar * meter * L, "beat_length": 60000 / bpm2, "meter": meter})
    buf = np.zeros(int((duration + 2000) * SR / 1000))
    hum = 0.0 if style in ("edm",) else 2.0  # ms of human jitter around the grid
    key = rng.uniform(110, 220)
    scale = key * 2 ** (np.array([0, 2, 3, 5, 7, 8, 10, 12]) / 12)
    intro_end = offset + 8 * meter * segments[0]["beat_length"] if style == "ambient_intro" else -1

    beat_times = []
    for k, seg in enumerate(segments):
        end = segments[k + 1]["time"] if k + 1 < len(segments) else duration
        L = seg["beat_length"]
        n = int(np.floor((end - seg["time"]) / L - 1e-9)) + 1
        beat_times += [(seg["time"] + b * L, b % seg["meter"], L) for b in range(n)]

    for t, pos, L in beat_times:
        j = lambda: rng.normal(0, hum) if hum else 0.0
        if style == "ambient_intro" and t < intro_end:
            if pos == 0:
                place(buf, pad(rng.choice(scale), L * 4 / 1000), t + j())
            continue
        if style in ("edm", "ambient_intro"):
            place(buf, kick(rng), t + j())
            if pos in (1, 3):
                place(buf, snare(rng), t + j(), 0.8)
            for h in (0.5,):
                place(buf, hat(rng), t + h * L + j())
            place(buf, pluck(rng.choice(scale) / 2, L / 1000 * 0.9), t + j(), 0.6)
        elif style == "rock":
            if pos in (0, 2) or rng.random() < 0.15:
                place(buf, kick(rng), t + j())
            if pos in (1, 3):
                place(buf, snare(rng), t + j())
            for h in (0, 0.5):
                place(buf, hat(rng), t + h * L + j(), 0.8)
            place(buf, pluck(rng.choice(scale) / 2, L / 1000 * 1.8, 0.004), t + j(), 0.7)
        elif style == "soft":
            # no drums at all: only note onsets with soft attacks
            if pos == 0:
                place(buf, pad(scale[0], L * seg_meter(segments, t) / 1000), t)
            for h in (0, 0.5) if rng.random() < 0.5 else (0,):
                place(buf, pluck(rng.choice(scale), L / 1000 * 1.5, 0.008), t + h * L + j())
        elif style == "syncopated":
            # house/trance-style: offbeat bass and loud offbeat stabs, kick only on 1 and 3
            if pos in (0, 2):
                place(buf, kick(rng), t + j())
            if pos == 1 and rng.random() < 0.4:
                place(buf, kick(rng) * 0.7, t + 0.75 * L + j())
            if pos in (1, 3):
                place(buf, snare(rng), t + j(), 0.8)
            if rng.random() < 0.3:
                place(buf, snare(rng) * 0.3, t + 0.25 * L + j())
            place(buf, hat(rng), t + 0.5 * L + j())
            place(buf, pluck(rng.choice(scale) / 2, L / 1000 * 0.4), t + 0.5 * L + j(), 0.8)
            if rng.random() < 0.6:
                place(buf, pluck(rng.choice(scale) * 2, L / 1000 * 0.6), t + 0.75 * L + j(), 0.9)
    buf /= np.abs(buf).max() * 1.1
    return style, buf, segments, duration


def seg_meter(segments, t):
    return [s for s in segments if s["time"] <= t + 1e-6][-1]["meter"] if any(s["time"] <= t + 1e-6 for s in segments) else segments[0]["meter"]


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def measure_shift(ref, path):
    """Shift (ms) of the BASS-decoded file relative to the source, via cross-correlation."""
    y, sr = decode_bass(path)
    assert sr == SR
    n = min(len(ref), len(y), SR * 20)
    a, b = ref[:n], y[:n]
    spec = np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n))
    cc = np.fft.irfft(spec)
    lag = int(np.argmax(np.concatenate([cc[-n:], cc[:n]]))) - n
    return lag * 1000 / SR


ENCODINGS = {
    "mp3_cbr192": ["lame", "--quiet", "-b", "192"],
    "mp3_v2": ["lame", "--quiet", "-V", "2"],
    "mp3_notag": ["lame", "--quiet", "-t", "-b", "192"],
    "ogg": None,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        style, x, segments, duration = make_song(rng, i)
        d = OUT / f"s{i:03d}"
        d.mkdir(exist_ok=True)
        write_wav(d / "src.wav", x)
        files = {}
        for name, cmd in ENCODINGS.items():
            ext = "ogg" if name == "ogg" else "mp3"
            f = d / f"{name}.{ext}"
            if cmd is None:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "src.wav"), "-c:a", "libvorbis", "-q:a", "6", str(f)], check=True)
            else:
                subprocess.run([*cmd, str(d / "src.wav"), str(f)], check=True)
            files[f.name] = round(measure_shift(x, f), 3)
        (d / "src.wav").unlink()
        (d / "truth.json").write_text(json.dumps({
            "style": style, "segments": segments, "span": [segments[0]["time"], duration], "shift_ms": files,
        }, indent=1))
        bpms = "→".join(f"{60000 / s['beat_length']:.2f}" for s in segments)
        print(f"s{i:03d} {style:14} {bpms:>16} bpm  offset {segments[0]['time']:7.1f}  shifts {files}")


if __name__ == "__main__":
    main()
