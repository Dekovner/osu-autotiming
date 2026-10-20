"""Competitors, each reduced to what a mapper gets from them: a BPM and an offset.

Beat trackers return beat lists; we take their BPM and the circular-mean phase of
their beats, which is how such output ends up as a red line in practice.
"""

import os
from pathlib import Path

import numpy as np

from autotiming.grid import Segment


def _grid_from_beats(bpm: float, beats_s) -> list[Segment]:
    L = 60000.0 / bpm
    b = np.asarray(beats_s) * 1000.0
    ang = np.angle(np.exp(2j * np.pi * b / L).mean())
    return [Segment(float((ang / (2 * np.pi) * L) % L), float(L))]


def librosa_beat(path) -> list[Segment]:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return _grid_from_beats(float(np.atleast_1d(tempo)[0]), beats)


def essentia_rhythm(path) -> list[Segment]:
    import essentia
    import essentia.standard as es

    essentia.log.infoActive = False
    audio = es.MonoLoader(filename=str(path), sampleRate=44100)()
    bpm, beats, *_ = es.RhythmExtractor2013(method="multifeature")(audio)
    return _grid_from_beats(float(bpm), beats)


TA_DIR = Path(__file__).resolve().parents[1] / "vendor" / "timinganalyzer"
TA_PREFIX = Path(__file__).resolve().parents[1] / "data" / "wineprefix"


def timing_analyzer(path, timeout=180) -> list[Segment]:
    """Timing Analyzer v0.32.4 (the tool osu! mappers use), run under wine.

    It's an interactive console app: asks for a BPM (empty = auto), then draws
    "BPM: x" and "Offset: y" with console cursor moves, so it runs in a pty.
    The audio is copied under an ASCII name next to nothing else, since wine
    passes argv through untranslated.
    """
    import os
    import pty
    import re
    import select
    import shutil
    import subprocess
    import tempfile
    import time

    with tempfile.TemporaryDirectory(prefix="ta_") as tmp:
        audio = Path(tmp) / ("a" + Path(path).suffix.lower())
        shutil.copyfile(path, audio)
        env = dict(os.environ, WINEPREFIX=str(TA_PREFIX), WINEDEBUG="-all")
        master, slave = pty.openpty()
        p = subprocess.Popen(["wine", str(TA_DIR / "timinganlyz.exe"), str(audio)], cwd=TA_DIR,
                             stdin=slave, stdout=slave, stderr=slave, env=env)
        os.close(slave)
        out, prompt_at, last_key, t0 = "", None, 0.0, time.time()
        try:
            while time.time() - t0 < timeout:
                if select.select([master], [], [], 0.5)[0]:
                    try:
                        out += os.read(master, 65536).decode("utf-8", "replace")
                    except OSError:
                        break
                now = time.time()
                if prompt_at is None and "Input BPM" in out:
                    prompt_at = now
                # wine's console drops a key sent the instant the prompt shows: wait, and repeat until it starts
                if prompt_at and "Calculating" not in out and now - prompt_at > 1 and now - last_key > 3:
                    os.write(master, b"\r")
                    last_key = now
                m_bpm = re.search(r"BPM:\s*([-\d.]+)", out)
                m_off = re.search(r"Offset:\s*([-\d.]+)", out)
                if m_bpm and m_off and re.search(r"Offset:\s*[-\d.]+\s", out):
                    return [Segment(float(m_off.group(1)), 60000.0 / float(m_bpm.group(1)))]
        finally:
            p.kill()
            os.close(master)
    raise RuntimeError(f"Timing Analyzer gave no result: {out[-300:]!r}")


def ours(path) -> list[Segment]:
    from autotiming import detect as D

    # learned models that never saw this song (bench/octave_trees_exp.py, bench/phase_trees_exp.py crossfit)
    for env, name, attr in (("OCTAVE_CROSSFIT", "octave", "OCTAVE_TREES"), ("PHASE_CROSSFIT", "phase", "PHASE_TREES")):
        if os.environ.get(env) == "1":
            import hashlib
            import json

            cf = Path(__file__).resolve().parents[1] / "data" / f"{name}_crossfit"
            fold = json.loads((cf / "assign.json").read_text()).get(hashlib.md5(Path(path).read_bytes()).hexdigest())
            setattr(D, attr, cf / f"fold{fold}.npz" if fold is not None else Path(D.__file__).resolve().parent / "models" / f"{name}_trees.npz")
    return D.detect(path)


def ours_steady(path) -> list[Segment]:
    """The steady detector alone (before the beat tracker / hybrid)."""
    from autotiming.detect import detect

    return detect(path, steady_only=True)


def ours_fine(path) -> list[Segment]:
    from autotiming.detect import detect

    return detect(path, fine=True)


METHODS = {
    "librosa": librosa_beat, "essentia": essentia_rhythm, "timing_analyzer": timing_analyzer,
    "ours": ours, "ours_steady": ours_steady, "ours_fine": ours_fine,
}
