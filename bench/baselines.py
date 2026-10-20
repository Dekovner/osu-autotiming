"""Competitors, each reduced to what a mapper gets from them: a BPM and an offset.

Beat trackers return beat lists; we take their BPM and the circular-mean phase of
their beats, which is how such output ends up as a red line in practice.
"""

import numpy as np

from autotiming.grid import Segment


def _grid_from_beats(bpm: float, beats_s) -> list[Segment]:
    L = 60000.0 / bpm
    b = np.asarray(beats_s) * 1000.0
    ang = np.angle(np.exp(2j * np.pi * b / L).mean())
    return [Segment((ang / (2 * np.pi) * L) % L, L)]


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


def ours(path) -> list[Segment]:
    from autotiming.detect import detect

    return detect(path)


METHODS = {"librosa": librosa_beat, "essentia": essentia_rhythm, "ours": ours}
