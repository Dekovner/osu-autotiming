"""Audio decoding.

osu! (stable and lazer) decode through BASS with BASS_CONFIG_MP3_OLDGAPS=1
(see osu-framework AudioManager.cs). Other decoders (ffmpeg, librosa, ...) trim
MP3 encoder delay differently, which shifts every timestamp by a per-file amount.
Timing is only meaningful in osu!'s clock, so decode_bass is the reference.
"""

import ctypes
import os
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

# libs/x86_64/libbass.so lives under this directory (the repo's vendor/bass, or AUTOTIMING_BASS_DIR in a deploy)
BASS_DIR = Path(os.environ.get("AUTOTIMING_BASS_DIR", Path(__file__).resolve().parents[2] / "vendor" / "bass"))

BASS_SAMPLE_FLOAT = 0x100
BASS_STREAM_DECODE = 0x200000
BASS_DATA_FLOAT = 0x40000000
BASS_CONFIG_MP3_OLDGAPS = 68


class BassChannelInfo(ctypes.Structure):
    _fields_ = [
        ("freq", ctypes.c_uint32),
        ("chans", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ctype", ctypes.c_uint32),
        ("origres", ctypes.c_uint32),
        ("plugin", ctypes.c_uint32),
        ("sample", ctypes.c_uint32),
        ("filename", ctypes.c_char_p),
    ]


@cache
def _bass():
    lib = ctypes.CDLL(str(BASS_DIR / "libs" / "x86_64" / "libbass.so"))
    lib.BASS_StreamCreateFile.restype = ctypes.c_uint32
    lib.BASS_StreamCreateFile.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32,
    ]
    lib.BASS_ChannelGetData.restype = ctypes.c_uint32
    lib.BASS_ChannelGetData.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    lib.BASS_ChannelGetInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(BassChannelInfo)]
    lib.BASS_StreamFree.argtypes = [ctypes.c_uint32]
    lib.BASS_ChannelGetLength.restype = ctypes.c_uint64
    lib.BASS_ChannelGetLength.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.BASS_SetConfig(BASS_CONFIG_MP3_OLDGAPS, 1)
    if not lib.BASS_Init(0, 44100, 0, None, None) and lib.BASS_ErrorGetCode() != 14:  # 14 = already
        raise RuntimeError(f"BASS_Init failed: {lib.BASS_ErrorGetCode()}")
    return lib


class TooLongError(ValueError):
    """The audio runs past decode_bass's max_seconds; seconds is BASS's estimate of its length."""

    def __init__(self, seconds: float):
        super().__init__(f"{seconds / 60:.1f} minutes of audio")
        self.seconds = seconds


def decode_bass(path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Decode like osu! does. Returns (mono float32 samples, sample rate).

    With max_seconds, longer audio raises TooLongError before it is decoded: a few MB
    of near-silent ogg hold hours of audio, gigabytes once decoded.
    """
    lib = _bass()
    h = lib.BASS_StreamCreateFile(0, str(path).encode(), 0, 0, BASS_STREAM_DECODE | BASS_SAMPLE_FLOAT)
    if not h:
        raise RuntimeError(f"BASS can't open {path}: error {lib.BASS_ErrorGetCode()}")
    try:
        info = BassChannelInfo()
        lib.BASS_ChannelGetInfo(h, ctypes.byref(info))
        # downmix each chunk straight into one preallocated mono array: holding the
        # interleaved stereo and its copies cost ~3x the song in memory
        n_bytes = lib.BASS_ChannelGetLength(h, 0)  # BASS_POS_BYTE; an estimate for some mp3s
        n_est = n_bytes // 4 // info.chans
        limit = None if max_seconds is None else int(max_seconds * info.freq)
        if limit is not None and n_est > limit:
            raise TooLongError(n_est / info.freq)
        out = np.empty(max(1, n_est + 1), np.float32)
        pos = 0
        buf = (ctypes.c_float * (1 << 18))()
        while True:
            n = lib.BASS_ChannelGetData(h, buf, ctypes.sizeof(buf) | BASS_DATA_FLOAT)
            if n == 0xFFFFFFFF or n == 0:
                break
            if limit is not None and pos > limit:  # the length estimate was short
                raise TooLongError(pos / info.freq)
            mono = np.frombuffer(buf, dtype=np.float32, count=n // 4).reshape(-1, info.chans).mean(axis=1)
            if pos + len(mono) > len(out):
                out = np.resize(out, max(2 * len(out), pos + len(mono)))
            out[pos:pos + len(mono)] = mono
            pos += len(mono)
    finally:
        lib.BASS_StreamFree(h)
    return out[:pos].astype(np.float32, copy=False), info.freq


def decode_ffmpeg(path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode with ffmpeg (what most generic tools effectively do)."""
    if sr is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        sr = int(probe.stdout.split()[0].strip(","))
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32), sr
