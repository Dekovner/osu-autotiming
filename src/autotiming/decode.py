"""Audio decoding.

osu! (stable and lazer) decode through BASS with BASS_CONFIG_MP3_OLDGAPS=1
(see osu-framework AudioManager.cs). Other decoders (ffmpeg, librosa, ...) trim
MP3 encoder delay differently, which shifts every timestamp by a per-file amount.
Timing is only meaningful in osu!'s clock, so decode_bass is the reference.
"""

import ctypes
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

BASS_DIR = Path(__file__).resolve().parents[2] / "vendor" / "bass"

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
    lib.BASS_SetConfig(BASS_CONFIG_MP3_OLDGAPS, 1)
    if not lib.BASS_Init(0, 44100, 0, None, None) and lib.BASS_ErrorGetCode() != 14:  # 14 = already
        raise RuntimeError(f"BASS_Init failed: {lib.BASS_ErrorGetCode()}")
    return lib


def decode_bass(path) -> tuple[np.ndarray, int]:
    """Decode like osu! does. Returns (mono float32 samples, sample rate)."""
    lib = _bass()
    h = lib.BASS_StreamCreateFile(0, str(path).encode(), 0, 0, BASS_STREAM_DECODE | BASS_SAMPLE_FLOAT)
    if not h:
        raise RuntimeError(f"BASS can't open {path}: error {lib.BASS_ErrorGetCode()}")
    try:
        info = BassChannelInfo()
        lib.BASS_ChannelGetInfo(h, ctypes.byref(info))
        chunks = []
        buf = (ctypes.c_float * (1 << 18))()
        while True:
            n = lib.BASS_ChannelGetData(h, buf, ctypes.sizeof(buf) | BASS_DATA_FLOAT)
            if n == 0xFFFFFFFF or n == 0:
                break
            chunks.append(np.frombuffer(buf, dtype=np.float32, count=n // 4).copy())
    finally:
        lib.BASS_StreamFree(h)
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    pcm = pcm.reshape(-1, info.chans).mean(axis=1)
    return pcm.astype(np.float32), info.freq


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
