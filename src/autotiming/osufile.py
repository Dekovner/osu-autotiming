"""Minimal .osu reader: just what timing work needs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RedLine:
    time: float  # ms, osu! clock
    beat_length: float  # ms per beat
    meter: int

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length


@dataclass
class OsuFile:
    path: Path
    audio: str = ""
    mode: int = 0
    title: str = ""
    artist: str = ""
    creator: str = ""
    version: str = ""
    beatmap_id: int = 0
    set_id: int = 0
    red_lines: list[RedLine] = field(default_factory=list)
    first_object: float | None = None
    last_object: float | None = None


def read_osu(path) -> OsuFile:
    path = Path(path)
    osu = OsuFile(path)
    section = ""
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Metadata"):
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            match key:
                case "AudioFilename": osu.audio = value
                case "Mode": osu.mode = int(value)
                case "Title": osu.title = value
                case "Artist": osu.artist = value
                case "Creator": osu.creator = value
                case "Version": osu.version = value
                case "BeatmapID": osu.beatmap_id = int(value or 0)
                case "BeatmapSetID": osu.set_id = int(value or 0)
        elif section == "TimingPoints":
            parts = line.split(",")
            if len(parts) < 2:
                continue
            beat_length = float(parts[1])
            uninherited = parts[6].strip() == "1" if len(parts) > 6 else beat_length > 0
            if uninherited and beat_length > 0:
                meter = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 4
                osu.red_lines.append(RedLine(float(parts[0]), beat_length, meter))
        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) > 2:
                t = float(parts[2])
                if osu.first_object is None:
                    osu.first_object = t
                osu.last_object = t
    osu.red_lines.sort(key=lambda r: r.time)
    return osu
