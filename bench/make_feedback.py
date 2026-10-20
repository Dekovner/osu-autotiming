"""Maps that mappers tested the bot on by hand and reported, as a bench dataset.

    uv run python bench/make_feedback.py      # then: bench/run.py feedback

Each set's .osu and audio files are symlinked from the osu! Songs folder into
data/feedback/<set id>/, so the game's folder is never written to.
"""

from pathlib import Path

SONGS = Path.home() / ".local/share/osu-wine/osu!/Songs"
OUT = Path(__file__).resolve().parents[1] / "data" / "feedback"

# set folder -> what was wrong with the bot's output (2026-10-20 manual tests)
MAPS = {
    "971207 DJ Nanasaki - Unorthodox Red": "stepped slowdown 210 -> 150 missed, then a half-beat phase reset",
    "2034952 1 800 PAIN - OUTSIDE": "junk 171 line in the intro; outro 256 instead of 128",
    "2166086 TAICHU, rusowsky - PAYDAY": "short beat-by-beat ritardando 190/180/170 into 162.5 missed",
    "2347284 Supire - Dead Aire": "11 lines on a single 172 song",
    "2464036 HOYO-MiX - Monodrama": "185.031 not rounded; 170/165 transition missed",
    "2614557 t+pazolite - Bouquet with Hatred & Ugliness": "many tempo changes, octave chaos",
    "2491580 QUOK - SCHAST'E LJUBIT TISHINU 2": "steady 140 (control)",
    "2478028 ReeK - Galaxy Collision": "long ramps 200 -> 150 and 156 -> 200 not tracked",
    "2589354 Mr Bill - Pastel (voljum Remix) (Cut Ver)": "syncopated: grid 3/16 beat early",
}


def main():
    for name in MAPS:
        src = SONGS / name
        dst = OUT / name.split()[0]
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix.lower() in (".osu", ".mp3", ".ogg", ".wav") and not (dst / f.name).exists():
                (dst / f.name).symlink_to(f)
        (dst / "ok").touch()
        print(dst, len(list(dst.iterdir())) - 1, "files")


if __name__ == "__main__":
    main()
