# osu-autotiming

Timing for osu! maps from the audio file: BPM, offset and red lines, printed as
`[TimingPoints]` lines you can paste into the editor.

- one steady grid fitted over the whole song, so the offset is consistent from the first note to the last
- tempo changes: sections, ramps and ritardandos via a beat tracker that only takes over where a steady grid can't hold the onsets
- red lines on downbeats, connected to each other (no double barlines), BPMs rounded the way mappers type them (150, not 150.009)
- decodes like osu! does (BASS, same MP3 gap handling), so offsets don't shift in game

## Accuracy

Benchmarked against the timing of 791 ranked maps (share of the mapped time where
the grid is within ±10 ms / ±5 ms of the ranked one). Ranked timing is not exact
truth: ±4 ms between mappers is normal, more on hard songs.

| maps (both tools ran on the same 501) | this | Timing Analyzer 0.32.4 |
|---|---|---|
| all | 82% / 63% | 73% / 57% |
| many BPM changes | 67% / 48% | 36% / 25% |
| a single red line | 88% / 69% | 87% / 69% |

Timing Analyzer gave no result on 50 of these maps (mostly multi-BPM ones); they
count as 0. On single-BPM songs both are at the noise floor of ranked timing.

## Use

```sh
uv sync --extra bot
# BASS is not redistributed: see vendor/bass/README.md
uv run python bot/dry_run.py song.mp3            # downbeat guessed
uv run python bot/dry_run.py song.mp3 1234       # a downbeat you know, in ms
```

From Python:

```python
from autotiming.detect import analyze
from autotiming.osufile import timing_lines

t = analyze("song.mp3")
print("\n".join(timing_lines(t.segments)))
```

## Discord bot

`bot/` is a Discord bot with a `/timing audio [downbeat]` command. Put
`DISCORD_TOKEN=...` in `bot/.env`, then `docker compose -p osu-timing-bot --profile discord up -d --build`
in `bot/` (without the profile only the site's `api` service starts). `BOT_WORKERS` sets how
many songs are analysed at once.

## Benchmark

`bench/` holds the evaluation: `fetch_ranked.py` downloads ranked sets,
`run.py ranked` runs the detector over them, `baselines.py` wraps librosa,
essentia and Timing Analyzer for comparison.
