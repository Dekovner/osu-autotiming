# osu-autotiming

Timing for osu! maps from the audio file: BPM, offset and red lines, printed as
`[TimingPoints]` lines you can paste into the editor. Try it at
[auneu.space/timing](https://auneu.space/timing/).

- one steady grid fitted over the whole song, so the offset is consistent from the first note to the last
- tempo changes: sections, ramps and ritardandos via a beat tracker that only takes over where a steady grid can't hold the onsets
- red lines on downbeats ("1" from bass, chord changes and hit strength), connected to each other (no double barlines), BPMs rounded the way mappers type them (150, not 150.009)
- songs in 3: the meter is detected and lines sit on "1" of a bar of 3
- x2 / x½: trees pick the octave per section, and say so when they're unsure
- decodes like osu! does (BASS, same MP3 gap handling), so offsets don't shift in game

## Accuracy

Benchmarked against the timing of 791 ranked maps (share of the mapped time where
the grid is within ±10 ms / ±5 ms of the ranked one). Ranked timing is not exact
truth: ±4 ms between mappers is normal, more on hard songs.

| maps (both tools ran on the same 501) | this | Timing Analyzer 0.32.4 |
|---|---|---|
| all | 87% / 67% | 73% / 57% |
| many BPM changes | 73% / 54% | 36% / 25% |
| a single red line | 93% / 74% | 87% / 69% |

Timing Analyzer gave no result on 50 of these maps (mostly multi-BPM ones); they
count as 0. Where the grid matches the ranked one, 93% of the mapper's barlines are
"1" in ours too (meter-aware). Every number comes from crossfit runs: no song is
judged by trees trained on it.

## Site

[auneu.space/timing](https://auneu.space/timing/) (`web/`): drop an mp3, ogg or wav
(songs up to 15 min, marathons up to 60 min in a queue of their own) and get the red
lines on a waveform editor:

- playback with a metronome, speed and grid divisor; lines snap to the grid
- move lines, change BPM in fine steps, shift everything by a beat or half a beat, x2 / x½ where the octave is in doubt
- paste timing points (ctrl+v) or drop an .osu to compare with an existing map's red lines
- **tap mode** for stretches with no steady tempo (rubato, a live intro, fermatas): play and hit z / x on
  every beat. Your lag is measured against the song's clear hits, each tap moves onto the hit it was aimed
  at, and the taps get red lines the way a mapper draws them (a new one only where the tempo moves, at a
  round BPM where it fits) from the first tap to the last; the timing after them carries on.
  You decide where the beats are, the page only makes them precise
- **send fix**: the song and your corrected timing go to the bench set, which is how the detector learns what it gets wrong

`web/api.py` is the HTTP API behind it (queue, marathon limits, fixes, a usage log
with visitors as salted IP hashes); `web/static/` is the page.

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

`bench/` holds the evaluation: `fetch_ranked.py` downloads ranked sets (`fetch_heavy.py`
the ones with many red lines), `run.py ranked|heavy|nominated|feedback|site` runs the
detector over them, `table.py` compares coverage, `bar_eval.py` checks "1", and
`baselines.py` wraps librosa, essentia and Timing Analyzer for comparison.
