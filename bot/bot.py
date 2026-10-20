"""Discord bot: /timing <audio> [downbeat] -> osu! red lines.

    DISCORD_TOKEN=... python bot/bot.py

One song is analysed at a time in a niced worker process (the VPS also runs a
VPN); others wait in line. Replies in English: the server is international.
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import discord
from discord import app_commands

log = logging.getLogger("redline")

MAX_BYTES = int(os.environ.get("MAX_BYTES", str(30 * 1024 * 1024)))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "15"))
AUDIO_EXT = {".mp3", ".ogg", ".wav"}
INLINE_LINES = 20  # longer timings come as a file instead of a code block


def _worker_init():
    os.nice(10)


def analyse_file(path: str, downbeat_ms: float | None) -> dict:
    """Runs in the worker process. Returns plain data for the reply."""
    from autotiming.decode import decode_bass
    from autotiming.detect import NoRhythmError, analyze
    from autotiming.osufile import timing_lines

    t0 = time.time()
    try:
        pcm, sr = decode_bass(path)
    except Exception as e:  # not audio BASS can read
        return {"error": f"Couldn't read that file as audio ({e})."}
    minutes = len(pcm) / sr / 60
    if minutes > MAX_MINUTES:
        return {"error": f"That's {minutes:.1f} minutes; the limit is {MAX_MINUTES:g}."}
    try:
        r = analyze(path, downbeat_ms)
    except NoRhythmError:
        return {"error": "I couldn't find any rhythm in that audio (silence or a drone?)."}
    return {
        "lines": timing_lines(r.segments),
        "sections": [(s.time, s.bpm) for s in r.segments],
        "phase_uncertain": r.phase_uncertain,
        "alt_lines": timing_lines(r.alt_segments) if r.phase_uncertain and r.alt_segments else [],
        "seconds": time.time() - t0,
        "minutes": minutes,
    }


def fmt_time(ms: float) -> str:
    m, s = divmod(max(ms, 0) / 1000, 60)
    return f"{int(m):02d}:{s:06.3f}"


def build_reply(name: str, res: dict, downbeat_ms: float | None) -> tuple[str, list[discord.File]]:
    if "error" in res:
        return f"**{name}**: {res['error']}", []
    secs = res["sections"]
    bpms = sorted({round(b, 2) for _, b in secs})
    head = (f"**{name}**: {len(secs)} red line{'s' if len(secs) != 1 else ''}, "
            + (f"{bpms[0]:g} BPM" if len(bpms) == 1 else f"{min(bpms):g}-{max(bpms):g} BPM")
            + f" (analysed in {res['seconds']:.1f}s)")
    notes = []
    if downbeat_ms is None:
        notes.append("Red lines sit on downbeats I guessed; if they land mid-bar, run again with `downbeat:` set to a downbeat in ms.")
    if res["phase_uncertain"]:
        notes.append("Beat phase is uncertain here: if the grid feels on the offbeat, try the alternative (+1/2 beat) in the file.")
    text = "[TimingPoints]\n" + "\n".join(res["lines"]) + "\n"
    files = []
    if res["alt_lines"]:
        text_alt = "[TimingPoints]\n" + "\n".join(res["alt_lines"]) + "\n"
        files.append(discord.File(io.BytesIO(text_alt.encode()), filename="timing_alt_half_beat.txt"))
    body = head
    if len(res["lines"]) <= INLINE_LINES:
        body += f"\n```ini\n{text}```"
    else:
        body += "\nTiming is long, see the file."
        files.insert(0, discord.File(io.BytesIO(text.encode()), filename="timing.txt"))
    if notes:
        body += "\n" + "\n".join(f"-# {n}" for n in notes)
    return body[:1990], files


class Redline(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
        self.pool = ProcessPoolExecutor(max_workers=1, initializer=_worker_init)
        self.waiting = 0

    async def setup_hook(self):
        guild_id = os.environ.get("DISCORD_GUILD_ID")
        if guild_id:  # instant command updates on the home server while developing
            g = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=g)
            await self.tree.sync(guild=g)
        await self.tree.sync()


client = Redline()


@client.tree.command(name="timing", description="Time a song for osu!: BPM, offset and red lines")
@app_commands.describe(audio="mp3 / ogg / wav of the song", downbeat="Optional: time (ms) of any downbeat, the '1' of a bar")
async def timing(interaction: discord.Interaction, audio: discord.Attachment, downbeat: float | None = None):
    ext = Path(audio.filename).suffix.lower()
    if ext not in AUDIO_EXT:
        await interaction.response.send_message("Send an mp3, ogg or wav file.", ephemeral=True)
        return
    if audio.size > MAX_BYTES:
        await interaction.response.send_message(f"That file is too big (limit {MAX_BYTES >> 20} MB).", ephemeral=True)
        return
    ahead = client.waiting
    client.waiting += 1
    await interaction.response.defer(thinking=True)
    if ahead:
        await interaction.followup.send(f"Queued behind {ahead} song{'s' if ahead > 1 else ''}...", ephemeral=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"audio{ext}"
            await audio.save(path)
            res = await asyncio.get_running_loop().run_in_executor(client.pool, analyse_file, str(path), downbeat)
    except Exception:
        log.exception("analysis failed for %s", audio.filename)
        res = {"error": "Something went wrong on my side; the owner has been told."}
    finally:
        client.waiting -= 1
    log.info("timing %r by %s (%s): %s", audio.filename, interaction.user, interaction.user.id,
             res.get("error") or f"{len(res['lines'])} lines in {res['seconds']:.1f}s")
    body, files = build_reply(audio.filename, res, downbeat)
    await interaction.followup.send(body, files=files)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client.run(os.environ["DISCORD_TOKEN"], log_handler=None)


if __name__ == "__main__":
    main()
