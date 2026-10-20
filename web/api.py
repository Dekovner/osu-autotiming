"""HTTP API behind auneu.space/timing: audio in, red lines and a waveform out.

    python web/api.py            # listens on API_HOST:API_PORT (default 127.0.0.1:8431)

POST /analyze  (multipart: file=<audio>, optional downbeat=<ms>)
  -> {"lines": [{"time", "beat_length", "meter"}...], "phase_uncertain", "alt": [...],
      "duration_ms", "audio_lag_ms", "wave": {"ms_per_px": 1, "min": base64 int8, "max": base64 int8},
      "env": {"ms": 1, "data": base64 uint8}, "seconds"}

The waveform and the 1 ms energy envelope come from the BASS decode osu! uses,
so the page can line its own (browser) decode up with osu! time. Songs are
analysed in API_WORKERS niced processes; a small per-IP limit keeps one
visitor from filling the queue.
"""

import asyncio
import base64
import logging
import os
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aiohttp import web

log = logging.getLogger("timing-api")

MAX_BYTES = int(os.environ.get("MAX_BYTES", str(30 * 1024 * 1024)))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "15"))
WORKERS = int(os.environ.get("API_WORKERS", "2"))
PER_IP_PER_HOUR = int(os.environ.get("PER_IP_PER_HOUR", "60"))
AUDIO_EXT = {".mp3", ".ogg", ".wav"}


def _worker_init():
    os.nice(10)


def analyse(path: str, downbeat_ms: float | None) -> dict:
    """Runs in a worker process: plain data only."""
    import numpy as np

    from autotiming.decode import decode_bass
    from autotiming.detect import OSU_OFFSET_MS, NoRhythmError, analyze

    t0 = time.time()
    try:
        pcm, sr = decode_bass(path)
    except Exception as e:
        return {"error": f"couldn't read that file as audio ({e})"}
    minutes = len(pcm) / sr / 60
    if minutes > MAX_MINUTES:
        return {"error": f"that's {minutes:.1f} minutes; the limit is {MAX_MINUTES:g}"}
    # 1 ms min/max for drawing, 1 ms energy for lining the browser's decode up with this one
    spm = sr / 1000.0
    n = int(len(pcm) / spm)
    idx = (np.arange(n + 1) * spm).astype(np.int64)
    mn = np.minimum.reduceat(pcm[: idx[-1]], idx[:-1]) if n else np.zeros(0)
    mx = np.maximum.reduceat(pcm[: idx[-1]], idx[:-1]) if n else np.zeros(0)
    peak = max(float(np.abs(pcm).max()), 1e-9)
    q = lambda a: base64.b64encode(np.clip(np.round(a / peak * 127), -127, 127).astype(np.int8).tobytes()).decode()
    energy = np.add.reduceat(pcm[: idx[-1]].astype(np.float64) ** 2, idx[:-1]) if n else np.zeros(0)
    db = 10 * np.log10(energy / (energy.max() + 1e-12) + 1e-9)  # 0 dB = loudest ms
    env = base64.b64encode(np.clip((db + 80) * 255 / 80, 0, 255).astype(np.uint8).tobytes()).decode()
    try:
        r = analyze(path, downbeat_ms)
    except NoRhythmError:
        return {"error": "no rhythm found in that audio (silence or a drone?)"}
    seg = lambda ss: [{"time": s.time, "beat_length": s.beat_length, "meter": s.meter} for s in ss]
    return {
        "lines": seg(r.segments),
        "phase_uncertain": bool(r.phase_uncertain),
        "alt": seg(r.alt_segments or []),
        "duration_ms": len(pcm) / sr * 1000,
        "audio_lag_ms": -OSU_OFFSET_MS,  # a red line at t times the attack at t + this in the decode
        "wave": {"ms_per_px": 1, "min": q(mn), "max": q(mx)},
        "env": {"ms": 1, "data": env},
        "seconds": time.time() - t0,
    }


class Limiter:
    def __init__(self, per_hour: int):
        self.per_hour, self.hits = per_hour, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.time(), self.hits[key]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            return False
        q.append(now)
        return True


async def handle_analyze(request: web.Request) -> web.Response:
    app = request.app
    ip = request.headers.get("X-Real-IP") or request.remote or "?"
    if not app["limiter"].allow(ip):
        return web.json_response({"error": "too many songs from you this hour, try later"}, status=429)
    reader = await request.multipart()
    downbeat, path = None, None
    with tempfile.TemporaryDirectory() as tmp:
        async for part in reader:
            if part.name == "downbeat":
                txt = (await part.text()).strip()
                if txt:
                    try:
                        downbeat = float(txt)
                    except ValueError:
                        return web.json_response({"error": "downbeat must be a number of ms"}, status=400)
            elif part.name == "file":
                ext = Path(part.filename or "").suffix.lower()
                if ext not in AUDIO_EXT:
                    return web.json_response({"error": "send an mp3, ogg or wav file"}, status=400)
                path = Path(tmp) / f"audio{ext}"
                size = 0
                with path.open("wb") as f:
                    while chunk := await part.read_chunk(1 << 16):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            return web.json_response({"error": f"file too big (limit {MAX_BYTES >> 20} MB)"}, status=413)
                        f.write(chunk)
        if path is None:
            return web.json_response({"error": "no file"}, status=400)
        try:
            res = await asyncio.get_running_loop().run_in_executor(app["pool"], analyse, str(path), downbeat)
        except Exception:
            log.exception("analysis failed")
            res = {"error": "something went wrong on our side"}
    log.info("analyze from %s: %s", ip, res.get("error") or f"{len(res['lines'])} lines in {res['seconds']:.1f}s")
    return web.json_response(res, status=400 if "error" in res else 200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def make_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BYTES + (1 << 20))
    app["pool"] = ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init)
    app["limiter"] = Limiter(PER_IP_PER_HOUR)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    web.run_app(make_app(), host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8431")))
