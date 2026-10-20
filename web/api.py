"""HTTP API behind auneu.space/timing: audio in, red lines and a waveform out.

    python web/api.py            # listens on API_HOST:API_PORT (default 127.0.0.1:8431)

POST /analyze  (multipart: file=<audio>, optional downbeat=<ms>)
  -> {"job": id, "status": "queued", "ahead": n, "running": m}   (answered at once)
GET /job/<id>
  -> {"status": "queued", "ahead", "running"} | {"status": "running"}
   | {"status": "done", "lines": [{"time", "beat_length", "meter"}...], "phase_uncertain", "alt": [...],
      "duration_ms", "audio_lag_ms", "wave": {"ms_per_px": 1, "min": base64 int8, "max": base64 int8},
      "env": {"ms": 1, "data": base64 uint8}, "seconds"}

The waveform and the 1 ms energy envelope come from the BASS decode osu! uses,
so the page can line its own (browser) decode up with osu! time. Songs wait
in one queue (the page shows its place in it) and are analysed in API_WORKERS
niced processes; a per-IP limit and a queue cap keep one visitor from filling it.
"""

import asyncio
import base64
import logging
import os
import secrets
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
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "30"))  # songs waiting; beyond this new uploads are turned away
JOB_TTL = 600.0  # seconds a finished result is kept for the page to fetch
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


class Job:
    def __init__(self, tmp: tempfile.TemporaryDirectory, path: Path, downbeat: float | None, ip: str):
        self.id = secrets.token_urlsafe(12)
        self.tmp, self.path, self.downbeat, self.ip = tmp, path, downbeat, ip
        self.status, self.result, self.done_at = "queued", None, None


async def handle_analyze(request: web.Request) -> web.Response:
    """Take the upload, queue it, answer at once with the job id and how many songs are ahead."""
    app = request.app
    ip = request.headers.get("X-Real-IP") or request.remote or "?"
    if len(app["waiting"]) >= MAX_QUEUE:
        return web.json_response({"error": "the queue is full right now, try again in a minute"}, status=503)
    if not app["limiter"].allow(ip):
        return web.json_response({"error": "too many songs from you this hour, try later"}, status=429)
    reader = await request.multipart()
    downbeat, path = None, None
    tmp = tempfile.TemporaryDirectory()
    try:
        async for part in reader:
            if part.name == "downbeat":
                txt = (await part.text()).strip()
                if txt:
                    try:
                        downbeat = float(txt)
                    except ValueError:
                        tmp.cleanup()
                        return web.json_response({"error": "downbeat must be a number of ms"}, status=400)
            elif part.name == "file":
                ext = Path(part.filename or "").suffix.lower()
                if ext not in AUDIO_EXT:
                    tmp.cleanup()
                    return web.json_response({"error": "send an mp3, ogg or wav file"}, status=400)
                path = Path(tmp.name) / f"audio{ext}"
                size = 0
                with path.open("wb") as f:
                    while chunk := await part.read_chunk(1 << 16):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            tmp.cleanup()
                            return web.json_response({"error": f"file too big (limit {MAX_BYTES >> 20} MB)"}, status=413)
                        f.write(chunk)
    except Exception:
        tmp.cleanup()
        raise
    if path is None:
        tmp.cleanup()
        return web.json_response({"error": "no file"}, status=400)
    job = Job(tmp, path, downbeat, ip)
    app["jobs"][job.id] = job
    app["waiting"].append(job)
    await app["queue"].put(job)
    return web.json_response({"job": job.id, **position(app, job)})


def position(app, job: Job) -> dict:
    """Where a job stands: songs queued before it, and how many are being analysed right now."""
    if job.status == "queued":
        return {"status": "queued", "ahead": app["waiting"].index(job), "running": app["running"]}
    return {"status": job.status}


async def handle_job(request: web.Request) -> web.Response:
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        return web.json_response({"error": "unknown or expired job, upload the song again"}, status=404)
    if job.status != "done":
        return web.json_response(position(request.app, job))
    res = job.result
    return web.json_response({"status": "done", **res}, status=400 if "error" in res else 200)


async def worker(app):
    """One of WORKERS loops: take the oldest queued song, analyse it in the process pool."""
    loop = asyncio.get_running_loop()
    while True:
        job = await app["queue"].get()
        app["waiting"].remove(job)
        app["running"] += 1
        job.status = "running"
        try:
            job.result = await loop.run_in_executor(app["pool"], analyse, str(job.path), job.downbeat)
        except Exception:
            log.exception("analysis failed")
            job.result = {"error": "something went wrong on our side"}
        finally:
            app["running"] -= 1
            job.status, job.done_at = "done", time.time()
            job.tmp.cleanup()
        r = job.result
        log.info("job from %s: %s", job.ip, r.get("error") or f"{len(r['lines'])} lines in {r['seconds']:.1f}s")


async def janitor(app):
    """Forget finished jobs after JOB_TTL seconds (the page fetches its result within a second or two)."""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for k in [k for k, j in app["jobs"].items() if j.done_at and now - j.done_at > JOB_TTL]:
            del app["jobs"][k]


async def start_background(app):
    app["tasks"] = [asyncio.create_task(worker(app)) for _ in range(WORKERS)] + [asyncio.create_task(janitor(app))]


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "queued": len(request.app["waiting"]), "running": request.app["running"]})


def make_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BYTES + (1 << 20))
    app["pool"] = ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init)
    app["limiter"] = Limiter(PER_IP_PER_HOUR)
    app["queue"], app["waiting"], app["running"], app["jobs"] = asyncio.Queue(), [], 0, {}
    app.on_startup.append(start_background)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_get("/job/{id}", handle_job)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    web.run_app(make_app(), host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8431")))
