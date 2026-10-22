"""HTTP API behind auneu.space/timing: audio in, red lines and a waveform out.

    python web/api.py            # listens on API_HOST:API_PORT (default 127.0.0.1:8431)

POST /analyze  (multipart: file=<audio>, optional downbeat=<ms>)
  -> {"job": id, "status": "queued", "ahead": n, "running": m, "marathon": bool}   (answered at once)
GET /job/<id>
  -> {"status": "queued", "ahead", "running", "marathon"} | {"status": "running", "marathon"}
   | {"status": "done", "lines": [{"time", "beat_length", "meter"}...], "phase_uncertain", "alt": [...],
      "downbeat_uncertain", "downbeat_alt": beats from "1" to the runner-up "1" (1..3) or null,
      "octave_alts": [{"lines": [index...], "factor": 2 | 0.5}] red lines whose tempo is in doubt,
      "triplet_alt": [lines] a song in 3 read the other way (waltz vs 4/4 in triplets), or [],
      "duration_ms", "audio_lag_ms", "wave": {"ms_per_px": 1, "min": base64 int8, "max": base64 int8},
      "env": {"ms": 1, "data": base64 uint8}, "hits": {"t": base64 int32 LE, osu ms x 10,
      "r": base64 uint8, 0..255 = local strength rank 0.5..1} (the clear hits tap mode locks onto), "seconds"}
POST /feedback  (multipart: file=<audio>, timing={"original": lines, "fixed": lines, "file", "note"})
  -> {"ok": true}   a visitor's corrected timing, kept with the song in FEEDBACK_DIR

Every upload, refusal and fix also adds a line to USAGE_DIR/usage.jsonl (visitors as a salted
hash of their IP): who uses the site outlives a redeploy, unlike the container's log.

The waveform and the 1 ms energy envelope come from the BASS decode osu! uses,
so the page can line its own (browser) decode up with osu! time. Songs wait
in one queue (the page shows its place in it) and are analysed in API_WORKERS
niced processes; per-IP limits (songs an hour, songs waiting at once) and a queue
cap keep one visitor from filling it.

Marathons (longer than MAX_MINUTES, up to MARATHON_MINUTES; files up to
MARATHON_BYTES) wait in a queue of their own and run MARATHON_WORKERS at a time in
their own processes on MARATHON_THREADS cores each, so songs never wait behind one
and an hour of audio (~2 GB while analysed) can't run six times over; each visitor
gets MARATHON_PER_IP_PER_DAY.
Length, not size, makes a marathon (a 5 minute wav is 50 MB): a small probe process
reads each upload's length from BASS (no decoding, milliseconds) and queues it where
it belongs; should BASS read an mp3 short, the song worker's decode stops at the
limit and hands it over. A file over MAX_BYTES from a visitor with no marathons left
today is turned away while it uploads, rather than after 160 MB went through the tunnel.
"""

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import secrets
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from aiohttp import web

log = logging.getLogger("timing-api")

MAX_BYTES = int(os.environ.get("MAX_BYTES", str(30 * 1024 * 1024)))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "15"))
WORKERS = int(os.environ.get("API_WORKERS", "2"))
PER_IP_PER_HOUR = int(os.environ.get("PER_IP_PER_HOUR", "60"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "30"))  # songs waiting; beyond this new uploads are turned away
PER_IP_PENDING = int(os.environ.get("PER_IP_PENDING", "3"))  # one visitor's songs queued or running at once
MARATHON_BYTES = int(os.environ.get("MARATHON_BYTES", str(160 * 1024 * 1024)))
MARATHON_MINUTES = float(os.environ.get("MARATHON_MINUTES", "60"))
MARATHON_WORKERS = int(os.environ.get("MARATHON_WORKERS", "1"))
MARATHON_THREADS = int(os.environ.get("MARATHON_THREADS", "4"))  # cores each marathon's tempo search spreads over
MARATHON_QUEUE = int(os.environ.get("MARATHON_QUEUE", "5"))
MARATHON_PER_IP_PER_DAY = int(os.environ.get("MARATHON_PER_IP_PER_DAY", "3"))
JOB_TTL = 600.0  # seconds a finished result is kept for the page to fetch
AUDIO_EXT = {".mp3", ".ogg", ".wav"}
# fixes visitors send from the editor (POST /feedback): the song and their corrected timing, a
# bench set of the songs the detector gets wrong (bench/run.py site). Songs only, no marathons.
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/data/feedback"))
FEEDBACK_PER_IP_PER_DAY = int(os.environ.get("FEEDBACK_PER_IP_PER_DAY", "10"))
FEEDBACK_MAX_TOTAL = int(os.environ.get("FEEDBACK_MAX_TOTAL", str(3 << 30)))  # bytes kept at most, then refused
FEEDBACK_MAX_LINES = 2000
FEEDBACK_MAX_NOTE = 1000
# one JSON line per upload, refusal and fix (who uses the site, and how), in a volume of its own so
# a redeploy keeps it; visitors are a salted hash of their IP (bench/fetch_usage.sh, bench/usage.py)
USAGE_DIR = Path(os.environ.get("USAGE_DIR", "/data/usage"))


def _worker_init(threads: int = 1):
    """Songs run API_WORKERS side by side on one thread each; a marathon gets several."""
    import numba

    os.nice(10)
    numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))


def probe_seconds(path: str) -> float | None:
    """Runs in the probe process: the upload's length, None if BASS can't open it (the song
    worker then says so)."""
    from autotiming.decode import bass_seconds

    try:
        return bass_seconds(path)
    except Exception:
        return None


PROBE_TIMEOUT = 10.0  # seconds; reading a length takes milliseconds


async def probe(app, path: Path) -> float | None:
    """Length of an upload from the probe process, which strangers' files may crash or hang:
    then it is replaced, and the upload goes the song queue's way (whose decode checks too)."""
    try:
        return await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(app["probe"], probe_seconds, str(path)),
                                      PROBE_TIMEOUT)
    except (BrokenProcessPool, asyncio.TimeoutError):
        log.warning("probe process lost on an upload, starting a new one")
        old, app["probe"] = app["probe"], ProcessPoolExecutor(max_workers=1)
        for proc in list(getattr(old, "_processes", {}).values()):
            proc.kill()
        old.shutdown(wait=False, cancel_futures=True)
        return None


def analyse(path: str, downbeat_ms: float | None, max_minutes: float, marathon_minutes: float | None = None) -> dict:
    """Runs in a worker process: plain data only. Audio longer than max_minutes but within
    marathon_minutes comes back as {"marathon": seconds}, before it is decoded."""
    import numpy as np

    from autotiming.decode import TooLongError, decode_bass
    from autotiming.detect import OSU_OFFSET_MS, NoRhythmError, analyze

    t0 = time.time()
    try:
        pcm, sr = decode_bass(path, max_seconds=max_minutes * 60)
    except TooLongError as e:
        if marathon_minutes is not None and e.seconds <= marathon_minutes * 60:
            return {"marathon": e.seconds}
        return {"error": f"that's {e.seconds / 60:.1f} minutes; the limit is {marathon_minutes or max_minutes:g}"}
    except Exception as e:
        log.info("undecodable upload: %s", e)
        return {"error": "couldn't read that file as audio"}
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
    del energy, db
    try:
        r = analyze(path, downbeat_ms, decoded=(pcm, sr))
    except NoRhythmError:
        return {"error": "no rhythm found in that audio (silence or a drone?)"}
    seg = lambda ss: [{"time": s.time, "beat_length": s.beat_length, "meter": s.meter} for s in ss]
    ht, hr = r.hits if r.hits is not None else (np.zeros(0), np.zeros(0))
    hits = {"t": base64.b64encode(np.round(ht * 10).astype("<i4").tobytes()).decode(),
            "r": base64.b64encode(np.clip(np.round((hr - 0.5) * 510), 0, 255).astype(np.uint8).tobytes()).decode()}
    return {
        "lines": seg(r.segments),
        "phase_uncertain": bool(r.phase_uncertain),
        "downbeat_uncertain": bool(r.downbeat_uncertain),
        "downbeat_alt": r.downbeat_alt,  # the runner-up "1", in beats after the chosen one
        "octave_alts": r.octave_alts or [],  # tempo x2 / x0.5 in doubt: which red lines, and the factor
        "alt": seg(r.alt_segments or []),
        "triplet_alt": seg(r.triplet_alt or []),  # in 3: the waltz / triplet reading we didn't pick
        "duration_ms": len(pcm) / sr * 1000,
        "audio_lag_ms": -OSU_OFFSET_MS,  # a red line at t times the attack at t + this in the decode
        "wave": {"ms_per_px": 1, "min": q(mn), "max": q(mx)},
        "env": {"ms": 1, "data": env},
        "hits": hits,  # tap mode locks taps onto these
        "seconds": time.time() - t0,
    }


def visitor(ip: str) -> str:
    """The same visitor gives the same 12 hex digits; the salt never leaves the usage volume."""
    salt = USAGE_DIR / ".salt"
    if not salt.exists():
        salt.write_text(secrets.token_hex(16))
    return hashlib.sha256((salt.read_text().strip() + ip).encode()).hexdigest()[:12]


def usage(ev: str, ip: str, **kw):
    """Append one event to USAGE_DIR/usage.jsonl; a full disk or a missing volume never fails a request."""
    try:
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        with (USAGE_DIR / "usage.jsonl").open("a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "ev": ev, "who": visitor(ip), **kw}) + "\n")
    except OSError as e:
        log.warning("usage log: %s", e)


class Limiter:
    def __init__(self, limit: int, window: float = 3600.0):
        self.limit, self.window, self.hits = limit, window, defaultdict(deque)

    def _recent(self, key: str) -> deque:
        now, q = time.time(), self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def left(self, key: str) -> int:
        return self.limit - len(self._recent(key))

    def allow(self, key: str) -> bool:
        q = self._recent(key)
        if len(q) >= self.limit:
            return False
        q.append(time.time())
        return True

    def prune(self):
        now = time.time()
        for k in [k for k, q in self.hits.items() if not q or now - q[-1] > self.window]:
            del self.hits[k]


class Job:
    def __init__(self, tmp: tempfile.TemporaryDirectory, path: Path, downbeat: float | None, ip: str):
        self.id = secrets.token_urlsafe(12)
        self.tmp, self.path, self.downbeat, self.ip = tmp, path, downbeat, ip
        self.status, self.result, self.done_at, self.marathon = "queued", None, None, False
        self.seconds = None  # the upload's length as the probe read it


def refusal(app, ip: str) -> tuple[int, str] | None:
    """Turn an upload away when the queue is full or this visitor already has songs in it."""
    if len(app["waiting"]) >= MAX_QUEUE:
        return 503, "the queue is full right now, try again in a minute"
    if sum(j.ip == ip and j.status != "done" for j in app["jobs"].values()) >= PER_IP_PENDING:
        return 429, f"you already have {PER_IP_PENDING} songs in the queue, wait for them"
    return None


def marathon_refusal(app, ip: str) -> tuple[int, str] | None:
    """Turn a marathon away when their queue is full or this visitor has had today's."""
    if len(app["mwaiting"]) >= MARATHON_QUEUE:
        return 503, "too many marathons waiting right now, try again in a few minutes"
    if app["mlimiter"].left(ip) <= 0:
        return 429, f"marathons (over {MAX_MINUTES:g} min, or files over {MAX_BYTES >> 20} MB) are " \
                    f"{MARATHON_PER_IP_PER_DAY} a day each and you've had yours: try tomorrow"
    return None


def refuse(r: tuple[int, str]) -> web.Response:
    return web.json_response({"error": r[1]}, status=r[0])


def enqueue(app, job: Job, marathon: bool):
    job.status, job.marathon = "queued", marathon
    if marathon:
        app["mlimiter"].allow(job.ip)  # marathon_refusal() was asked first: there is one left
    (app["mwaiting"] if marathon else app["waiting"]).append(job)
    (app["mqueue"] if marathon else app["queue"]).put_nowait(job)


async def handle_analyze(request: web.Request) -> web.Response:
    """Take the upload, queue it, answer at once with the job id and how many songs are ahead."""
    ip = request.headers.get("X-Real-IP") or request.remote or "?"
    resp = await take_upload(request, ip)
    if resp.status >= 400:  # turned away before the queue: the worker logs the rest
        usage("refused", ip, status=resp.status, err=json.loads(resp.text).get("error"))
    return resp


async def take_upload(request: web.Request, ip: str) -> web.Response:
    app = request.app
    if (r := refusal(app, ip)) is not None:
        return refuse(r)
    if not app["limiter"].allow(ip):
        return web.json_response({"error": "too many songs from you this hour, try later"}, status=429)
    reader = await request.multipart()
    downbeat, path, big = None, None, False
    tmp = tempfile.TemporaryDirectory()
    try:
        async for part in reader:
            if part.name == "downbeat":
                txt = (await part.text()).strip()
                if txt:
                    try:
                        downbeat = float(txt)
                        if not math.isfinite(downbeat):
                            raise ValueError(txt)
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
                        if size > MARATHON_BYTES:
                            tmp.cleanup()
                            return web.json_response({"error": f"file too big (limit {MARATHON_BYTES >> 20} MB)"}, status=413)
                        if size > MAX_BYTES and not big:  # most likely a marathon: may this visitor have one?
                            big = True
                            if (r := marathon_refusal(app, ip)) is not None:
                                tmp.cleanup()
                                return refuse(r)
                        f.write(chunk)
    except Exception:
        tmp.cleanup()
        raise
    if path is None:
        tmp.cleanup()
        return web.json_response({"error": "no file"}, status=400)
    if (r := refusal(app, ip)) is not None:  # again: parallel uploads all passed the first check
        tmp.cleanup()
        return refuse(r)
    seconds = await probe(app, path)
    marathon = seconds is not None and seconds > MAX_MINUTES * 60
    if seconds is not None and seconds > MARATHON_MINUTES * 60:
        tmp.cleanup()
        return web.json_response({"error": f"that's {seconds / 60:.1f} minutes; the limit is {MARATHON_MINUTES:g}"}, status=413)
    if marathon and (r := marathon_refusal(app, ip)) is not None:
        tmp.cleanup()
        return refuse(r)
    job = Job(tmp, path, downbeat, ip)
    job.seconds = seconds
    app["jobs"][job.id] = job
    enqueue(app, job, marathon)
    return web.json_response({"job": job.id, **position(app, job)})


def position(app, job: Job) -> dict:
    """Where a job stands: songs queued before it (in its own queue), and how many of its kind are being analysed."""
    if job.status == "queued":
        waiting = app["mwaiting"] if job.marathon else app["waiting"]
        return {"status": "queued", "ahead": waiting.index(job), "running": app["mrunning" if job.marathon else "running"],
                "marathon": job.marathon}
    return {"status": job.status, "marathon": job.marathon}


async def handle_job(request: web.Request) -> web.Response:
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        return web.json_response({"error": "unknown or expired job, upload the song again"}, status=404)
    if job.status != "done":
        return web.json_response(position(request.app, job))
    res = job.result
    return web.json_response({"status": "done", **res}, status=400 if "error" in res else 200)


def clean_lines(lines) -> list[dict] | None:
    """Red lines from the page as plain numbers, None if they don't look like a timing."""
    if not isinstance(lines, list) or not 0 < len(lines) <= FEEDBACK_MAX_LINES:
        return None
    out = []
    for l in lines:
        try:
            t, L, m = float(l["time"]), float(l["beat_length"]), int(l["meter"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(t) and abs(t) < 4e6 and math.isfinite(L) and 5 <= L <= 10000 and 1 <= m <= 32):
            return None
        out.append({"time": t, "beat_length": L, "meter": m})
    return out


def feedback_bytes() -> int:
    return sum(f.stat().st_size for f in FEEDBACK_DIR.rglob("*") if f.is_file()) if FEEDBACK_DIR.exists() else 0


async def handle_feedback(request: web.Request) -> web.Response:
    """A visitor's fix: the song plus the timing we gave and the one they corrected it to.
    Kept as <audio md5>/audio.<ext> and one fix-*.json per sending (the same song twice is stored once)."""
    app = request.app
    ip = request.headers.get("X-Real-IP") or request.remote or "?"
    if app["flimiter"].left(ip) <= 0:
        return web.json_response({"error": f"that's {FEEDBACK_PER_IP_PER_DAY} fixes today, thanks! send more tomorrow"}, status=429)
    if feedback_bytes() > FEEDBACK_MAX_TOTAL:
        return web.json_response({"error": "the fix box is full for now, thanks anyway"}, status=503)
    reader = await request.multipart()
    meta, path = None, None
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=FEEDBACK_DIR, prefix=".upload-")  # same filesystem: kept by a rename
    try:
        async for part in reader:
            if part.name == "timing":
                try:
                    meta = json.loads(await part.text())
                except ValueError:
                    return web.json_response({"error": "bad timing"}, status=400)
            elif part.name == "file":
                ext = Path(part.filename or "").suffix.lower()
                if ext not in AUDIO_EXT:
                    return web.json_response({"error": "send an mp3, ogg or wav file"}, status=400)
                path, size, md5 = Path(tmp.name) / f"audio{ext}", 0, hashlib.md5()
                with path.open("wb") as f:
                    while chunk := await part.read_chunk(1 << 16):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            return web.json_response({"error": f"fixes are for songs up to {MAX_BYTES >> 20} MB"}, status=413)
                        md5.update(chunk)
                        f.write(chunk)
        if path is None or not isinstance(meta, dict):
            return web.json_response({"error": "no file or no timing"}, status=400)
        original, fixed = clean_lines(meta.get("original")), clean_lines(meta.get("fixed"))
        if original is None or fixed is None:
            return web.json_response({"error": "bad timing"}, status=400)
        seconds = await probe(app, path)
        if seconds is None or seconds > MAX_MINUTES * 60:
            return web.json_response({"error": "that file isn't a song we can read (up to 15 min)"}, status=400)
        d = FEEDBACK_DIR / md5.hexdigest()
        d.mkdir(parents=True, exist_ok=True)
        if not any(d.glob("audio.*")):
            path.replace(d / path.name)
        note = str(meta.get("note") or "")[:FEEDBACK_MAX_NOTE]
        name = str(meta.get("file") or "")[:200]
        rec = {"time": time.time(), "file": name, "seconds": seconds, "note": note, "original": original, "fixed": fixed}
        out = d / f"fix-{int(time.time())}-{secrets.token_hex(3)}.json"
        out.with_suffix(".tmp").write_text(json.dumps(rec))
        out.with_suffix(".tmp").replace(out)
        app["flimiter"].allow(ip)  # only fixes that were kept count
        usage("fix", ip, song=d.name, audio=round(seconds, 1), lines=[len(original), len(fixed)], note=bool(note))
        log.info("fix received: %s (%d -> %d lines)%s", d.name, len(original), len(fixed), f": {note[:80]!r}" if note else "")
        return web.json_response({"ok": True})
    finally:
        tmp.cleanup()


async def worker(app, marathon: bool = False):
    """One of WORKERS (or MARATHON_WORKERS) loops: take the oldest queued song, analyse it in its process pool."""
    loop = asyncio.get_running_loop()
    queue, waiting, pool = (app["mqueue"], app["mwaiting"], app["mpool"]) if marathon else (app["queue"], app["waiting"], app["pool"])
    running = "mrunning" if marathon else "running"
    limits = (MARATHON_MINUTES,) if marathon else (MAX_MINUTES, MARATHON_MINUTES)
    while True:
        job = await queue.get()
        waiting.remove(job)
        app[running] += 1
        job.status = "running"
        try:
            job.result = await loop.run_in_executor(pool, analyse, str(job.path), job.downbeat, *limits)
        except Exception:
            log.exception("analysis failed")
            job.result = {"error": "something went wrong on our side"}
        finally:
            app[running] -= 1
        r = job.result
        if "marathon" in r:  # longer than a song: over to the marathon queue, if this visitor may have one
            if (refused := marathon_refusal(app, job.ip)) is None:
                log.info("job from %s: %.1f min, moved to the marathon queue", job.ip, r["marathon"] / 60)
                enqueue(app, job, True)
                continue
            r = job.result = {"error": refused[1]}
        job.status, job.done_at = "done", time.time()
        job.tmp.cleanup()
        log.info("%s from %s: %s", "marathon" if marathon else "job", job.ip,
                 r.get("error") or f"{len(r['lines'])} lines in {r['seconds']:.1f}s")
        audio = round(job.seconds, 1) if job.seconds is not None else None
        if "error" in r:
            usage("job", job.ip, marathon=marathon, audio=audio, err=r["error"])
        else:
            usage("job", job.ip, marathon=marathon, audio=audio, lines=len(r["lines"]), secs=round(r["seconds"], 1),
                  downbeat=job.downbeat is not None, doubt=[k for k, v in (("phase", r["phase_uncertain"]),
                  ("downbeat", r["downbeat_uncertain"]), ("octave", r["octave_alts"])) if v])


async def janitor(app):
    """Forget finished jobs after JOB_TTL seconds (the page fetches its result within a second or two)."""
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for k in [k for k, j in app["jobs"].items() if j.done_at and now - j.done_at > JOB_TTL]:
            del app["jobs"][k]
        app["limiter"].prune()
        app["mlimiter"].prune()
        app["flimiter"].prune()


async def start_background(app):
    app["tasks"] = ([asyncio.create_task(worker(app)) for _ in range(WORKERS)]
                    + [asyncio.create_task(worker(app, marathon=True)) for _ in range(MARATHON_WORKERS)]
                    + [asyncio.create_task(janitor(app))])


async def handle_health(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response({"ok": True, "queued": len(app["waiting"]), "running": app["running"],
                              "marathons_queued": len(app["mwaiting"]), "marathons_running": app["mrunning"]})


def make_app() -> web.Application:
    app = web.Application(client_max_size=MARATHON_BYTES + (1 << 20))
    app["pool"] = ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init)
    app["mpool"] = ProcessPoolExecutor(max_workers=MARATHON_WORKERS, initializer=_worker_init, initargs=(MARATHON_THREADS,))
    app["probe"] = ProcessPoolExecutor(max_workers=1)
    app["limiter"] = Limiter(PER_IP_PER_HOUR)
    app["mlimiter"] = Limiter(MARATHON_PER_IP_PER_DAY, 86400.0)
    app["flimiter"] = Limiter(FEEDBACK_PER_IP_PER_DAY, 86400.0)
    app["queue"], app["waiting"], app["running"], app["jobs"] = asyncio.Queue(), [], 0, {}
    app["mqueue"], app["mwaiting"], app["mrunning"] = asyncio.Queue(), [], 0
    app.on_startup.append(start_background)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_get("/job/{id}", handle_job)
    app.router.add_post("/feedback", handle_feedback)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    web.run_app(make_app(), host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8431")))
