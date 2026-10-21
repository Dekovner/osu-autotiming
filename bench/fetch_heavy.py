"""More ranked sets with heavy timing (many red lines, real tempo changes).

    uv run python bench/fetch_heavy.py scan --pages 150     # search + one .osu header per new set
    uv run python bench/fetch_heavy.py pick --n 1000         # -> data/heavy_pick.json
    uv run python bench/fetch_ranked.py download --pick data/heavy_pick.json --out data/heavy

The search doesn't say how a set is timed, so scan reads one difficulty's .osu from
osu.ppy.sh up to [HitObjects] (the timing points are above it) and keeps the red lines.
Sets already in data/sets, nominated or feedback are skipped.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
SCAN = DATA / "heavy_scan.jsonl"
PICK = DATA / "heavy_pick.json"
API = "https://catboy.best"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/140.0 osu-autotiming-bench"}


def known_sets():
    have = set()
    for d in ("sets", "nominated", "feedback", "heavy"):
        if (DATA / d).exists():
            have |= {int(p.name) for p in (DATA / d).iterdir() if p.name.isdigit()}
    return have


def red_lines(bid):
    """[(time, beat_length, meter)] of one difficulty, reading only the header."""
    with requests.get(f"https://osu.ppy.sh/osu/{bid}", headers=UA, timeout=60, stream=True) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(16384):
            buf += chunk
            if b"[HitObjects]" in buf:
                break
    text = buf.decode("utf-8-sig", "replace")
    out, section = [], ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
            continue
        if section != "[TimingPoints]" or not line:
            continue
        f = line.split(",")
        try:
            uninherited = len(f) < 7 or f[6] == "1"
            if uninherited and float(f[1]) > 0:
                out.append((float(f[0]), float(f[1]), int(f[2]) if len(f) > 2 else 4))
        except ValueError:
            continue
    return out


def summary(lines):
    if not lines:
        return {}
    bl = np.array([b for _, b, _ in lines])
    # distinct tempos: lines more than 2% apart in beat length
    tempos = []
    for b in bl:
        if all(abs(np.log(b / t)) > 0.02 for t in tempos):
            tempos.append(b)
    jumps = int(np.sum(np.abs(np.diff(np.log(bl))) > 0.02)) if len(bl) > 1 else 0
    return {"n_red": len(lines), "n_tempo": len(tempos), "n_jump": jumps}


def cmd_scan(args):
    have = known_sets()
    done = set()
    if SCAN.exists():
        done = {json.loads(l)["id"] for l in SCAN.open()}
    lock = threading.Lock()
    out = SCAN.open("a")

    def one(s):
        maps = [b for b in s["beatmaps"] if 60 <= b["total_length"] <= 600]
        if not maps:
            return
        # osu!standard if there is one (the most played timing), the hardest difficulty
        std = [b for b in maps if b["mode_int"] == 0] or maps
        b = max(std, key=lambda b: b.get("difficulty_rating", 0))
        try:
            lines = red_lines(b["id"])
        except Exception as e:  # one bad request shouldn't stop a long scan
            print(f"  {s['id']}: {e}")
            return
        row = {"id": s["id"], "bid": b["id"], "mode": b["mode_int"], "length": b["total_length"],
               "artist": s["artist"], "title": s["title"], "creator": s["creator"], "genre_id": (s.get("genre") or {}).get("id"),
               **summary(lines), "lines": lines}
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    with ThreadPoolExecutor(args.threads) as ex:
        for page in range(args.start, args.start + args.pages):
            for attempt in range(5):  # the mirror's search times out now and then
                try:
                    r = requests.get(f"{API}/api/v2/search", params={"q": "", "status": 1, "limit": 100, "offset": page * 100},
                                     headers=UA, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    print(f"page {page}: {e}, retrying", flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                continue
            batch = [s for s in r.json() if s["id"] not in have and s["id"] not in done]
            if not r.json():
                break
            list(ex.map(one, batch))
            done |= {s["id"] for s in batch}
            print(f"page {page}: {len(batch)} new, scanned {len(done)}", flush=True)
            time.sleep(0.5)


def cmd_pick(args):
    rows = [json.loads(l) for l in SCAN.open()]
    rows = [r for r in rows if r.get("n_red")]
    rng = random.Random(args.seed)
    # tempo changes first (the goal), then many same-tempo resets
    heavy = [r for r in rows if r["n_jump"] >= 2]
    resets = [r for r in rows if r["n_jump"] < 2 and r["n_red"] > 5]
    rng.shuffle(heavy)
    rng.shuffle(resets)
    n_reset = min(len(resets), args.n // 5)
    picked = heavy[: args.n - n_reset] + resets[:n_reset]
    PICK.write_text(json.dumps({r["id"]: "heavy" if r["n_jump"] >= 2 else "resets" for r in picked}, indent=0))
    print(f"{len(rows)} scanned: {len(heavy)} with >=2 tempo jumps, {len(resets)} many-resets; picked {len(picked)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--pages", type=int, default=150)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--threads", type=int, default=4)
    pk = sub.add_parser("pick")
    pk.add_argument("--n", type=int, default=1000)
    pk.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    {"scan": cmd_scan, "pick": cmd_pick}[args.cmd](args)


if __name__ == "__main__":
    main()
