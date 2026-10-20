"""Build the ranked-map benchmark set from the catboy.best mirror.

    uv run python bench/fetch_ranked.py meta --pages 60
    uv run python bench/fetch_ranked.py pick --n 400
    uv run python bench/fetch_ranked.py download

Ranked timing is NOT exact truth (±4 ms is normal, ±10 ms on hard songs), so the
pick deliberately includes every Featured Artist track that was ranked more than
once: same audio, independent timers -> a direct measure of label noise.
"""

import argparse
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parents[1] / "data"
META = DATA / "ranked_meta.jsonl"
PICK = DATA / "pick.json"
SETS = DATA / "sets"
API = "https://catboy.best"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/140.0 osu-autotiming-bench"}
KEEP_SET = ("id", "artist", "title", "creator", "bpm", "genre_id", "language_id", "track_id", "ranked_date")


def cmd_meta(args):
    DATA.mkdir(exist_ok=True)
    seen = set()
    if META.exists():
        seen = {json.loads(l)["id"] for l in META.open()}
    with META.open("a") as out:
        for page in range(args.pages):
            r = requests.get(f"{API}/api/v2/search", params={
                "q": args.q, "status": 1, "limit": 100, "offset": page * 100, **({"mode": args.mode} if args.mode is not None else {}),
            }, headers=UA, timeout=60)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            new = 0
            for s in batch:
                if s["id"] in seen:
                    continue
                seen.add(s["id"])
                row = {k: s.get(k) for k in KEEP_SET}
                row["modes"] = sorted({b["mode_int"] for b in s["beatmaps"]})
                row["length"] = max(b["total_length"] for b in s["beatmaps"])
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                new += 1
            print(f"page {page}: {len(batch)} sets, {new} new, total {len(seen)}")
            time.sleep(1)


def cmd_pick(args):
    rows = [json.loads(l) for l in META.open()]
    rng = random.Random(args.seed)
    by_track = defaultdict(list)
    for r in rows:
        if r["track_id"]:
            by_track[r["track_id"]].append(r)
    # same track, different timers: only groups with distinct creators say anything about label noise
    groups = [g for g in by_track.values() if len({r["creator"] for r in g}) > 1]
    rng.shuffle(groups)
    dup = [r for g in groups[: args.dup_groups] for r in g[:4]]
    picked = {r["id"]: "dup_track" for r in dup}
    rest = [r for r in rows if r["id"] not in picked and 60 <= r["length"] <= 600]
    by_genre = defaultdict(list)
    for r in rest:
        by_genre[r["genre_id"]].append(r)
    for g in by_genre.values():
        rng.shuffle(g)
    # round-robin over genres so electronic doesn't swamp rock/live recordings
    while sum(v == "sample" for v in picked.values()) < args.n and any(by_genre.values()):
        for g in by_genre.values():
            if g:
                picked[g.pop()["id"]] = "sample"
    PICK.write_text(json.dumps(picked, indent=0))
    print(f"{len(rows)} sets in meta, {len(groups)} FA tracks timed by several mappers, "
          f"{len(dup)} sets from {min(len(groups), args.dup_groups)} of them; picked {len(picked)}")


def cmd_download(args):
    picked = json.loads(Path(args.pick).read_text()) if args.pick else json.loads(PICK.read_text())
    sets = Path(args.out) if args.out else SETS
    sets.mkdir(parents=True, exist_ok=True)
    todo = [int(i) for i in picked if not (sets / str(i) / "ok").exists()]
    print(f"{len(todo)} to download")
    for n, set_id in enumerate(todo, 1):
        try:
            r = requests.get(f"{API}/d/{set_id}n", headers=UA, timeout=180)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            d = sets / str(set_id)
            d.mkdir(exist_ok=True)
            names = z.namelist()
            audio = set()
            for name in names:
                if name.lower().endswith(".osu") and "/" not in name:
                    data = z.read(name)
                    (d / Path(name).name).write_bytes(data)
                    for line in data.decode("utf-8-sig", "replace").splitlines():
                        if line.startswith("AudioFilename:"):
                            audio.add(line.split(":", 1)[1].strip())
                            break
            lower = {n.lower(): n for n in names}
            for a in audio:
                if a.lower() in lower:
                    (d / a).write_bytes(z.read(lower[a.lower()]))
            (d / "ok").touch()
            print(f"[{n}/{len(todo)}] {set_id}: {len(r.content) >> 20} MB")
        except Exception as e:  # mirror hiccups shouldn't kill a long run
            print(f"[{n}/{len(todo)}] {set_id}: FAILED {e}")
        time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("meta")
    m.add_argument("--pages", type=int, default=50)
    m.add_argument("--q", default="")
    m.add_argument("--mode", type=int)
    pk = sub.add_parser("pick")
    pk.add_argument("--n", type=int, default=400)
    pk.add_argument("--seed", type=int, default=1)
    pk.add_argument("--dup-groups", type=int, default=150)
    dl = sub.add_parser("download")
    dl.add_argument("--pick", help="json list/dict of set ids (default: data/pick.json)")
    dl.add_argument("--out", help="directory for the sets (default: data/sets)")
    args = p.parse_args()
    {"meta": cmd_meta, "pick": cmd_pick, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    main()
