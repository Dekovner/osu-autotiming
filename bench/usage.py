"""Who uses auneu.space/timing: the api's usage log (bench/fetch_usage.sh) day by day.

    USAGE_ME=1.2.3.4,5.6.7.8 uv run python bench/usage.py [--days N]

USAGE_ME: your own IPs (kept out of the repo), counted apart as "me". Visitors are the api's salted
IP hashes; the salt comes along in data/usage/.salt, so your IPs hash the same way here.
"""

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "data" / "usage"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="last N days")
    ap.add_argument("--dir", type=Path, default=DIR, help="where fetch_usage.sh put the log")
    args = ap.parse_args()
    salt = (args.dir / ".salt").read_text().strip()
    me = {hashlib.sha256((salt + ip.strip()).encode()).hexdigest()[:12]
          for ip in os.environ.get("USAGE_ME", "").split(",") if ip.strip()}
    since = time.time() - args.days * 86400
    evs = [e for e in map(json.loads, (args.dir / "usage.jsonl").read_text().splitlines()) if e["t"] >= since]
    days = defaultdict(list)
    for e in evs:
        days[time.strftime("%Y-%m-%d", time.localtime(e["t"]))].append(e)

    print(f"{'day':10}  {'visitors':>8}  {'songs':>5}  {'failed':>6}  {'refused':>7}  {'fixes':>5}  {'audio min':>9}  {'me':>4}")
    for day in sorted(days):
        them = [e for e in days[day] if e["who"] not in me]
        jobs = [e for e in them if e["ev"] == "job"]
        print(f"{day:10}  {len({e['who'] for e in them}):8}  {sum('err' not in e for e in jobs):5}"
              f"  {sum('err' in e for e in jobs):6}  {sum(e['ev'] == 'refused' for e in them):7}"
              f"  {sum(e['ev'] == 'fix' for e in them):5}  {sum(e.get('audio') or 0 for e in jobs) / 60:9.1f}"
              f"  {sum(e['ev'] == 'job' for e in days[day]) - len(jobs):4}")

    them = [e for e in evs if e["who"] not in me]
    top = Counter(e["who"] for e in them if e["ev"] == "job")
    if top:
        print(f"\nbusiest visitors ({len(top)} in all): " + ", ".join(f"{w} {n}" for w, n in top.most_common(8)))
    doubt = Counter(d for e in them if e["ev"] == "job" for d in e.get("doubt", []))
    ok = sum(e["ev"] == "job" and "err" not in e for e in them)
    if ok:
        print("songs with a doubt note: " + (", ".join(f"{d} {n / ok:.0%}" for d, n in doubt.most_common()) or "none"))
    errs = Counter(e["err"] for e in them if e.get("err"))
    for err, n in errs.most_common(5):
        print(f"  {n:3}x {err}")


if __name__ == "__main__":
    main()
