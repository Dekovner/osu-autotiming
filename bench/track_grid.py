"""Try tracker settings on the DEV subset: tracked-beat accuracy and coverage per style.

    uv run python bench/track_grid.py '[{"TRACK_FPS": "200"}, {"TRACK_TLAMBDA": "30"}]' [workers]

Each setting is passed as environment variables to track_eval (beats cache is
keyed by the settings, so repeats are free).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPRESS = '{"sigma_ms": 6, "line_cost": 100}'


def run(env_over, workers):
    env = dict(os.environ, DEV="1", TRACK_ACT="prob", **env_over)
    py = [sys.executable, str(HERE / "track_eval.py")]
    subprocess.run(py + ["beats", str(workers)], env=env, check=True, capture_output=True)
    out = subprocess.run(py + ["sweep", str(workers), COMPRESS], env=env, check=True, capture_output=True, text=True).stdout
    acc = subprocess.run([sys.executable, str(HERE / "beat_accuracy.py")], env=env, check=True, capture_output=True, text=True).stdout
    print(f"=== {env_over or 'defaults'}")
    print("\n".join(l for l in out.splitlines() if l and not l.startswith("params")))
    print(acc.rstrip(), flush=True)


def main():
    configs = json.loads(sys.argv[1])
    workers = sys.argv[2] if len(sys.argv) > 2 else "12"
    for c in configs:
        run(c, workers)


if __name__ == "__main__":
    main()
