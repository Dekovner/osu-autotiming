"""Print what the bot would reply for a local file, without Discord.

    uv run --extra bot python bot/dry_run.py song.mp3 [downbeat_ms]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bot import analyse_file, build_reply  # noqa: E402


def main():
    path = sys.argv[1]
    downbeat = float(sys.argv[2]) if len(sys.argv) > 2 else None
    body, files = build_reply(Path(path).name, analyse_file(path, downbeat), downbeat)
    print(body)
    for f in files:
        print(f"\n--- attachment {f.filename} ---")
        print(f.fp.read().decode())


if __name__ == "__main__":
    main()
