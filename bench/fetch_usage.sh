#!/bin/bash
# Copy the site's usage log (web/api.py usage(): a line per upload, refusal and fix) into data/usage/,
# for bench/usage.py. It lives in the api's docker volume on the home server, reached through
# the VPS like the deploy; the log only grows, so the copy here is replaced each time.
set -euo pipefail
cd "$(dirname "$0")/.."
VPS=vps  # a Host in ~/.ssh/config; no addresses in the repo
# the home server's ssh is the tunnel's own (its ExecStart without -NT and the forward)
KZ=$(ssh "$VPS" "sed -nE 's/^ExecStart=(.*) -NT (.*) -L [^ ]+ (.*)$/\1 \2 \3/p' /etc/systemd/system/osu-timing-tunnel.service")
mkdir -p data/usage
ssh "$VPS" "$KZ 'docker run --rm -v osu-timing-bot_usage:/d:ro --entrypoint tar osu-timing-bot_api -C /d -cz .'" \
  | tar -xz -C data/usage
echo "$(wc -l < data/usage/usage.jsonl) events in data/usage/usage.jsonl"
