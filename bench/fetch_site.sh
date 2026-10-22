#!/bin/bash
# Copy the fixes visitors sent from auneu.space/timing (web/api.py POST /feedback) into data/site/,
# for `bench/run.py site`. They live in the api's docker volume on the home server, reached
# through the VPS like the deploy. Files already here are kept; run it again to get new ones.
set -euo pipefail
cd "$(dirname "$0")/.."
VPS=vps  # a Host in ~/.ssh/config; no addresses in the repo
# the home server's ssh is the tunnel's own (its ExecStart without -NT and the forward)
KZ=$(ssh "$VPS" "sed -nE 's/^ExecStart=(.*) -NT (.*) -L [^ ]+ (.*)$/\1 \2 \3/p' /etc/systemd/system/osu-timing-tunnel.service")
mkdir -p data/site
# the api image has tar; the volume is mounted read-only into a throwaway container
ssh "$VPS" "$KZ 'docker run --rm -v osu-timing-bot_feedback:/d:ro --entrypoint tar osu-timing-bot_api -C /d -cz .'" \
  | tar -xz -C data/site --skip-old-files
echo "$(ls data/site | wc -l) songs, $(find data/site -name 'fix-*.json' | wc -l) fixes in data/site"
