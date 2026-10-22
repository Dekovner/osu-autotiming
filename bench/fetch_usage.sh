#!/bin/bash
# Copy the site's usage log (web/api.py usage(): a line per upload, refusal and fix) into data/usage/,
# for bench/usage.py. It lives in the api's docker volume on the home server (KZ), reached through
# the VPS like the deploy; the log only grows, so the copy here is replaced each time.
set -euo pipefail
cd "$(dirname "$0")/.."
VKEY=~/.ssh/id_ed25519_vps
VPS=auneu@VPS_IP
KZ_USER=$(ssh -i "$VKEY" "$VPS" "grep -oE '[A-Za-z0-9_.-]+@HOME_HOST\.space' /etc/systemd/system/osu-timing-tunnel.service | head -1 | cut -d@ -f1")
KZ="ssh -F /dev/null -p PORT -i ~/.ssh/HOME_KEY -o IdentitiesOnly=yes $KZ_USER@HOME_HOST.space"
mkdir -p data/usage
ssh -i "$VKEY" "$VPS" "$KZ 'docker run --rm -v osu-timing-bot_usage:/d:ro --entrypoint tar osu-timing-bot_api -C /d -cz .'" \
  | tar -xz -C data/usage
echo "$(wc -l < data/usage/usage.jsonl) events in data/usage/usage.jsonl"
