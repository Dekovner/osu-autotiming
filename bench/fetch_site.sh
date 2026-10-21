#!/bin/bash
# Copy the fixes visitors sent from auneu.space/timing (web/api.py POST /feedback) into data/site/,
# for `bench/run.py site`. They live in the api's docker volume on the home server (KZ), reached
# through the VPS like the deploy. Files already here are kept; run it again to get new ones.
set -euo pipefail
cd "$(dirname "$0")/.."
VKEY=~/.ssh/id_ed25519_vps
VPS=auneu@VPS_IP
KZ_USER=$(ssh -i "$VKEY" "$VPS" "grep -oE '[A-Za-z0-9_.-]+@HOME_HOST\.space' /etc/systemd/system/osu-timing-tunnel.service | head -1 | cut -d@ -f1")
KZ="ssh -F /dev/null -p PORT -i ~/.ssh/HOME_KEY -o IdentitiesOnly=yes $KZ_USER@HOME_HOST.space"
mkdir -p data/site
# the api image has tar; the volume is mounted read-only into a throwaway container
ssh -i "$VKEY" "$VPS" "$KZ 'docker run --rm -v osu-timing-bot_feedback:/d:ro --entrypoint tar osu-timing-bot_api -C /d -cz .'" \
  | tar -xz -C data/site --skip-old-files
echo "$(ls data/site | wc -l) songs, $(find data/site -name 'fix-*.json' | wc -l) fixes in data/site"
