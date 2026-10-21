# build from the repo root:  docker compose -f bot/compose.yml up -d --build
# one image for both services: the Discord bot and the HTTP API behind auneu.space/timing
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*
# BASS decodes audio exactly like osu! does; it is not redistributed in the repo
RUN curl -fsSL -o /tmp/bass.zip https://www.un4seen.com/files/bass24-linux.zip \
    && mkdir -p /opt/bass/libs/x86_64 \
    && unzip -j /tmp/bass.zip libs/x86_64/libbass.so -d /opt/bass/libs/x86_64 \
    && rm /tmp/bass.zip

WORKDIR /app
# dependencies in their own layer: a code change then rebuilds in seconds, not minutes of
# pip on a VPS that also carries a VPN (keep in sync with pyproject.toml)
RUN pip install --no-cache-dir numpy scipy requests numba "discord.py>=2.4"
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY bot ./bot
COPY web/api.py ./web/api.py

ENV PYTHONUNBUFFERED=1 AUTOTIMING_BASS_DIR=/opt/bass NUMBA_CACHE_DIR=/tmp/numba OMP_NUM_THREADS=1
# uploads go through BASS, a closed native decoder: don't let it run as root
RUN useradd --system --uid 10001 --no-create-home timing
# visitors' fixes (web/api.py, POST /feedback): a fresh named volume mounted here takes this owner
RUN mkdir -p /data/feedback && chown timing /data/feedback
USER timing
CMD ["python", "bot/bot.py"]
