# Phase 7: single shared image for all runtime services.
#
# NOTE (honesty, not a formality): this Dockerfile has NOT been built or
# run in the environment this project was developed in -- Docker itself
# is unavailable there. It has been statically reviewed line by line and
# cross-checked against requirements.txt and the actual repository
# layout, but "builds correctly" has not been executed and verified
# end-to-end. Report this limitation plainly if you build it and it
# doesn't work as expected.

FROM python:3.12-slim

WORKDIR /app

# System deps for scientific/ML stack (numpy/scipy/xgboost wheels
# generally cover this, but build-essential guards against any package
# needing to compile from source on this base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + config + persisted artifacts + dataset. Excludes
# tests/, docs, and anything in .dockerignore (see docker/.dockerignore).
COPY src/ src/
COPY config/ config/
COPY Data/ Data/
COPY artifacts/ artifacts/

# logs/ is created at runtime by src/observability/logging_config.py,
# but declared here so it exists with correct permissions from the start
# rather than relying on first-write auto-creation inside the container.
RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# No single ENTRYPOINT/CMD -- docker-compose.yml sets `command:` per
# service (traffic-gateway vs. each worker), since they all share this
# one image but run different entrypoint modules.
