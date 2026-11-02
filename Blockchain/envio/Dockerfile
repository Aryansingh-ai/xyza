# ===========================================================
# DAANDRISHTI Envio HyperIndex — Isolated Docker Environment
# ===========================================================
#
# Base: node:22-bookworm (Debian 12, glibc 2.36)
#
# envio-linux-x64 native binary requires GLIBC_2.32.
# Debian 11 (Bullseye) ships glibc 2.31 — too old.
# Debian 12 (Bookworm) ships glibc 2.36 — correct.
#
# The Docker CLI is installed so that `envio dev` can spawn its
# own internal containers (Postgres + Hasura) via the host
# Docker socket mounted at /var/run/docker.sock.
#
# Windows node_modules/ is EXCLUDED via .dockerignore.
# All Linux dependencies are installed inside the image layer.
#
FROM node:22-bookworm

# ------------------------------------
# Install Docker CLI (not the daemon)
# Required because `envio dev` uses Docker Compose internally
# to start Postgres and Hasura GraphQL Engine.
# ------------------------------------
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg \
     | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && chmod a+r /etc/apt/keyrings/docker.gpg \
  && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update \
  && apt-get install -y docker-ce-cli \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------
# Install Node dependencies (Linux)
# package.json is copied first so Docker caches this layer
# unless dependencies change.
# ------------------------------------
COPY package.json package-lock.json ./
RUN npm install

# ------------------------------------
# Copy project source files
# node_modules/ is excluded via .dockerignore
# ------------------------------------
COPY . .

# Default command — generate types then start dev
CMD sh -c "npm run codegen && npm install && npm run dev-reset"
