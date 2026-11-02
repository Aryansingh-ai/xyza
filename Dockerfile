FROM node:20-bookworm-slim

WORKDIR /app

# Install native build tools required for compiling better-sqlite3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package manifests first to leverage Docker layer caching
COPY Blockchain/api/package*.json ./Blockchain/api/

# Install production dependencies for the API
WORKDIR /app/Blockchain/api
RUN npm install --omit=dev

# Return to repository root workdir and copy the rest of the application
WORKDIR /app
COPY . .

# Default environment configuration
ENV PORT=3000
EXPOSE 3000

CMD ["node", "Blockchain/api/index.js"]
