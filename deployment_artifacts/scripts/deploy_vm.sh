#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/deployment_artifacts"

log() {
    printf '[deploy] %s\n' "$*"
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Required environment variable missing: $name" >&2
        exit 1
    fi
}

require_env ML_HOST

if [[ -n "${DUCKDNS_DOMAIN:-}" || -n "${DUCKDNS_TOKEN:-}" ]]; then
    require_env DUCKDNS_DOMAIN
    require_env DUCKDNS_TOKEN
fi

if [[ ! -f "$ARTIFACTS_DIR/compose.yaml" || ! -f "$ARTIFACTS_DIR/Dockerfile" || ! -f "$ARTIFACTS_DIR/caddy/Caddyfile" ]]; then
    echo "Run this script from a repository checkout that includes deployment_artifacts/." >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "This script requires sudo on the VM for Docker setup and optional DuckDNS configuration." >&2
    exit 1
fi

if ! sudo -n true >/dev/null 2>&1; then
    echo "This script requires passwordless sudo for the current VM user." >&2
    echo "Grant NOPASSWD sudo or run the setup steps as a user that can sudo non-interactively." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1 || ! id -nG "$USER" | grep -qw docker; then
    log "Ensuring Docker, Compose, and docker-group access are configured for the current user."
    "$ARTIFACTS_DIR/scripts/install_docker_ubuntu.sh"
fi

DOCKER_CMD=(docker)
if ! docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
        DOCKER_CMD=(sudo docker)
    else
        echo "Docker is installed but the current shell cannot access the daemon." >&2
        echo "Open a new shell or run 'newgrp docker', then rerun this script." >&2
        exit 1
    fi
fi

compose() {
    "${DOCKER_CMD[@]}" compose -f "$ARTIFACTS_DIR/compose.yaml" "$@"
}

cat > "$ARTIFACTS_DIR/.env.production" <<EOF
ML_HOST=$ML_HOST

MLIVE_BACKEND_BASE_URL=${MLIVE_BACKEND_BASE_URL:-http://backend:8000}
MLIVE_PUBLIC_BACKEND_URL=${MLIVE_PUBLIC_BACKEND_URL:-https://$ML_HOST}
MLIVE_ALLOW_INSECURE_LOCAL=${MLIVE_ALLOW_INSECURE_LOCAL:-0}

MLIVE_STORE_BACKEND=${MLIVE_STORE_BACKEND:-redis}
MLIVE_REDIS_URL=${MLIVE_REDIS_URL:-redis://redis:6379/0}
MLIVE_REDIS_PREFIX=${MLIVE_REDIS_PREFIX:-mlive}

MLIVE_MAX_RAW_POINTS=${MLIVE_MAX_RAW_POINTS:-6000}
MLIVE_MAX_QUEUE_TASKS=${MLIVE_MAX_QUEUE_TASKS:-2000}
MLIVE_MAX_HISTORY_ENTRIES=${MLIVE_MAX_HISTORY_ENTRIES:-2000}
MLIVE_WORKER_POLL_SECONDS=${MLIVE_WORKER_POLL_SECONDS:-0.05}
EOF

if [[ -n "${DUCKDNS_DOMAIN:-}" ]]; then
    log "Configuring DuckDNS updater."
    sudo mkdir -p /opt/duckdns
    sudo tee /opt/duckdns/update.sh >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail
curl -fsS "https://www.duckdns.org/update?domains=$DUCKDNS_DOMAIN&token=$DUCKDNS_TOKEN&ip=" >/opt/duckdns/last.log
EOF
    sudo chmod 700 /opt/duckdns/update.sh
    sudo /opt/duckdns/update.sh

    existing_crontab="$(crontab -l 2>/dev/null || true)"
    if ! grep -Fq '/opt/duckdns/update.sh' <<<"$existing_crontab"; then
        {
            printf '%s\n' "$existing_crontab"
            echo '*/5 * * * * /opt/duckdns/update.sh >/dev/null 2>&1'
        } | crontab -
    fi
fi

log "Building backend, worker, and streamlit images."
compose build --pull backend worker streamlit

log "Starting the full stack."
compose up -d
compose ps

log "Deployment complete. Open https://$ML_HOST once DNS and certificates settle."