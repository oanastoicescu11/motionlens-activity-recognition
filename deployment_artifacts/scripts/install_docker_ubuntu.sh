#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /etc/os-release ]]; then
    echo "This script expects an Ubuntu host with /etc/os-release available." >&2
    exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "This script currently supports Ubuntu only. Detected: ${ID:-unknown}" >&2
    exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Docker and Docker Compose are already installed."
else
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg git

    sudo install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    fi
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! getent group docker >/dev/null 2>&1; then
    sudo groupadd docker
fi

if ! id -nG "$USER" | grep -qw docker; then
    sudo usermod -aG docker "$USER"
    echo "Added $USER to the docker group. New shells will pick this up automatically."
fi

docker --version || true
docker compose version || true