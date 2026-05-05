#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/.vm-deploy-bundle}"

copy_path() {
    local src="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
}

required_files=(
    "$ROOT_DIR/pyproject.toml"
    "$ROOT_DIR/run_worker.py"
    "$ROOT_DIR/src/artifacts/model/inference_bundle.joblib"
    "$ROOT_DIR/deployment_artifacts/Dockerfile"
    "$ROOT_DIR/deployment_artifacts/compose.yaml"
    "$ROOT_DIR/deployment_artifacts/.env.production.example"
    "$ROOT_DIR/deployment_artifacts/caddy/Caddyfile"
    "$ROOT_DIR/deployment_artifacts/scripts/deploy_vm.sh"
    "$ROOT_DIR/deployment_artifacts/scripts/install_docker_ubuntu.sh"
)

for path in "${required_files[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required deployment file missing: $path" >&2
        exit 1
    fi
done

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/deployment_artifacts"

copy_path "$ROOT_DIR/pyproject.toml" "$OUTPUT_DIR/pyproject.toml"
copy_path "$ROOT_DIR/run_worker.py" "$OUTPUT_DIR/run_worker.py"

if [[ -f "$ROOT_DIR/.dockerignore" ]]; then
    copy_path "$ROOT_DIR/.dockerignore" "$OUTPUT_DIR/.dockerignore"
fi

copy_path "$ROOT_DIR/src" "$OUTPUT_DIR/src"
copy_path "$ROOT_DIR/deployment_artifacts/Dockerfile" "$OUTPUT_DIR/deployment_artifacts/Dockerfile"
copy_path "$ROOT_DIR/deployment_artifacts/compose.yaml" "$OUTPUT_DIR/deployment_artifacts/compose.yaml"
copy_path "$ROOT_DIR/deployment_artifacts/.env.production.example" "$OUTPUT_DIR/deployment_artifacts/.env.production.example"
copy_path "$ROOT_DIR/deployment_artifacts/caddy" "$OUTPUT_DIR/deployment_artifacts/caddy"
copy_path "$ROOT_DIR/deployment_artifacts/scripts" "$OUTPUT_DIR/deployment_artifacts/scripts"

echo "Created minimal VM deployment bundle at: $OUTPUT_DIR"
echo "Included: pyproject.toml, run_worker.py, .dockerignore, src/, deployment_artifacts/"
echo "Excluded by design: tests/, processing/, data/, output/, plans/, README files, and other repo-only content"