#!/usr/bin/env bash
# Terminate a Runpod pod via GraphQL.
#
# Usage:  scripts/runpod_terminate.sh <pod_id>
# Reads RUNPOD_API_KEY from .env at repo root.
#
# Refuses to act without a non-empty pod ID, and lists remaining pods after
# so you can verify the right machine went down.
set -euo pipefail

POD_ID="${1:-}"
if [[ -z "${POD_ID}" ]]; then
  echo "usage: $0 <pod_id>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "${ROOT}/.env" ]]; then
  echo "error: ${ROOT}/.env not found (need RUNPOD_API_KEY)" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "${ROOT}/.env"
set +a
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "error: RUNPOD_API_KEY missing in .env" >&2
  exit 1
fi

read -r -p "Terminate pod ${POD_ID}? (y/N) " ans
if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
  echo "aborted"
  exit 0
fi

curl -sS -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}"
echo ""

echo "--- remaining pods ---"
curl -sS -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } } } }"}' \
  | python3 -m json.tool
