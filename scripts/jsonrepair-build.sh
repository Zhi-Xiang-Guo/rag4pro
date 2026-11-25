#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$project_root/native/build"
mkdir -p "$build_dir"
cd "$project_root/native/jsonrepair"
CGO_ENABLED=0 go build -mod=readonly -trimpath -o "$build_dir/ragflow-jsonrepair" .
printf '%s\n' "$build_dir/ragflow-jsonrepair"
