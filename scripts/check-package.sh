#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
java -jar target/evidence-rag-java.jar --check-runtime
