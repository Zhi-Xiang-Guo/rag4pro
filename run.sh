#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ -x /usr/libexec/java_home ]]; then
  export JAVA_HOME="$(/usr/libexec/java_home -v 21)"
fi
if [[ -n "${JAVA_HOME:-}" ]]; then export PATH="$JAVA_HOME/bin:$PATH"; fi
bash scripts/native-build.sh
bash scripts/jsonrepair-build.sh
docker compose up -d --wait elasticsearch
mvn -q package
export EVIDENCE_RAG_ES_URL="${EVIDENCE_RAG_ES_URL:-http://127.0.0.1:${EVIDENCE_RAG_ES_PORT:-19200}}"
export EVIDENCE_RAG_PORT="${EVIDENCE_RAG_PORT:-${EVIDENCE_RAG_HTTP_PORT:-18080}}"
exec java -jar target/evidence-rag-java.jar
