#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m unittest discover -s evaluation -p 'test_*.py'
mvn -q -DargLine=-Xmx768m test
