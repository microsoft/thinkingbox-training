#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKEND=${BACKEND:-http://127.0.0.1:8001}
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" -m uvicorn sysmerge_proxy:app \
  --app-dir "$SCRIPT_DIR" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8014}"
