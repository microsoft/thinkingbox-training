#!/bin/bash
# Start the MCP proxy and Typesense for training rollouts.
#
# Usage:
#   THINKINGBOX_ROOT=/path/to/framework \
#   THINKINGBOX_DATA=/path/to/data \
#   TYPESENSE_API_KEY=<runtime-value> ./scripts/start_servers.sh
# Stop with Ctrl+C (delegates to background_tasks.sh which handles cleanup).

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_contract.sh

require_directory_var THINKINGBOX_ROOT
require_directory_var THINKINGBOX_DATA
require_env TYPESENSE_API_KEY
export TB_MCP_START_SERVERS_FILE="$THINKINGBOX_DATA/servers/servers.yaml"

export PATH="$THINKINGBOX_ROOT/.venv/bin:$PATH"

# Sanity checks
if [[ ! -r "$TB_MCP_START_SERVERS_FILE" ]]; then
  echo "ERROR: TB_MCP_START_SERVERS_FILE not readable: $TB_MCP_START_SERVERS_FILE" >&2
  exit 1
fi
if ! command -v tb >/dev/null 2>&1; then
  echo "ERROR: 'tb' not on PATH. Did you 'source .venv/bin/activate'?" >&2
  exit 1
fi
if ! command -v typesense-server >/dev/null 2>&1; then
  echo "ERROR: 'typesense-server' not on PATH" >&2
  exit 1
fi

echo "MCP servers file : $TB_MCP_START_SERVERS_FILE"
echo "THINKINGBOX_DATA : $THINKINGBOX_DATA"
echo

exec "$THINKINGBOX_ROOT/scripts/background_tasks.sh"
