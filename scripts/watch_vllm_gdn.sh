#!/usr/bin/env bash
# Keep the rollout vLLM alive. Adapter restoration is handled by the trainer's
# pre-rollout gate, which verifies registry path and executes an active LoRA probe.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${VLLM_START_SCRIPT:=${PROJECT_ROOT}/scripts/run_vllm_gdn_tp2.sh}"
: "${VLLM_HEALTH_URL:=http://127.0.0.1:8000/health}"
: "${VLLM_PID_FILE:=/tmp/vllm_gdn.pid}"
: "${VLLM_WATCHDOG_LOCK:=/tmp/vllm_gdn_watchdog.lock}"
: "${VLLM_WATCHDOG_LOG:=/tmp/vllm_watchdog.log}"
: "${VLLM_SERVER_LOG:=/tmp/vllm.restart.log}"
: "${VLLM_BOOT_GRACE:=180}"
: "${VLLM_POLL_SECONDS:=30}"
: "${VLLM_LOW_MEMORY_MIB:=2000}"
: "${VLLM_METRICS_URL:=http://127.0.0.1:8000/metrics}"
# Consecutive unhealthy polls before restarting (30s poll => 90s of failure).
: "${VLLM_FAIL_THRESHOLD:=3}"
# A queue that is non-empty and completely unchanging for this long means the
# engine is wedged mid-collective but has not yet been aborted by NCCL's watchdog.
: "${VLLM_STUCK_QUEUE_SECONDS:=600}"
: "${VLLM_CUDA_VISIBLE_DEVICES:=6,7}"
: "${VLLM_WATCHDOG_ID:=$(printf '%s' "${VLLM_PID_FILE}" | tr -c 'A-Za-z0-9_' '_')}"
# Overridable so the process selector can be exercised against a fixture tree.
: "${VLLM_PROC_ROOT:=/proc}"

exec 9>"${VLLM_WATCHDOG_LOCK}"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"${VLLM_WATCHDOG_LOG}"
}

server_pid() {
  if [[ -s "${VLLM_PID_FILE}" ]]; then
    tr -cd '0-9' <"${VLLM_PID_FILE}"
  fi
}

start_server() {
  setsid nohup env VLLM_WATCHDOG_ID="${VLLM_WATCHDOG_ID}" \
    bash "${VLLM_START_SCRIPT}" >>"${VLLM_SERVER_LOG}" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${VLLM_PID_FILE}"
  log "started vLLM pid=${pid}"
  sleep "${VLLM_BOOT_GRACE}"
}

stop_server() {
  local pid=$1
  local sess
  # Resolve the family session BEFORE signalling: once the leader dies its
  # session id can no longer be looked up, only inferred from the pid.
  sess="$(server_session "${pid}")"
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  sleep 5
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  reap_engine_orphans "${sess}"
  wait_for_gpu_release
}

# Session id of this watchdog's server family. start_server launches the server
# under `setsid`, so the API server is the session leader and its id IS the
# recorded pid; every descendant inherits that session and keeps it even after
# the leader dies -- which is exactly the orphan case. Falls back to the recorded
# pid so the dead-PID path still resolves the right family.
server_session() {
  local pid="$1" sess=""
  [[ -n "${pid}" ]] || return 0
  sess="$(ps -o sess= -p "${pid}" 2>/dev/null | tr -d ' ')"
  printf '%s\n' "${sess:-${pid}}"
}

# PIDs of vLLM processes belonging to THIS watchdog's server family.
#
# Session scoping is the load-bearing signal. vLLM re-creates the environment of
# its spawned EngineCore/Worker_TP children, so the inherited VLLM_WATCHDOG_ID
# marker survives ONLY on the API server -- verified live: the API server carries
# the marker and its EngineCore/Worker_TP children carry none. Marker-only
# matching therefore selected nothing on the dead-PID path, the one path where no
# process group survives to signal, so the orphans this reaper exists to clear
# kept holding the GPUs. The marker is still honoured as a second signal so a
# process that DID inherit it is reaped even if the session lookup failed.
# Name matching alone is not acceptable: it would kill a second healthy vLLM
# engine sharing the pod's PID namespace.
vllm_family_pids() {
  local target_session="$1" pid sess
  ps -eo pid=,sess=,args= | awk '
    index($0, "VLLM::EngineCore") || index($0, "VLLM::Worker_TP") ||
    index($0, "vllm serve") { print $1, $2 }' |
  while read -r pid sess; do
    if [[ -n "${target_session}" && "${sess}" == "${target_session}" ]]; then
      printf '%s\n' "${pid}"
      continue
    fi
    if [[ -r "${VLLM_PROC_ROOT}/${pid}/environ" ]] &&
      tr '\0' '\n' <"${VLLM_PROC_ROOT}/${pid}/environ" 2>/dev/null |
        grep -Fxq "VLLM_WATCHDOG_ID=${VLLM_WATCHDOG_ID}"; then
      printf '%s\n' "${pid}"
    fi
  done
}

# When the engine is wedged, signalling the API server's process group is not
# enough: EngineCore, the Worker_TP shards and the multiprocessing
# worker processes can be re-parented and keep the whole GPU allocation, so the
# restart would then fail to claim memory.
reap_engine_orphans() {
  local target_session="${1:-}" pids=() pid
  mapfile -t pids < <(vllm_family_pids "${target_session}")
  (( ${#pids[@]} )) || return 0
  log "reaping ${#pids[@]} vLLM orphan(s) in session ${target_session:-unknown}: ${pids[*]}"
  for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  sleep 8
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null || true
  done
}

wait_for_gpu_release() {
  local i used
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      -i "${VLLM_CUDA_VISIBLE_DEVICES}" 2>/dev/null |
      awk '{s+=$1} END {print s+0}' || true)"
    (( ${used:-0} < VLLM_LOW_MEMORY_MIB )) && return 0
    sleep 2
  done
  log "GPU memory still ${used:-unknown}MiB after cleanup; starting anyway"
}

# Running+waiting requests summed across engines; empty string when unavailable.
# Must never fail: this runs under `set -e`, so a transient scrape error here
# would otherwise terminate the watchdog itself.
queue_depth() {
  local body
  body="$(curl -sS --max-time 5 "${VLLM_METRICS_URL}" 2>/dev/null || true)"
  [[ -n "${body}" ]] || return 0
  printf '%s\n' "${body}" | awk '
    /^vllm:num_requests_running[{ ]/ || /^vllm:num_requests_waiting[{ ]/ \
      { s += $NF; seen = 1 } END { if (seen) print int(s) }' || true
}

log "watchdog started"
failures=0
stuck_depth=""
stuck_since=0
while true; do
  pid="$(server_pid)"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    # A failed API server can leave EngineCore/Worker_TP children alive and
    # holding all GPU memory. Clean this watchdog's process family before every
    # replacement, including the dead-PID startup path. The dead leader's pid is
    # still the session id its surviving children carry.
    reap_engine_orphans "$(server_session "${pid}")"
    wait_for_gpu_release
    start_server
    failures=0
    stuck_depth=""
    stuck_since=0
    continue
  fi

  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
    "${VLLM_HEALTH_URL}" 2>/dev/null || true)"
  memory_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "${VLLM_CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -n 1 || true)"

  # A NCCL collective deadlock (LoRA registry mutation racing an in-flight decode)
  # aborts the engine but leaves the process holding its full GPU allocation, so
  # the previous `health != 200 AND memory released` condition could never fire on
  # the very failure this watchdog exists to catch. Restart on sustained health
  # failure alone; memory is now diagnostics, and is drained during stop_server.
  if [[ "${code}" != "200" ]]; then
    failures=$((failures + 1))
    log "health=${code:-none} gpu6_mem=${memory_mib:-unknown}MiB failures=${failures}"
    if (( failures >= VLLM_FAIL_THRESHOLD )); then
      log "vLLM unhealthy x${failures}; restarting pid=${pid}"
      stop_server "${pid}"
      start_server
      failures=0
      stuck_depth=""
      stuck_since=0
    fi
    sleep "${VLLM_POLL_SECONDS}"
    continue
  fi
  failures=0

  # Health can still report 200 while the engine is wedged inside a collective and
  # NCCL has not yet hit its abort timeout. In that window the only visible symptom
  # is a queue that is non-empty and perfectly frozen, so detect it directly rather
  # than waiting for the engine to die.
  depth="$(queue_depth)"
  if [[ -n "${depth}" && "${depth}" -gt 0 && "${depth}" == "${stuck_depth}" ]]; then
    now="$(date +%s)"
    if (( stuck_since == 0 )); then
      stuck_since="${now}"
    fi
    stalled=$(( now - stuck_since ))
    if (( stalled >= VLLM_STUCK_QUEUE_SECONDS )); then
      log "queue frozen at ${depth} request(s) for ${stalled}s with health=200; \
engine wedged, restarting pid=${pid}"
      stop_server "${pid}"
      start_server
      stuck_depth=""
      stuck_since=0
    fi
  else
    stuck_depth="${depth}"
    stuck_since=0
  fi
  sleep "${VLLM_POLL_SECONDS}"
done
