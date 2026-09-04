#!/usr/bin/env bash

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "ERROR: required environment variable ${name} is not set" >&2
      return 2
    fi
  done
}

require_file_var() {
  local name="$1"
  require_env "${name}" || return
  if [[ ! -r "${!name}" ]]; then
    echo "ERROR: ${name} does not name a readable file: ${!name}" >&2
    return 2
  fi
}

require_directory_var() {
  local name="$1"
  require_env "${name}" || return
  if [[ ! -d "${!name}" ]]; then
    echo "ERROR: ${name} does not name a directory: ${!name}" >&2
    return 2
  fi
}

reject_placeholder_file() {
  local path="$1"
  if grep -Eq "<[A-Za-z0-9_]+>" "${path}"; then
    echo "ERROR: unresolved placeholder in ${path}" >&2
    return 2
  fi
}
