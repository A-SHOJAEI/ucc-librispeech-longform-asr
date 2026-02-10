#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

mkdir -p "${VENV_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  # PEP 668 safe: never install into system interpreter.
  "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
fi

if [[ -x "${VENV_DIR}/bin/pip" ]]; then
  exit 0
fi

GETPIP_URL="https://bootstrap.pypa.io/get-pip.py"
GETPIP_PATH="${VENV_DIR}/get-pip.py"

if [[ ! -f "${GETPIP_PATH}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${GETPIP_URL}" -o "${GETPIP_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${GETPIP_PATH}" "${GETPIP_URL}"
  else
    echo "ERROR: need curl or wget to fetch get-pip.py" >&2
    exit 1
  fi
fi

"${VENV_DIR}/bin/python" "${GETPIP_PATH}" --disable-pip-version-check >/dev/null

if [[ ! -x "${VENV_DIR}/bin/pip" ]]; then
  echo "ERROR: pip bootstrap failed for ${VENV_DIR}" >&2
  exit 1
fi
