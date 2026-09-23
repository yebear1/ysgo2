#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${GO2_PYTHON:-/home/user/unitree_mujoco/.venv/bin/python}"
REPORT_DIR="${ROOT_DIR}/reports"

cd "${ROOT_DIR}"
mkdir -p "${REPORT_DIR}"

"${PYTHON_BIN}" deploy/deploy_mujoco/test_high_level_navigation.py
"${PYTHON_BIN}" deploy/deploy_mujoco/test_corridor_navigation.py
"${PYTHON_BIN}" deploy/deploy_mujoco/test_terrain_navigator.py
"${PYTHON_BIN}" deploy/deploy_mujoco/test_navigation_geometry.py
"${PYTHON_BIN}" deploy/deploy_mujoco/test_vslam_navigation.py

set +e
"${PYTHON_BIN}" deploy/deploy_mujoco/test_stairs_headless.py \
  --speed 0.55 --duration 12 \
  2>&1 | tee "${REPORT_DIR}/stairs_benchmark_latest.txt"
STAIRS_STATUS=${PIPESTATUS[0]}

"${PYTHON_BIN}" deploy/deploy_mujoco/home_navigation_benchmark.py \
  --json-output "${REPORT_DIR}/navigation_benchmark_latest.json" \
  --markdown-output "${REPORT_DIR}/navigation_benchmark_latest.md" \
  "$@"
HOME_STATUS=$?
set -e

if [[ ${STAIRS_STATUS} -ne 0 || ${HOME_STATUS} -ne 0 ]]; then
  exit 1
fi
