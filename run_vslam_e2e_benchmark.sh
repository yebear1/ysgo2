#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT_DIR}/deploy/deploy_mujoco/configs/vslam_e2e_benchmark.yaml"
REPORT_DIR="${ROOT_DIR}/reports/vslam_e2e"
DATABASE="${ROOT_DIR}/maps/vslam_e2e_benchmark.rtabmap.db"
PYTHON_BIN="${GO2_PYTHON:-/home/user/unitree_mujoco/.venv/bin/python}"
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/go2_rl_gym_vslam_e2e_${UID}.lock"
# A separate ROS domain also isolates the suite from an interactive simulator.
export ROS_DOMAIN_ID="${GO2_BENCHMARK_ROS_DOMAIN_ID:-73}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another VSLAM end-to-end benchmark is already running." >&2
  exit 3
fi

mkdir -p "${REPORT_DIR}" "$(dirname "${DATABASE}")"
cd "${ROOT_DIR}"
archive="${REPORT_DIR}/history/$(date +%Y%m%d-%H%M%S)-$$"
for artifact in mapping.json localization.json report.json report.md \
  mapping.trace.json localization.trace.json mapping.map.npz localization.map.npz \
  mapping.rtabmap.log localization.rtabmap.log mapping.sensors localization.sensors; do
  if [[ -e "${REPORT_DIR}/${artifact}" ]]; then
    mkdir -p "${archive}"
    mv "${REPORT_DIR}/${artifact}" "${archive}/"
  fi
done

summarize() {
  "${PYTHON_BIN}" deploy/deploy_mujoco/summarize_vslam_e2e.py \
    --config "${CONFIG}" \
    --mapping "${REPORT_DIR}/mapping.json" \
    --localization "${REPORT_DIR}/localization.json" \
    --database "${DATABASE}" \
    --json-output "${REPORT_DIR}/report.json" \
    --markdown-output "${REPORT_DIR}/report.md"
}

GO2_RTABMAP_LOG="${REPORT_DIR}/mapping.rtabmap.log" \
./start_go2_vslam.sh --new-map --no-rviz --database="${DATABASE}" "$@" \
  --vslam-benchmark-phase mapping \
  --vslam-benchmark-config "${CONFIG}" \
  --vslam-benchmark-output "${REPORT_DIR}/mapping.json"

if [[ ! -s "${DATABASE}" ]]; then
  echo "Mapping database was not saved: ${DATABASE}" >&2
  exit 2
fi

if ! "${PYTHON_BIN}" - "${REPORT_DIR}/mapping.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
raise SystemExit(0 if result.get("finished") and not result.get("failed_reason") else 1)
PY
then
  echo "Mapping failed; localization phase skipped." >&2
  summarize || true
  exit 1
fi

GO2_RTABMAP_LOG="${REPORT_DIR}/localization.rtabmap.log" \
./start_go2_vslam.sh --localization --no-rviz --database="${DATABASE}" "$@" \
  --vslam-benchmark-phase localization \
  --vslam-benchmark-config "${CONFIG}" \
  --vslam-benchmark-output "${REPORT_DIR}/localization.json"

summarize
