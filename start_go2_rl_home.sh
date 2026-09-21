#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec /home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/deploy_go2.py --config go2_home.yaml "$@"
