"""The final turn must not hide a failed physical return to the start."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


class ReportTests(unittest.TestCase):
    def test_final_pose_not_arrival_flag_is_scored(self):
        root = Path(__file__).parent
        config_path = root / "configs/vslam_e2e_benchmark.yaml"
        config = yaml.safe_load(config_path.read_text())
        mapping = dict(finished=True, failed_reason=None, loop_closures=1,
                       contact_episodes=0, max_position_drift_m=0.1)
        localization = dict(mapping, tracking_lost_events=1, recovery_time_s=0.2)
        localization["goals"] = [dict(name=g["name"], truth_position_error_m=0.1)
                                  for g in config["localization"]["goals"]]
        # Arrival was good; the subsequent turn drifted outside the limit.
        localization["truth_final_pose"] = [0.4, 0.0, 0.0]
        with tempfile.TemporaryDirectory(prefix="go2-vslam-report-") as directory:
            path = Path(directory)
            (path / "mapping.json").write_text(json.dumps(mapping))
            (path / "localization.json").write_text(json.dumps(localization))
            (path / "map.db").write_bytes(b"fixture")
            result = subprocess.run([
                sys.executable, str(root / "summarize_vslam_e2e.py"),
                "--config", str(config_path),
                "--mapping", str(path / "mapping.json"),
                "--localization", str(path / "localization.json"),
                "--database", str(path / "map.db"),
                "--json-output", str(path / "report.json"),
                "--markdown-output", str(path / "report.md"),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((path / "report.json").read_text())
            self.assertFalse(report["checks"]["return_position"])
            self.assertAlmostEqual(report["return_position_error_m"], 0.4)


if __name__ == "__main__":
    unittest.main()
