#!/usr/bin/env python3
"""Deterministic coverage/accounting tests; no ROS processes are launched."""

import math
import unittest

from run_vslam_batch import aggregate, cases, stall_episodes, timing_stats


def trace_row(time, x=0, yaw=0, active=True, blocked=False, visible=True, goal=0):
    return {"simulation_time": time, "truth_pose": [x, 0, yaw],
            "vslam_pose": [x, 0, yaw] if visible else None,
            "active_goal_index": goal, "camera_blocked": blocked,
            "control": {"navigation_active": active}}


class BatchTests(unittest.TestCase):
    def test_predeclared_matrix(self):
        matrix = cases()
        self.assertEqual(len(matrix), 10)
        self.assertEqual(len({r["id"] for r in matrix}), 10)
        self.assertEqual(len({tuple(r["order"]) for r in matrix[:6]}), 6)
        self.assertTrue(all(r["initial_yaw_deg"] == 0 for r in matrix[:6]))
        self.assertEqual([r["initial_yaw_deg"] for r in matrix[6:]], [-30, 30, 60, 180])

    def test_stall_episode_not_overcounted(self):
        trace = [trace_row(t) for t in range(31)]
        self.assertEqual(len(stall_episodes(trace)), 1)
        self.assertEqual(stall_episodes(trace)[0]["duration_s"], 30)
        self.assertEqual(stall_episodes(trace[:10]), [])
        trace += [trace_row(31, x=1)] + [trace_row(t, x=1) for t in range(32, 44)]
        self.assertEqual(len(stall_episodes(trace)), 2)

    def test_purposeful_movement_and_occlusion(self):
        self.assertEqual(stall_episodes([trace_row(t, x=t * .03) for t in range(30)]), [])
        self.assertEqual(stall_episodes([trace_row(t, yaw=t * .1) for t in range(30)]), [])
        for kwargs in ({"blocked": True}, {"visible": False}, {"active": False}):
            self.assertEqual(stall_episodes([trace_row(t, **kwargs) for t in range(30)]), [])
        trace = [trace_row(t, goal=t // 8) for t in range(30)]
        self.assertEqual(stall_episodes(trace), [])

    def test_failure_and_missing_metrics_stay_in_denominator(self):
        rows = [dict(status="completed", passed=True, contact_episodes=0, stall_episodes=0, wall_seconds=10),
                dict(status="completed", passed=False, contact_episodes=3, stall_episodes=2, wall_seconds=20),
                dict(status="completed", passed=False, contact_episodes=None, stall_episodes=None,
                     wall_seconds=30, failure_reason="crash"), dict(status="pending")]
        result = aggregate(rows, 10)
        self.assertEqual(result["success_rate_all_planned"], .1)
        self.assertEqual(result["failed_runs"], 2)
        self.assertEqual(result["total_known_contact_episodes"], 3)
        self.assertEqual(result["total_stall_episodes"], 2)
        self.assertEqual(result["runs_with_unknown_contact_metrics"], 1)
        self.assertEqual(result["wall_seconds_all_completed"]["mean"], 20)
        self.assertEqual(timing_stats([1, 2, 100])["p95_nearest_rank"], 100)


if __name__ == "__main__":
    unittest.main()
