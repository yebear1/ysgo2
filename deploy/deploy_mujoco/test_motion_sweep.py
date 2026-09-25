#!/usr/bin/env python3
"""Measured-endpoint checks for the flat-home rectangular motion envelope."""

from types import SimpleNamespace

import numpy as np

from terrain_navigator import TerrainNavigator


def scan(points):
    points = np.asarray(points, dtype=float)
    return SimpleNamespace(
        planar_ranges=np.linalg.norm(points, axis=1),
        planar_angles=np.arctan2(points[:, 1], points[:, 0]),
        planar_max_range=2.0,
    )


def main():
    nav = TerrainNavigator({"planar_obstacle_check": True,
                            "body_half_length": 0.39, "corridor_half_width": 0.20})
    book = scan([[-0.30, 0.23]])
    assert not nav._swept_motion_clear([0.20, 0, -0.45], book)
    assert nav._swept_motion_clear([0.20, 0, 0], book)
    assert nav._swept_motion_clear([0, -0.15, 0], book)
    assert not nav._swept_motion_clear([0, 0.15, 0], book)
    nav.analyze = lambda lidar: ("CLEAR", float("inf"))
    corrected = nav._guard_swept_motion([0.20, 0, -0.45], book)
    assert np.linalg.norm(corrected[:2]) > 0
    assert nav._swept_motion_clear(corrected, book)
    # Existing overlap may be escaped, but never made deeper.
    overlap = scan([[-0.30, 0.19]])
    assert nav._swept_motion_clear([0, -0.15, 0], overlap)
    assert not nav._swept_motion_clear([0, 0.15, 0], overlap)
    # Exact footprint fits a 0.44 m passage: no circular/inflated wall test.
    corridor = scan([[x, y] for x in np.linspace(-0.6, 1.0, 12) for y in [-0.22, 0.22]])
    assert nav._swept_motion_clear([0.5, 0, 0], corridor)
    assert not nav._swept_motion_clear([0.2, 0, 0.45], corridor)
    assert np.allclose(nav._guard_swept_motion([0, 0, 0], overlap), 0)
    corner = scan([[0.40, 0.19], [0.37, 0.215], [0.32, 0.27]])
    assert not nav._swept_motion_clear([0, 0, 0.45], corner)
    escape = nav._guard_swept_motion([0, 0, 0.45], corner)
    assert np.linalg.norm(escape) > 0, escape
    assert nav._swept_motion_clear(escape, corner)
    # Limiting yaw after validation can turn a safe arc into a collision.
    endpoint = scan([[0.45, -0.175]])
    bounds = ([-0.30, -0.25, -0.45], [0.65, 0.25, 0.45])
    desired = [0.8, 0, 0.8]
    assert nav._swept_motion_clear(desired, endpoint)
    assert not nav._swept_motion_clear(np.clip(desired, *bounds), endpoint)
    limited = nav._guard_swept_motion(desired, endpoint, bounds)
    assert nav._swept_motion_clear(limited, endpoint)
    assert np.all(limited >= bounds[0]) and np.all(limited <= bounds[1])
    print(f"rear-foot arc guarded: {corrected}; narrow straight passage remains open")


if __name__ == "__main__":
    main()
