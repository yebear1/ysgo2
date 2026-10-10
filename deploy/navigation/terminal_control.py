"""Shared goal-pose feedback, including the verified terminal hysteresis."""
import math
import numpy as np


def terminal_command(pose, goal, requested_yaw, velocity=None, active=False,
                     tolerance=.15, max_yaw_rate=.9):
    if requested_yaw is None:
        return None, False
    delta = np.asarray(goal) - np.asarray(pose)[:2]
    distance = float(np.linalg.norm(delta))
    if distance < .65:
        active = True
    elif distance > .90:
        active = False
    if not active:
        return None, active
    error = math.atan2(math.sin(requested_yaw-pose[2]), math.cos(requested_yaw-pose[2]))
    if distance <= tolerance and abs(error) <= math.radians(4):
        return None, active
    command = np.zeros(3, dtype=np.float32)
    command[2] = np.clip(1.8*error, -max_yaw_rate, max_yaw_rate)
    if abs(error) <= math.radians(10):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        local = np.array([c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]])
        velocity = np.zeros(2) if velocity is None else np.asarray(velocity)[:2]
        command[:2] = np.clip(.8*local-.15*velocity, -.20, .20)
    return command, active
