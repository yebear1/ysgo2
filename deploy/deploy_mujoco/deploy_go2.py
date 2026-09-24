import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils
from lidar_heightmap import LidarHeightMap
from perceptive_observation import PerceptiveObservationBuilder
from terrain_navigator import TerrainNavigator
from goal_navigator import GoalNavigator
from high_level_nav_policy import HighLevelNavigationPolicy
from ros2_vslam_bridge import Ros2VslamBridge
from vslam_e2e_benchmark import VslamE2EBenchmark

import os
import signal
from contextlib import nullcontext
import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from argparse import ArgumentParser
import pygame
import glfw


class KeyboardCommand:
    def __init__(self, config=None, goal_names=None):
        config = config or {}
        self.goal_names = list(goal_names or [])[:9]
        self.command = np.zeros(3, dtype=np.float32)
        self.reset_requested = False
        self.goal_request = None
        self.goal_cancel_requested = False
        self.forward_speed = float(config.get("forward", 0.8))
        self.backward_speed = float(config.get("backward", -0.6))
        self.turn_speed = float(config.get("turn", 1.0))

    def handle_key(self, keycode):
        if glfw.KEY_1 <= keycode <= glfw.KEY_9:
            index = keycode - glfw.KEY_1
            if index >= len(self.goal_names):
                return
            self.command[:] = 0.0
            self.goal_cancel_requested = False
            self.goal_request = self.goal_names[index]
            print(f"Navigation requested: {self.goal_request}")
            return
        if keycode in (glfw.KEY_0, glfw.KEY_KP_0):
            self.command[:] = 0.0
            self.goal_cancel_requested = True
            print("Navigation cancelled")
            return
        if keycode in (glfw.KEY_UP, glfw.KEY_KP_8):
            self.command[:] = (self.forward_speed, 0.0, 0.0)
        elif keycode in (glfw.KEY_DOWN, glfw.KEY_KP_2):
            self.command[:] = (self.backward_speed, 0.0, 0.0)
        elif keycode in (glfw.KEY_LEFT, glfw.KEY_KP_4):
            self.command[:] = (0.0, 0.0, self.turn_speed)
        elif keycode in (glfw.KEY_RIGHT, glfw.KEY_KP_6):
            self.command[:] = (0.0, 0.0, -self.turn_speed)
        elif keycode in (glfw.KEY_SPACE, glfw.KEY_KP_5):
            self.command[:] = 0.0
        elif keycode == glfw.KEY_R:
            self.command[:] = 0.0
            self.reset_requested = True
        else:
            return
        self.goal_cancel_requested = True
        print(
            f"Command: Vx={self.command[0]:+.2f}, "
            f"Vy={self.command[1]:+.2f}, Wz={self.command[2]:+.2f}"
        )

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_yaw(q):
    qw, qx, qy, qz = q
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_xbox_command(joystick, max_cmd):
    pygame.event.pump()
    dead_zone = 0.1
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone: lx = 0
    if abs(ly) < dead_zone: ly = 0
    if abs(rx) < dead_zone: rx = 0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


def get_environment_contact_details(model, data):
    """Return strongest non-floor robot/environment contact details."""
    strongest = None
    ignored_environment_geoms = {"floor", "living_rug"}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])

        # The furnished scene geoms are attached to world body 0; the robot
        # geoms are attached to articulated bodies. Ignore self contacts.
        if (body1 == 0) == (body2 == 0):
            continue
        environment_geom = geom1 if body1 == 0 else geom2
        robot_geom = geom2 if body1 == 0 else geom1
        environment_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom)
            or f"geom{environment_geom}"
        )
        if environment_name in ignored_environment_geoms:
            continue
        robot_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            or f"geom{robot_geom}"
        )
        contact_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        force = float(np.linalg.norm(contact_force[:3]))
        if strongest is None or force > strongest[0]:
            strongest = (force, robot_name, environment_name)

    return strongest


def get_environment_contact(model, data):
    """Return the strongest non-floor robot/environment contact for debugging."""
    strongest = get_environment_contact_details(model, data)
    if strongest is None:
        return "Contact: none"
    force, robot_name, environment_name = strongest
    return f"Contact: {robot_name} <-> {environment_name} ({force:.0f} N)"


def normalize_view_options(viewer):
    """Keep visual meshes and the environment visible after keyboard input."""
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    for flag in (
        mujoco.mjtVisFlag.mjVIS_ACTUATOR,
        mujoco.mjtVisFlag.mjVIS_ACTIVATION,
        mujoco.mjtVisFlag.mjVIS_JOINT,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
        mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
        mujoco.mjtVisFlag.mjVIS_SELECT,
    ):
        viewer.opt.flags[flag] = 0

    # group 0 contains simplified robot collision primitives, group 1 is the
    # furnished environment, and group 2 contains the robot's visual meshes.
    viewer.opt.geomgroup[:] = 0
    viewer.opt.geomgroup[1] = 1
    viewer.opt.geomgroup[2] = 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="go2.yaml", help="Config file in deploy/deploy_mujoco/configs.")
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Start autonomous navigation to a world-coordinate goal.",
    )
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--headless", action="store_true", help="Render sensors without a desktop viewer.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    parser.add_argument("--save-moe-latent", action="store_true", help="Whether to save mixture of experts latent vectors.")
    parser.add_argument(
        "--ros2-vslam",
        action="store_true",
        help="Publish the simulated front RGB-D camera for RTAB-Map.",
    )
    parser.add_argument(
        "--vslam-benchmark-phase", choices=("mapping", "localization")
    )
    parser.add_argument("--vslam-benchmark-config")
    parser.add_argument("--vslam-benchmark-output")
    args = parser.parse_args()
    if args.headless and args.save_video:
        parser.error("--save-video requires the desktop viewer")
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    save_moe_latent = args.save_moe_latent
    config_file = args.config

    if save_video:
        import imageio
    if visualize_moe_weights:
        from matplotlib import pyplot as plt

    pygame.init()
    use_joystick = False
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if args.ros2_vslam:
            config.setdefault("ros2_vslam", {})["enabled"] = True
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        render_decimation = max(
            1, int(config.get("render_decimation", control_decimation))
        )

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        lidar_config = config.get("lidar", {})
        perceptive_policy = bool(config.get("perceptive_policy", False))
        navigation_config = config.get("navigation", {})
        goal_navigation_config = dict(config.get("goal_navigation", {}))
        high_level_navigation_config = dict(
            config.get("high_level_navigation", {})
        )
        if "policy_path" in high_level_navigation_config:
            high_level_navigation_config["policy_path"] = str(
                high_level_navigation_config["policy_path"]
            ).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        camera_config = config.get("camera", {})
        ros2_vslam_config = config.get("ros2_vslam", {})
        if ros2_vslam_config.get("enabled", False):
            # Global planning must consume RTAB-Map's optimized occupancy grid,
            # never the simulator's scene geometry.
            goal_navigation_config["map_source"] = "rtabmap"
            # Flat-home navigation must avoid small objects as well as tall
            # furniture. A single torso-height range ring misses foot hazards.
            lidar_config = dict(lidar_config)
            lidar_config["planar_scan_heights"] = ros2_vslam_config.get(
                "planar_scan_heights", [0.04, 0.32]
            )
            navigation_config = dict(navigation_config)
            navigation_config["max_step_up"] = float(
                ros2_vslam_config.get("max_step_up", 0.04)
            )
            navigation_config["planar_obstacle_check"] = bool(
                ros2_vslam_config.get("planar_obstacle_check", True)
            )
        keyboard_config = config.get("keyboard", {})

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    keyboard = KeyboardCommand(
        keyboard_config,
        goal_navigation_config.get("presets", {}).keys(),
    )

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    stationary_updates = 0
    hold_active = False
    hold_target = default_angles.copy()
    hold_settle_updates = max(1, int(0.5 / (simulation_dt * control_decimation)))
    hold_kps = np.full_like(kps, 40.0)
    hold_kds = np.full_like(kds, 1.5)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mujoco.mj_forward(m, d)
    lidar = None
    lidar_scan_steps = 1
    if lidar_config.get("enabled", False):
        lidar = LidarHeightMap(m, d, lidar_config)
        lidar_scan_steps = max(1, int(lidar.scan_interval / simulation_dt))
    if perceptive_policy and lidar is None:
        raise RuntimeError("perceptive_policy requires lidar.enabled: true")
    privileged_builder = PerceptiveObservationBuilder(
        m, simulation_dt * control_decimation
    ) if perceptive_policy else None
    navigator = TerrainNavigator(navigation_config)
    goal_navigator = GoalNavigator(m, d, goal_navigation_config)
    high_level_navigator = HighLevelNavigationPolicy(
        high_level_navigation_config
    )
    if high_level_navigator.enabled and lidar is None:
        raise RuntimeError("high_level_navigation requires lidar.enabled: true")
    if lidar is not None:
        lidar.scan()
    if args.goal is not None and not ros2_vslam_config.get("enabled", False):
        goal_navigator.set_goal(args.goal, d.qpos[:2], d.time)

    renderer = mujoco.Renderer(m, height=360, width=640) if save_video else None
    ros2_vslam = None
    if ros2_vslam_config.get("enabled", False):
        if args.vslam_benchmark_output:
            ros2_vslam_config["diagnostic_dir"] = str(
                Path(args.vslam_benchmark_output).with_suffix(".sensors")
            )
        ros2_vslam = Ros2VslamBridge(m, d, ros2_vslam_config)
    vslam_benchmark = None
    if args.vslam_benchmark_phase:
        if ros2_vslam is None:
            raise RuntimeError("VSLAM benchmark requires --ros2-vslam")
        if not args.vslam_benchmark_config or not args.vslam_benchmark_output:
            raise RuntimeError(
                "VSLAM benchmark requires config and output paths"
            )
        vslam_benchmark = VslamE2EBenchmark(
            args.vslam_benchmark_config,
            args.vslam_benchmark_phase,
            args.vslam_benchmark_output,
        )
        goal_navigator.goal_tolerance = float(
            vslam_benchmark.phase_config.get("goal_tolerance", goal_navigator.goal_tolerance)
        )
        def stop_benchmark(signum, _frame):
            vslam_benchmark.failed_reason = f"interrupted by signal {signum}"
            vslam_benchmark.finished = True
        signal.signal(signal.SIGTERM, stop_benchmark)
        signal.signal(signal.SIGINT, stop_benchmark)
    pending_goal_name = None
    pending_goal_coordinates = args.goal if ros2_vslam is not None else None
    
    # load policy
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    video_fps = 50
    if save_video:
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    if visualize_moe_weights:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.set_title(f"Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")
        bars = None
    
    if save_moe_latent:
        latent_save_dir = str(PATH_PARENT / "data_latents")
        os.makedirs(latent_save_dir, exist_ok=True)
        latent_filename = f"{model_name}_{cmd_str}_latents.npy"
        latent_path = os.path.join(latent_save_dir, latent_filename)
        all_latents = []

    preset_help = " | ".join(
        f"{index + 1}:{name}"
        for index, name in enumerate(goal_navigator.preset_names[:9])
    )
    print("Arrow keys or keypad 8/2/4/6: move | Space/keypad 5: stop | R: reset")
    if preset_help:
        print(f"Goal navigation: {preset_help} | 0:cancel")
    viewer_context = (
        nullcontext(None) if args.headless else
        mujoco.viewer.launch_passive(m, d, key_callback=keyboard.handle_key)
    )
    with viewer_context as viewer:

        # set viewer.camera to follow robot
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = 1
            viewer.cam.distance = float(camera_config.get("distance", 2.0))
            viewer.cam.elevation = float(camera_config.get("elevation", -20.0))
            viewer.cam.azimuth = float(camera_config.get("azimuth", 60.0))
            with viewer.lock():
                normalize_view_options(viewer)

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        wall_clock_start = time.perf_counter()
        simulation_clock_start = float(d.time)
        realtime_factor = 1.0
        contact_status = "Contact: none"
        while (viewer is None or viewer.is_running()) and time.time() - start < simulation_duration:
            if keyboard.reset_requested:
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                action.fill(0.0)
                last_action.fill(0.0)
                target_dof_pos = default_angles.copy()
                stationary_updates = 0
                hold_active = False
                if privileged_builder is not None:
                    privileged_builder.reset(d.qvel[6:])
                navigator.reset()
                goal_navigator.cancel()
                high_level_navigator.reset()
                if lidar is not None:
                    lidar.scan()
                if ros2_vslam is not None:
                    ros2_vslam.reset()
                keyboard.reset_requested = False
                wall_clock_start = time.perf_counter()
                simulation_clock_start = float(d.time)
                realtime_factor = 1.0

            vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            navigation_velocity = (
                ros2_vslam.navigation_velocity if ros2_vslam is not None else
                np.array([local_vel[0], local_vel[1], local_ang_vel[2]], dtype=np.float32)
            )
            global_pose = ros2_vslam.global_pose if ros2_vslam is not None else None
            if ros2_vslam is not None:
                map_update = ros2_vslam.take_navigation_map()
                if map_update is not None:
                    goal_navigator.update_occupancy_grid(map_update)
                rviz_goal = ros2_vslam.take_navigation_goal()
                if rviz_goal is not None:
                    pending_goal_coordinates = rviz_goal
                    pending_goal_name = None
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
            if counter % control_decimation == 0:
                if vslam_benchmark is not None:
                    vslam_benchmark.update_control(
                        global_pose, goal_navigator, d.time
                    )
                    if vslam_benchmark.consume_waypoint_change():
                        navigator.reset()
                        high_level_navigator.reset()
                if keyboard.goal_cancel_requested:
                    goal_navigator.cancel()
                    keyboard.goal_cancel_requested = False
                if keyboard.goal_request is not None:
                    if ros2_vslam is None:
                        goal_navigator.set_named_goal(
                            keyboard.goal_request, d.qpos[:2], d.time
                        )
                    else:
                        pending_goal_name = keyboard.goal_request
                    keyboard.goal_request = None

                if global_pose is not None and goal_navigator.map_ready:
                    if pending_goal_coordinates is not None:
                        goal_navigator.set_goal(
                            pending_goal_coordinates, global_pose[:2], d.time
                        )
                        pending_goal_coordinates = None
                    if pending_goal_name is not None:
                        goal_navigator.set_named_goal(
                            pending_goal_name, global_pose[:2], d.time
                        )
                        pending_goal_name = None

                benchmark_direct = (
                    vslam_benchmark is not None
                    and vslam_benchmark.direct_target is not None
                )
                benchmark_heading = (
                    vslam_benchmark.heading_command(global_pose)
                    if vslam_benchmark is not None else None
                )
                benchmark_recovery = (
                    vslam_benchmark.recovery_command(navigator.state, d.time)
                    if vslam_benchmark is not None else None
                )
                benchmark_tracking_recovery = (
                    vslam_benchmark.tracking_recovery_command(
                        global_pose, d.time, ros2_vslam.sensor_yaw, lidar
                    )
                    if vslam_benchmark is not None else None
                )
                goal_control = goal_navigator.active or benchmark_direct or (
                    benchmark_heading is not None
                )
                if benchmark_tracking_recovery is not None:
                    goal_control = False
                    manual_cmd = benchmark_tracking_recovery
                elif benchmark_recovery is not None:
                    navigator.reset()
                    goal_control = False
                    manual_cmd = benchmark_recovery
                elif benchmark_heading is not None:
                    manual_cmd = benchmark_heading
                elif benchmark_direct:
                    if global_pose is None:
                        manual_cmd = np.zeros(3, dtype=np.float32)
                    else:
                        manual_cmd = high_level_navigator.command(
                            vslam_benchmark.direct_target,
                            global_pose[:2], global_pose[2],
                            navigation_velocity,
                            lidar,
                        )
                elif goal_control:
                    if global_pose is None:
                        # Do not dead-reckon a global route from MuJoCo truth if
                        # visual tracking is temporarily lost.
                        manual_cmd = np.zeros(3, dtype=np.float32)
                    else:
                        manual_cmd = goal_navigator.update(
                            global_pose[:2], global_pose[2], d.time
                        )
                        if high_level_navigator.enabled and goal_navigator.active:
                            waypoint_index = min(
                                goal_navigator.waypoint_index,
                                len(goal_navigator.path) - 1,
                            )
                            learned_command = high_level_navigator.command(
                                goal_navigator.path[waypoint_index],
                                global_pose[:2],
                                global_pose[2],
                                navigation_velocity,
                                lidar,
                            )
                            if learned_command is not None:
                                manual_cmd = learned_command
                elif use_joystick:
                    manual_cmd = get_xbox_command(joystick, config["max_cmd"])
                else:
                    manual_cmd = keyboard.command.copy()
                navigation_yaw = (
                    float(global_pose[2])
                    if global_pose is not None
                    else (
                        ros2_vslam.sensor_yaw
                        if ros2_vslam is not None
                        else get_yaw(d.qpos[3:7])
                    )
                )
                cmd = navigator.update(
                    manual_cmd,
                    lidar,
                    navigation_yaw,
                    autonomous=goal_control,
                    lateral_velocity=float(navigation_velocity[1]),
                    goal_distance=(
                        goal_navigator.last_distance if goal_control else None
                    ),
                )
                if vslam_benchmark is not None:
                    # Keep inter-frame image motion inside the RGB-D visual
                    # odometry capture range.  This is a sensor constraint,
                    # not a truth-pose correction.
                    speed_limits = vslam_benchmark.phase_config
                    cmd[0] = np.clip(
                        cmd[0],
                        -float(speed_limits.get("max_reverse_speed", 0.30)),
                        float(speed_limits.get("max_linear_speed", 0.65)),
                    )
                    cmd[1] = np.clip(
                        cmd[1],
                        -float(speed_limits.get("max_lateral_speed", 0.25)),
                        float(speed_limits.get("max_lateral_speed", 0.25)),
                    )
                    cmd[2] = np.clip(
                        cmd[2],
                        -float(speed_limits.get("max_yaw_rate", 0.45)),
                        float(speed_limits.get("max_yaw_rate", 0.45)),
                    )
                if np.linalg.norm(cmd) < 1e-4:
                    stationary_updates += 1
                    if stationary_updates >= hold_settle_updates and not hold_active:
                        hold_target = default_angles.copy()
                        hold_active = True
                else:
                    stationary_updates = 0
                    hold_active = False
                controller_label = (
                    " NAV-RL"
                    if goal_control and high_level_navigator.enabled
                    else ""
                )
                show_str += (
                    f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, "
                    f"Wz={cmd[2]:.2f}{controller_label}"
                )
                if counter % (control_decimation * 50) == 0:
                    print(
                        f"{show_str}, RTF={realtime_factor:.2f}x | "
                        f"{navigator.status_text()} | {contact_status}",
                        end='\r',
                    )

            active_kps = hold_kps if hold_active else kps
            active_kds = hold_kds if hold_active else kds
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                active_kps,
                np.zeros_like(active_kds),
                d.qvel[6:],
                active_kds,
            )
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)
            if lidar is not None and counter % lidar_scan_steps == 0:
                lidar.scan()
            if ros2_vslam is not None:
                if vslam_benchmark is not None:
                    ros2_vslam.set_camera_blocked(vslam_benchmark.camera_blocked)
                ros2_vslam.update(d.time)
            if vslam_benchmark is not None:
                truth_pose = np.array(
                    [d.qpos[0], d.qpos[1], get_yaw(d.qpos[3:7])],
                    dtype=np.float64,
                )
                benchmark_contact = get_environment_contact_details(m, d)
                vslam_benchmark.observe(
                    d.time,
                    truth_pose,
                    ros2_vslam.global_pose,
                    benchmark_contact,
                    ros2_vslam,
                    control_diagnostics={
                        "command": np.asarray(cmd).tolist(),
                        "navigation_state": navigator.state,
                        "hazard": navigator.hazard,
                        # Diagnostic-only joint snapshot for reconstructing
                        # gait contact geometry offline; never a nav input.
                        "qpos": (
                            d.qpos.tolist()
                            if benchmark_contact is not None and benchmark_contact[0] >= 5.0
                            else None
                        ),
                    },
                )

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                if hold_active:
                    target_dof_pos = hold_target
                else:
                    qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                    dqj = d.qvel[6:] * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(d.qpos[3:7])
                    ang_vel = d.qvel[3:6] * ang_vel_scale

                    obs[:3] = ang_vel
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj[idx_mj2model]
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj[idx_mj2model]
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[idx_mj2model]
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    last_action = action
                    with torch.inference_mode():
                        if perceptive_policy:
                            local_linear_velocity = quat_rotate_inverse(
                                d.qpos[3:7], d.qvel[:3]
                            )
                            privileged = privileged_builder.build(
                                d,
                                obs,
                                local_linear_velocity,
                                tau,
                                lidar,
                                idx_mj2model,
                            )
                            result = policy(
                                obs_tensor, torch.from_numpy(privileged).unsqueeze(0)
                            )
                        else:
                            result = policy(obs_tensor)
                    if perceptive_policy:
                        action, latent = result
                        action = action.detach().numpy().squeeze()[idx_model2mj]
                        latent = latent.detach().numpy().squeeze()
                        if save_moe_latent:
                            all_latents.append(latent)
                    elif isinstance(result, tuple):
                        action, (weights, latent) = result  # moe
                        action = action.detach().numpy().squeeze()[idx_model2mj]
                        weights = weights.detach().numpy().squeeze()
                        latent = latent.detach().numpy().squeeze()
                        if visualize_moe_weights:
                            if bars is None:
                                x = np.arange(len(weights))
                                bars = ax.bar(x, weights)
                                ax.set_ylim(0, 1)
                            else:
                                for bar, w in zip(bars, weights):
                                    bar.set_height(w)

                            plt.draw()
                            plt.pause(0.001)
                        if save_moe_latent:
                            all_latents.append(latent)
                    else:
                        action = result.detach().cpu().numpy().squeeze()[idx_model2mj]
                    target_dof_pos = action * action_scale + default_angles

            # Rendering and Python-side point-cloud drawing at the 500 Hz
            # physics rate makes the viewer look like slow motion.  Render at
            # a human-visible rate and pace against accumulated simulation
            # time, so compute overhead does not add once per physics step.
            if viewer is not None and counter % render_decimation == 0:
                contact_status = get_environment_contact(m, d)
                grid_image = lidar.consume_image() if lidar is not None else None
                # The passive viewer renders on another thread.  Rebuilding
                # user geometry and changing display groups without its lock
                # lets that thread observe a half-cleared scene, which appears
                # as intermittent flashing.
                with viewer.lock():
                    normalize_view_options(viewer)
                    mujoco_render_utils.update_external_rendering(
                        viewer, ctype='viewer'
                    )
                    if lidar is not None:
                        lidar.append_point_cloud(viewer.user_scn)
                    goal_navigator.append_path(viewer.user_scn)
                if lidar is not None:
                    # These handle methods synchronize internally and must not
                    # be called while holding viewer.lock().
                    if grid_image is not None:
                        viewer.set_images(
                            (mujoco.MjrRect(10, 10, 240, 180), grid_image)
                        )
                    viewer.set_texts(
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            lidar.status_text(),
                                (
                                    f"{navigator.status_text()}\n"
                                    f"{goal_navigator.status_text()}\n"
                                    f"RTF: {realtime_factor:.2f}x\n{contact_status}"
                            ),
                        )
                    )
                viewer.sync()

            if counter % render_decimation == 0:
                simulated_elapsed = float(d.time) - simulation_clock_start
                target_wall_time = wall_clock_start + simulated_elapsed
                remaining = target_wall_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                wall_elapsed = time.perf_counter() - wall_clock_start
                if wall_elapsed > 1.0e-6:
                    realtime_factor = simulated_elapsed / wall_elapsed
            if vslam_benchmark is not None and vslam_benchmark.finished:
                # Save before leaving launch_passive(): GLFW teardown itself
                # can deadlock, before the old result-writing code is reached.
                vslam_benchmark.write_result(ros2_vslam)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)

    # writer.close()
    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()
    if save_moe_latent and len(all_latents) > 0:
        all_latents = np.array(all_latents)
        np.save(latent_path, all_latents)
        print(f"Latent vectors saved successfully to {latent_path}")
    if ros2_vslam is not None:
        if vslam_benchmark is not None:
            vslam_benchmark.write_result(ros2_vslam)
            # Destroying both the passive GLFW viewer and an off-screen
            # MuJoCo renderer can deadlock in some Mesa driver versions after
            # a long run.  The benchmark is a dedicated child process: once
            # its result is durable, let the launcher trap stop RTAB-Map and
            # save the database instead of risking an indefinite teardown.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        ros2_vslam.close()
