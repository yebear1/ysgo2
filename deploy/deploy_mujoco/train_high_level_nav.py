#!/usr/bin/env python3
"""Train a compact PPO policy for goal-directed indoor velocity control.

The environment is a vectorized 2-D family of randomized rooms.  The policy
only sees a local goal, body velocity, previous command and planar LiDAR.  It
therefore transfers directly to the MuJoCo bridge while the existing terrain
navigator remains the hard safety layer.
"""

import argparse
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.distributions import Normal


OBSERVATION_SIZE = 33
LIDAR_RAY_COUNT = 24


class NavigationWorld:
    def __init__(self, count, device="cpu", seed=7):
        self.count = int(count)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.dt = 0.20
        self.room_half_size = 4.0
        self.robot_radius = 0.25
        self.max_lidar_range = 4.0
        self.obstacle_count = 10
        self.max_steps = 180
        self.position = torch.zeros(self.count, 2, device=self.device)
        self.yaw = torch.zeros(self.count, device=self.device)
        self.goal = torch.zeros(self.count, 2, device=self.device)
        self.velocity = torch.zeros(self.count, 3, device=self.device)
        self.previous_command = torch.zeros(self.count, 3, device=self.device)
        self.obstacle_xy = torch.zeros(
            self.count, self.obstacle_count, 2, device=self.device
        )
        self.obstacle_radius = torch.zeros(
            self.count, self.obstacle_count, device=self.device
        )
        self.steps = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self.episode_successes = 0
        self.episode_collisions = 0
        self.episode_count = 0
        self.ray_angles = torch.linspace(
            -math.pi,
            math.pi,
            LIDAR_RAY_COUNT + 1,
            device=self.device,
        )[:-1]
        self.reset(torch.arange(self.count, device=self.device))

    def _rand(self, *shape):
        return torch.rand(
            *shape, device=self.device, generator=self.generator
        )

    def reset(self, indices):
        if indices.numel() == 0:
            return
        n = indices.numel()
        start = (self._rand(n, 2) * 2.0 - 1.0) * 2.7
        angle = self._rand(n) * 2.0 * math.pi
        distance = 2.0 + self._rand(n) * 3.0
        goal = start + torch.stack((angle.cos(), angle.sin()), dim=1) * distance[:, None]
        goal = goal.clamp(-3.35, 3.35)
        too_close = torch.linalg.norm(goal - start, dim=1) < 1.25
        goal[too_close] = -start[too_close]

        obstacles = (self._rand(n, self.obstacle_count, 2) * 2.0 - 1.0) * 3.45
        radii = 0.18 + self._rand(n, self.obstacle_count) * 0.38
        start_distance = torch.linalg.norm(obstacles - start[:, None, :], dim=2)
        goal_distance = torch.linalg.norm(obstacles - goal[:, None, :], dim=2)
        invalid = (start_distance < radii + 0.65) | (goal_distance < radii + 0.55)
        radii[invalid] = 0.0

        self.position[indices] = start
        self.goal[indices] = goal
        self.yaw[indices] = self._rand(n) * 2.0 * math.pi - math.pi
        self.velocity[indices] = 0.0
        self.previous_command[indices] = 0.0
        self.obstacle_xy[indices] = obstacles
        self.obstacle_radius[indices] = radii
        self.steps[indices] = 0

    def _lidar(self):
        angles = self.yaw[:, None] + self.ray_angles[None, :]
        directions = torch.stack((angles.cos(), angles.sin()), dim=2)

        positive_x = (self.room_half_size - self.position[:, 0:1]) / directions[:, :, 0].clamp_min(1.0e-6)
        negative_x = (-self.room_half_size - self.position[:, 0:1]) / directions[:, :, 0].clamp_max(-1.0e-6)
        positive_y = (self.room_half_size - self.position[:, 1:2]) / directions[:, :, 1].clamp_min(1.0e-6)
        negative_y = (-self.room_half_size - self.position[:, 1:2]) / directions[:, :, 1].clamp_max(-1.0e-6)
        wall_ranges = torch.minimum(
            torch.where(directions[:, :, 0] >= 0.0, positive_x, negative_x),
            torch.where(directions[:, :, 1] >= 0.0, positive_y, negative_y),
        )

        relative = self.obstacle_xy[:, :, None, :] - self.position[:, None, None, :]
        ray = directions[:, None, :, :]
        projection = (relative * ray).sum(dim=3)
        perpendicular_sq = (relative * relative).sum(dim=3) - projection.square()
        radius_sq = self.obstacle_radius[:, :, None].square()
        discriminant = radius_sq - perpendicular_sq
        valid = (self.obstacle_radius[:, :, None] > 0.0) & (projection > 0.0) & (discriminant >= 0.0)
        hit = projection - discriminant.clamp_min(0.0).sqrt()
        hit = torch.where(valid & (hit > 0.0), hit, torch.full_like(hit, 1.0e6))
        obstacle_ranges = hit.amin(dim=1)
        return torch.minimum(wall_ranges, obstacle_ranges).clamp(0.0, self.max_lidar_range)

    def observation(self):
        delta = self.goal - self.position
        cy, sy = self.yaw.cos(), self.yaw.sin()
        local_x = cy * delta[:, 0] + sy * delta[:, 1]
        local_y = -sy * delta[:, 0] + cy * delta[:, 1]
        distance = torch.sqrt(local_x.square() + local_y.square()).clamp_min(1.0e-6)
        goal_features = torch.stack(
            (
                (distance / 4.0).clamp_max(1.0),
                local_x / distance,
                local_y / distance,
            ),
            dim=1,
        )
        velocity_scale = torch.tensor([0.8, 0.4, 1.2], device=self.device)
        lidar = self._lidar() / self.max_lidar_range
        return torch.cat(
            (
                goal_features,
                (self.velocity / velocity_scale).clamp(-1.5, 1.5),
                (self.previous_command / velocity_scale).clamp(-1.0, 1.0),
                lidar,
            ),
            dim=1,
        )

    def step(self, action):
        action = action.clamp(-1.0, 1.0)
        command = torch.stack(
            (
                0.4 * (action[:, 0] + 1.0),
                0.4 * action[:, 1],
                1.2 * action[:, 2],
            ),
            dim=1,
        )
        old_distance = torch.linalg.norm(self.goal - self.position, dim=1)
        self.velocity = 0.55 * self.velocity + 0.45 * command
        cy, sy = self.yaw.cos(), self.yaw.sin()
        world_velocity = torch.stack(
            (
                cy * self.velocity[:, 0] - sy * self.velocity[:, 1],
                sy * self.velocity[:, 0] + cy * self.velocity[:, 1],
            ),
            dim=1,
        )
        candidate = self.position + world_velocity * self.dt
        self.yaw = torch.atan2(
            torch.sin(self.yaw + self.velocity[:, 2] * self.dt),
            torch.cos(self.yaw + self.velocity[:, 2] * self.dt),
        )

        obstacle_distance = torch.linalg.norm(
            candidate[:, None, :] - self.obstacle_xy, dim=2
        )
        collision = (
            (self.obstacle_radius > 0.0)
            & (obstacle_distance < self.obstacle_radius + self.robot_radius)
        ).any(dim=1)
        collision |= (candidate.abs() > self.room_half_size - self.robot_radius).any(dim=1)
        self.position = torch.where(collision[:, None], self.position, candidate)
        new_distance = torch.linalg.norm(self.goal - self.position, dim=1)
        reached = new_distance < 0.35
        self.steps += 1
        timeout = self.steps >= self.max_steps

        progress = old_distance - new_distance
        smoothness = (command - self.previous_command).square().sum(dim=1)
        goal_delta = self.goal - self.position
        goal_distance = torch.linalg.norm(goal_delta, dim=1).clamp_min(1.0e-6)
        heading_cosine = (
            self.yaw.cos() * goal_delta[:, 0]
            + self.yaw.sin() * goal_delta[:, 1]
        ) / goal_distance
        reward = 6.0 * progress + 0.025 * heading_cosine - 0.01
        reward -= 0.025 * smoothness
        reward -= collision.float() * 2.5
        reward += reached.float() * 6.0
        done = collision | reached | timeout
        self.previous_command = command

        finished = done.nonzero(as_tuple=False).flatten()
        if finished.numel():
            self.episode_successes += int(reached[finished].sum().item())
            self.episode_collisions += int(collision[finished].sum().item())
            self.episode_count += int(finished.numel())
            self.reset(finished)
        return self.observation(), reward, done


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((3,), -0.25))

    def distribution(self, observation):
        return Normal(self.actor(observation), self.log_std.exp())

    def value(self, observation):
        return self.critic(observation).squeeze(1)


class DeterministicActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        return self.actor(observation)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    world = NavigationWorld(args.num_envs, device=device, seed=args.seed)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    observation = world.observation()
    gamma = 0.99
    gae_lambda = 0.95
    clip = 0.20
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        observations = []
        actions = []
        log_probabilities = []
        rewards = []
        dones = []
        values = []
        with torch.no_grad():
            for _ in range(args.horizon):
                distribution = model.distribution(observation)
                action = distribution.sample()
                observations.append(observation)
                actions.append(action)
                log_probabilities.append(distribution.log_prob(action).sum(dim=1))
                values.append(model.value(observation))
                observation, reward, done = world.step(action)
                rewards.append(reward)
                dones.append(done)
            next_value = model.value(observation)

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        old_log_probabilities = torch.stack(log_probabilities)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(args.num_envs, device=device)
        for step in reversed(range(args.horizon)):
            mask = 1.0 - dones[step].float()
            following_value = next_value if step == args.horizon - 1 else values[step + 1]
            delta = rewards[step] + gamma * following_value * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
        returns = advantages + values

        flat_observations = observations.flatten(0, 1)
        flat_actions = actions.flatten(0, 1)
        flat_old_log_probabilities = old_log_probabilities.flatten()
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std() + 1.0e-8)
        sample_count = flat_observations.shape[0]

        for _ in range(args.epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, args.minibatch_size):
                batch = permutation[start : start + args.minibatch_size]
                distribution = model.distribution(flat_observations[batch])
                log_probability = distribution.log_prob(flat_actions[batch]).sum(dim=1)
                ratio = (log_probability - flat_old_log_probabilities[batch]).exp()
                unclipped = ratio * flat_advantages[batch]
                clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * flat_advantages[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (
                    model.value(flat_observations[batch]) - flat_returns[batch]
                ).square().mean()
                entropy = distribution.entropy().sum(dim=1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.004 * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                optimizer.step()

        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            episodes = max(world.episode_count, 1)
            success_rate = world.episode_successes / episodes
            collision_rate = world.episode_collisions / episodes
            elapsed = time.time() - start_time
            print(
                f"iteration={iteration:04d} reward={rewards.mean().item():+.3f} "
                f"success={success_rate:.1%} collision={collision_rate:.1%} "
                f"episodes={world.episode_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    actor = DeterministicActor(model.actor.cpu()).eval()
    scripted = torch.jit.script(actor)
    scripted.save(str(output))
    checkpoint = output.with_suffix(".checkpoint.pt")
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "iterations": args.iterations,
            "observation_size": OBSERVATION_SIZE,
        },
        checkpoint,
    )
    print(f"policy={output}")
    print(f"checkpoint={checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="deploy/pre_train/go2/high_level_nav.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
