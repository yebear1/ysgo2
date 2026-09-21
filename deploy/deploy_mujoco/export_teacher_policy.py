#!/usr/bin/env python3
"""Export the privileged (terrain-aware) teacher from a CTS checkpoint."""

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS


class TeacherPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.teacher_encoder = copy.deepcopy(model.teacher_encoder).cpu().eval()
        self.actor = copy.deepcopy(model.actor).cpu().eval()

    def forward(self, observation, privileged_observation):
        latent = self.teacher_encoder(privileged_observation)
        return self.actor(torch.cat((latent, observation), dim=1)), latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = ActorCriticMoECTS(
        num_obs=45,
        num_critic_obs=263,
        num_actions=12,
        num_envs=1,
        history_length=5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256, 256],
        expert_num=8,
        activation="elu",
        latent_dim=32,
        norm_type="l2norm",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher = torch.jit.script(TeacherPolicy(model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    teacher.save(str(args.output))

    obs = torch.zeros(1, 45)
    privileged = torch.zeros(1, 263)
    action, latent = teacher(obs, privileged)
    print(f"saved={args.output}")
    print(f"action_shape={tuple(action.shape)} latent_shape={tuple(latent.shape)}")


if __name__ == "__main__":
    main()
