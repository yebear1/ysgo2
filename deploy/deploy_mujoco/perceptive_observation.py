"""MuJoCo reconstruction of the privileged observation used by the CTS teacher."""

import mujoco
import numpy as np


class PerceptiveObservationBuilder:
    def __init__(self, model, control_dt):
        self.model = model
        self.control_dt = float(control_dt)
        self.previous_dof_velocity = np.zeros(12, dtype=np.float32)
        self.torque_limits = np.asarray(
            [23.7, 23.7, 35.55] * 4, dtype=np.float32
        )
        self.foot_body_ids = [
            model.body(name).id for name in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
        ]
        self.body_to_foot = {
            body_id: foot_index for foot_index, body_id in enumerate(self.foot_body_ids)
        }

    def reset(self, dof_velocity=None):
        if dof_velocity is None:
            self.previous_dof_velocity.fill(0.0)
        else:
            self.previous_dof_velocity[:] = dof_velocity

    def _foot_contact_norms(self, data):
        # Isaac Gym exposes the resultant force on each foot. Reconstruct a
        # close MuJoCo equivalent by accumulating contact-force magnitudes.
        forces = np.zeros(4, dtype=np.float32)
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            foot_index = self.body_to_foot.get(body1)
            if foot_index is None:
                foot_index = self.body_to_foot.get(body2)
            if foot_index is None:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, contact_force)
            forces[foot_index] += np.linalg.norm(contact_force[:3])
        return forces * 1.0e-3

    def build(
        self,
        data,
        observation,
        local_linear_velocity,
        applied_torque,
        lidar,
        mj_to_model,
    ):
        velocity = np.asarray(data.qvel[6:], dtype=np.float32)
        acceleration = (self.previous_dof_velocity - velocity) / self.control_dt * 1.0e-4
        self.previous_dof_velocity[:] = velocity

        torque = np.asarray(applied_torque, dtype=np.float32)[mj_to_model]
        acceleration = acceleration[mj_to_model]
        privileged = np.concatenate(
            (
                np.asarray(local_linear_velocity, dtype=np.float32) * 2.0,
                np.asarray(observation, dtype=np.float32),
                self._foot_contact_norms(data),
                torque / self.torque_limits,
                acceleration,
                lidar.policy_height_observation(float(data.qpos[2])),
            )
        ).astype(np.float32)
        if privileged.shape != (263,):
            raise RuntimeError(f"Expected 263 privileged values, got {privileged.shape}")
        return privileged
