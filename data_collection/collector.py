from copy import deepcopy

import numpy as np
from hydra.utils import instantiate

from data_collection.hdf5_writer import HDF5EpisodeWriter
from workspace import SERLWorkspace


class InsertCollector:
    def __init__(
        self,
        task_name="insert_sim",
        seed=0,
        output_dir="data/insert_hdf5",
    ):
        self.task_name = task_name
        self.seed = seed

        workspace = SERLWorkspace(task_name)

        self.env = instantiate(
            workspace.raw_config.environment,
            fake_env=True,
            seed=seed,
        )

        self.writer = HDF5EpisodeWriter(output_dir=output_dir)

    def collect_episode(
        self,
        policy,
        language_instruction="Insert the peg into the hole",
        episode_id=0,
        max_steps=None,
    ):
        episode_seed = self.seed + episode_id
        obs, _ = self.env.reset(seed=episode_seed)

        observations = []
        actions = []
        rewards = []
        terminated_flags = []
        truncated_flags = []
        success_flags = []

        if max_steps is None:
            max_steps = self.env.unwrapped._max_episode_length

        for _ in range(max_steps):
            action = np.asarray(policy(obs), dtype=np.float32)

            if action.shape != self.env.action_space.shape:
                raise ValueError(
                    f"Expected action shape {self.env.action_space.shape}, "
                    f"got {action.shape}"
                )

            action = np.clip(
                action,
                self.env.action_space.low,
                self.env.action_space.high,
            )

            next_obs, reward, terminated, truncated, step_info = (
                self.env.step(action)
            )

            observations.append(deepcopy(obs))
            full_action = np.append(action, 1.0)
            actions.append(full_action.copy())
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            success_flags.append(bool(step_info.get("succeed", False)))

            obs = next_obs

            if terminated or truncated:
                break

        episode_success = bool(success_flags[-1]) if success_flags else False

        self.writer.save_episode(
            episode_id=episode_id,
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminated=terminated_flags,
            truncated=truncated_flags,
            success=success_flags,
            metadata={
                "task": self.task_name,
                "language_instruction": language_instruction,
                "seed": episode_seed,
                "fps": round(1.0 / self.env.control_dt),
                "control_dt": self.env.control_dt,
                "physics_dt": self.env.physics_dt,
                "action_type": "cartesian_delta",
                "action_frame": "world",
                "action_scale": [0.01, 0.02],
                "pose_representation": "xyz_quaternion",
                "state_keys": (
                    "tcp_pose,tcp_vel,tcp_force,tcp_torque"
                ),
            },
        )

        return {
            "episode_id": f"insert_sim_{episode_id:06d}",
            "length": len(actions),
            "success": episode_success,
        }

    def close(self):
        self.env.close()