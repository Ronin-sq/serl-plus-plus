from pathlib import Path

import h5py
import numpy as np


class HDF5EpisodeWriter:
    def __init__(self, output_dir="data/insert_hdf5"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def flatten_state(obs):
        state = obs["state"]

        return np.concatenate(
            [
                state["tcp_pose"],
                # state["tcp_vel"],
                # state["tcp_force"],
                # state["tcp_torque"],
            ]
        ).astype(np.float32)

    def save_episode(
        self,
        episode_id,
        observations,
        actions,
        rewards,
        terminated,
        truncated,
        success,
        metadata,
    ):
        if not observations:
            raise ValueError("Cannot save an empty episode")

        path = self.output_dir / f"episode_{episode_id:06d}.h5"

        wrist1 = np.stack(
            [obs["images"]["wrist1"] for obs in observations]
        ).astype(np.uint8)

        wrist2 = np.stack(
            [obs["images"]["wrist2"] for obs in observations]
        ).astype(np.uint8)

        states = np.stack(
            [self.flatten_state(obs) for obs in observations]
        ).astype(np.float32)

        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        success = np.asarray(success, dtype=np.bool_)

        episode_length = len(actions)

        for name, array in {
            "states": states,
            "wrist1": wrist1,
            "wrist2": wrist2,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
        }.items():
            if len(array) != episode_length:
                raise ValueError(
                    f"{name} has length {len(array)}, "
                    f"expected {episode_length}"
                )

        with h5py.File(path, "w") as file:
            observation_group = file.create_group("observations")
            image_group = observation_group.create_group("images")

            image_group.create_dataset(
                "wrist1",
                data=wrist1,
                compression="gzip",
                compression_opts=4,
                chunks=(1, *wrist1.shape[1:]),
            )

            image_group.create_dataset(
                "wrist2",
                data=wrist2,
                compression="gzip",
                compression_opts=4,
                chunks=(1, *wrist2.shape[1:]),
            )

            observation_group.create_dataset(
                "state",
                data=states,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )

            file.create_dataset("actions", data=actions)
            file.create_dataset("rewards", data=rewards)
            file.create_dataset("terminated", data=terminated)
            file.create_dataset("truncated", data=truncated)
            file.create_dataset("success", data=success)

            file.attrs["episode_id"] = int(episode_id)
            file.attrs["episode_length"] = episode_length
            file.attrs["episode_success"] = bool(success[-1])

            for key, value in metadata.items():
                if isinstance(value, str) or np.isscalar(value):
                    file.attrs[key] = value
                else:
                    file.attrs[key] = np.asarray(value)

        return path