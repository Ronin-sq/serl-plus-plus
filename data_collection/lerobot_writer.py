from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class InsertLeRobotWriter:
    def __init__(
        self,
        root="data/lerobot/insert_sim",
        repo_id="local/insert_sim",
        fps=50,
        image_height=128,
        image_width=128,
        state_dim=19,
        action_dim=6,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.root,
            fps=fps,
            robot_type="franka_panda",
            features={
                "observation.images.wrist1": {
                    "dtype": "video",
                    "shape": (image_height, image_width, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.wrist2": {
                    "dtype": "video",
                    "shape": (image_height, image_width, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["action"],
                },
                "next.reward": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["reward"],
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["done"],
                },
                "next.success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["success"],
                },
            },
        )

    @staticmethod
    def flatten_state(obs):
        state = obs["state"]

        return np.concatenate(
            [
                state["tcp_pose"],
                state["tcp_vel"],
                state["tcp_force"],
                state["tcp_torque"],
            ]
        ).astype(np.float32)

    def add_frame(
        self,
        obs,
        action,
        reward,
        done,
        success,
        task,
    ):
        frame = {
            "observation.images.wrist1": obs["images"]["wrist1"],
            "observation.images.wrist2": obs["images"]["wrist2"],
            "observation.state": self.flatten_state(obs),
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": np.asarray([reward], dtype=np.float32),
            "next.done": np.asarray([done], dtype=np.bool_),
            "next.success": np.asarray([success], dtype=np.bool_),
        }

        self.dataset.add_frame(frame, task=task)

    def end_episode(self):
        self.dataset.save_episode()

    def finalize(self):
        self.dataset.finalize()