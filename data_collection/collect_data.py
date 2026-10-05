import argparse

import numpy as np

from data_collection.collector import InsertCollector
from infra.hardware.spacemouse.wsl_recevier import UDPGamepadExpert


def zero_policy(obs):
    # 当前只是测试数据管线
    return np.zeros(6, dtype=np.float32)


class OpenCVVisualizer:
    def __init__(self, window_name="insert teleoperation"):
        import cv2

        self.cv2 = cv2
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def __call__(
        self,
        obs,
        action,
        next_obs,
        reward,
        terminated,
        truncated,
        info,
        timestep,
    ):
        cv2 = self.cv2
        frames = []

        for camera_name in ("wrist1", "wrist2"):
            # Display the state after applying the current action.
            frame = np.asarray(next_obs["images"][camera_name])
            frame = np.ascontiguousarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
            frames.append(frame)

        canvas = cv2.hconcat(frames)
        text_lines = [
            f"step={timestep} reward={reward:.1f} action={np.round(action, 2)}",
            f"success={info.get('succeed', False)} done={terminated or truncated}",
            "q / ESC: stop episode",
        ]

        for index, line in enumerate(text_lines):
            cv2.putText(
                canvas,
                line,
                (10, 28 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True

        return False

    def close(self):
        self.cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()

    collector = InsertCollector(
        task_name="insert_sim",
        seed=0,
        output_dir="data/insert",
    )

    teleop = UDPGamepadExpert(
        host="0.0.0.0",
        port=9999,
        stale_timeout=0.25,
        # Change signs here after checking the physical direction of each axis.
        action_sign=[1, -1, 1, 1, 1, 1],
    )

    visualizer = OpenCVVisualizer() if args.visualize else None

    try:
        for episode_id in range(args.episodes):
            episode = collector.collect_episode(
                policy=teleop,
                episode_id=episode_id,
                max_steps=args.max_steps,
                step_callback=visualizer,
            )

            print(
                episode["episode_id"],
                "length=",
                episode["length"],
                "success=",
                episode["success"],
            )
    finally:
        if visualizer is not None:
            visualizer.close()
        teleop.close()
        collector.close()


if __name__ == "__main__":
    main()
