"""Test UDP gamepad teleoperation in the MuJoCo viewer.

This script intentionally does not create a collector or write any data.
Start the Windows sender first, then run:

    python -m data_collection.test_teleop
"""

import argparse
import time

import numpy as np
from hydra.utils import instantiate

from infra.hardware.spacemouse.wsl_recevier import UDPGamepadExpert
from workspace import SERLWorkspace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="insert_sim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--manual-test",
        action="store_true",
        help="Ignore UDP and move in +x for a short diagnostic test.",
    )
    parser.add_argument(
        "--action-sign",
        type=float,
        nargs=6,
        default=[1, -1, 1, 1, 1, 1],
        metavar=("X", "Y", "Z", "R", "P", "YAW"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    workspace = SERLWorkspace(args.task)
    env = instantiate(
        workspace.raw_config.environment,
        fake_env=False,
        seed=args.seed,
    )

    teleop = UDPGamepadExpert(
        host="0.0.0.0",
        port=args.port,
        stale_timeout=args.stale_timeout,
        action_sign=args.action_sign,
    )

    try:
        obs, _ = env.reset(seed=args.seed)
        print("MuJoCo viewer started.")
        print("Move the gamepad to control the end effector.")
        print("Press Ctrl+C to stop. Release the sender/deadman to stop motion.")

        step = 0
        while True:
            if args.manual_test:
                # This isolates MuJoCo/control from Windows and UDP.
                action = np.array([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
            else:
                action = np.asarray(teleop.get_action(obs), dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % args.print_every == 0:
                status = teleop.status()
                print(
                    f"step={step:05d} action={np.round(action, 3)} "
                    f"reward={reward} success={info.get('succeed', False)} "
                    f"packets={status['packet_count']} "
                    f"age={status['age']:.3f}s sender={status['sender']}"
                )

            if terminated or truncated:
                print(
                    "Episode finished: "
                    f"success={info.get('succeed', False)}; resetting."
                )
                obs, _ = env.reset()

            step += 1

            # The environment already advances physics at control_dt. This
            # small sleep prevents a tight Python loop from starving the UI.
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping teleoperation test.")
    finally:
        teleop.close()
        env.close()


if __name__ == "__main__":
    main()
