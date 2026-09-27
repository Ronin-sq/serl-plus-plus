import time
import numpy as np

from workspace import SERLWorkspace

workspace = SERLWorkspace("insert_sim")

# fake_env=False 会启用 MuJoCo human viewer
env = workspace.get_environment(fake_env=False, seed=0)

obs, info = env.reset(seed=0)
print("Environment started.")
print("Press Ctrl+C to stop.")

try:
    while True:
        # 零动作：保持当前位置
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"Episode finished: reward={reward}, "
                f"succeed={info.get('succeed', False)}"
            )
            obs, info = env.reset()

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping viewer...")

finally:
    env.close()