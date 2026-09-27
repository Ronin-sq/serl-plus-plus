import numpy as np

from data_collection.collector import InsertCollector


def zero_policy(obs):
    # 当前只是测试数据管线
    return np.zeros(6, dtype=np.float32)


def main():
    collector = InsertCollector(
        task_name="insert_sim",
        seed=0,
        output_dir="data/insert",
    )

    try:
        for episode_id in range(1):
            episode = collector.collect_episode(
                policy=zero_policy,
                episode_id=episode_id,
            )

            print(
                episode["episode_id"],
                "length=",
                episode["length"],
                "success=",
                episode["success"],
            )
    finally:
        collector.close()


if __name__ == "__main__":
    main()