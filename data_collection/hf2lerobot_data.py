import os
import glob
import shutil
from pathlib import Path
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# 兼容不同版本 LeRobot 导包路径
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets import LeRobotDataset


def decode_image(raw_data: np.ndarray) -> np.ndarray:
    """自动兼容处理原始图像矩阵或压缩后的 JPEG 字节流"""
    if raw_data.ndim == 1 or isinstance(raw_data, (bytes, bytearray)):
        # 针对以 JPEG 编码压缩存储的 ALOHA HDF5
        img = cv2.imdecode(np.asarray(raw_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码压缩图像")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif raw_data.ndim == 3:
        # 已经是 (H, W, C) 形状
        img = raw_data
        # 若原数据存储为 BGR，请按需解除注释：
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"无法识别的图像数据维度: {raw_data.shape}")
    return img


def convert_hdf5_to_lerobot(
    hdf5_dir: str,
    repo_id: str,
    robot_type: str = "aloha",
    fps: int = 50,
    output_dir: str | None = None,
    task_desc: str = "Perform the manipulation task",
):
    """
    将 HDF5 录像转为 LeRobot 标准数据集。
    
    参数:
        hdf5_dir: 存放 episode_*.hdf5 文件的目录
        repo_id: 数据集 ID（如 'local/aloha_pick_cube'）
        robot_type: 机器人型号标签
        fps: 采集帧率
        output_dir: 自定义输出目录（留空则保存在 ~/.cache/huggingface/lerobot/{repo_id}）
        task_desc: 任务语言描述
    """
    hdf5_files = sorted(
        [*Path(hdf5_dir).glob("*.h5"), *Path(hdf5_dir).glob("*.hdf5")]
    )
    if not hdf5_files:
        raise FileNotFoundError(f"在 {hdf5_dir} 下未找到任何 .hdf5 文件")

    print(f"共发现 {len(hdf5_files)} 个 Episode，正在读取首个文件提取元信息...")

    # 1. 探查第一个文件，动态获取特征维度与分辨率
    with h5py.File(hdf5_files[0], "r") as sample_h5:
        # 状态与动作维度
        state_dim = sample_h5["/observations/state"].shape[1]
        action_dim = sample_h5["/actions"].shape[1]
        
        # 探测相机配置与分辨率
        camera_names = list(sample_h5["/observations/images"].keys())
        first_cam = camera_names[0]
        sample_img = decode_image(sample_h5[f"/observations/images/{first_cam}"][0])
        img_h, img_w, img_c = sample_img.shape

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 相机列表: {camera_names}, 分辨率: {img_w}x{img_h}")
    obs_static = {0:"x", 1:"y", 2:"z", 3:"roll", 4:"pitch", 5:"yaw",6:"gripper"
    }
    action_static = {0:"x", 1:"y", 2:"z", 3:"roll", 4:"pitch", 5:"yaw",6:"gripper"
    }
    # 2. 定义 LeRobot 特征字典
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"{obs_static[i]}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"{action_static[i]}" for i in range(action_dim)],
        },
    }

    # 动态添加相机特征（推荐使用 video 格式自动转为 MP4 减小体积）
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, img_c),
            "names": ["height", "width", "channels"],
        }

    # 3. 创建 LeRobot 空数据集
    # 若输出目录已存在同名数据，先清理以防冲突
    if output_dir:
        target_path = Path(output_dir) / repo_id
        if target_path.exists():
            shutil.rmtree(target_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        root=output_dir,
    )

    # 4. 逐 Episode、逐 Frame 写入数据
    for ep_idx, ep_path in enumerate(tqdm(hdf5_files, desc="转换进度")):
        with h5py.File(ep_path, "r") as h5:
            qpos = np.array(h5["/observations/state"], dtype=np.float32)
            actions = np.array(h5["/actions"], dtype=np.float32)
            
            if len(qpos) != len(actions):
                raise ValueError(
                    f"{ep_path}: state/action 帧数不一致: "
                    f"{len(qpos)} != {len(actions)}"
                )
            num_frames = len(qpos)

            # 读取各相机帧
            cam_data = {}
            for cam in camera_names:
                cam_data[cam] = h5[f"/observations/images/{cam}"]

            # 逐帧填充
            for t in range(num_frames):
                frame = {
                    "observation.state": qpos[t],
                    "action": actions[t],
                    "task": task_desc,
                }
                for cam in camera_names:
                    frame[f"observation.images.{cam}"] = decode_image(cam_data[cam][t])

                dataset.add_frame(frame)

            # 保存当前 Episode（写入 Parquet 与视频切片）
            dataset.save_episode()

    # 5. 固化数据集并计算均值/方差等归一化元数据
    print("正在固化数据集并计算统计信息 (Normalize Stats)...")
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(f"转换完成！数据集已保存至: {dataset.root}")


if __name__ == "__main__":
    convert_hdf5_to_lerobot(
        hdf5_dir="./data/insert",
        repo_id="local/my_converted_dataset",
        robot_type="aloha",
        fps=50,
        task_desc="insert the cube into the hole",
    )
