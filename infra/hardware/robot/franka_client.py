from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import requests


ArrayLike = Union[float, int, Sequence[float], np.ndarray]

@dataclass(frozen=True)
class FrankaClientConfig:
    base_url: str = "http://127.0.0.1:5000"
    timeout: float = 5.0

class FrankaApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 5.0):
        self._config = FrankaClientConfig(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @staticmethod
    def _to_ndarray(payload: Any) -> np.ndarray:
        return np.asarray(payload, dtype=float)

    def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        url = f"{self._config.base_url}/{endpoint.lstrip('/')}"
        response = self._session.post(
            url,
            json=json_data,
            timeout=self._config.timeout if timeout is None else timeout,
        )
        response.raise_for_status()

        if not response.text:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def set_load(
        self,
        mass: float,
        center_of_mass: ArrayLike,
        load_inertia: ArrayLike,
    ) -> None:
        self._post("/set_load",
            {
                "mass": float(mass),
                "center_of_mass": np.asarray(center_of_mass, dtype=float).tolist(),
                "load_inertia": np.asarray(load_inertia, dtype=float).tolist(),
            },
        )

    def start_impedance(self) -> None:
        self._post("/startimp")

    def stop_impedance(self) -> None:
        self._post("/stopimp")

    def getpos(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getpos")["pose"])

    def get_vel(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getvel")["vel"])

    def get_force(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getforce")["force"])

    def get_torque(self) -> np.ndarray:
        return self._to_ndarray(self._post("/gettorque")["torque"])

    def get_q(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getq")["q"])

    def get_dq(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getdq")["dq"])

    def get_jacobian(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getjacobian")["jacobian"])

    def get_pos_euler(self) -> np.ndarray:
        return self._to_ndarray(self._post("/getpos_euler")["pose"])

    def get_state(self) -> Dict[str, np.ndarray]:
        data = self._post("/getstate")
        return {
            "pose": self._to_ndarray(data["pose"]),
            "vel": self._to_ndarray(data["vel"]),
            "force": self._to_ndarray(data["force"]),
            "torque": self._to_ndarray(data["torque"]),
            "q": self._to_ndarray(data["q"]),
            "dq": self._to_ndarray(data["dq"]),
            "jacobian": self._to_ndarray(data["jacobian"]),
            "gripper_pos": self._to_ndarray([data["gripper_pos"]]),
        }

    def pose(self, arr: ArrayLike) -> None:
        pose_arr = np.asarray(arr, dtype=float)
        self._post("/pose", {"arr": pose_arr.tolist()})

    def joint_reset(
        self,
        target_joint_positions: Sequence[float],
        motion_duration: float = 10.0,
    ) -> None:
        self._post(
            "/jointreset",
            {
                "target_joint_positions": list(target_joint_positions),
                "motion_duration": motion_duration,
            },
            timeout=max(self._config.timeout, motion_duration + 60.0),
        )

    def clear_errors(self) -> None:
        self._post("/clearerr")

    def open_gripper(self) -> None:
        self._post("/open_gripper")

    def close_gripper(self) -> None:
        self._post("/close_gripper")

    def close_gripper_slow(self) -> None:
        self._post("/close_gripper_slow")

    def move_gripper(self, position: int) -> None:
        self._post("/move_gripper", {"position": position})

    def get_gripper(self) -> float:
        return float(self._post("/get_gripper")["gripper_pos"])

    def update_param(self, values: Mapping[str, Any]) -> None:
        payload: Dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.astype(float).tolist()
            elif isinstance(value, (list, tuple)):
                payload[key] = [
                    float(item)
                    if isinstance(
                        item,
                        (float, int, np.floating, np.integer),
                    )
                    else item
                    for item in value
                ]
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                payload[key] = float(value)
            else:
                payload[key] = value
        self._post("/update_param", payload)