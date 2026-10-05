"""UDP gamepad receiver for teleoperation from Windows/WSL."""

import json
import socket
import threading
import time

import numpy as np


class UDPGamepadExpert:
    """Expose the latest UDP command as a normalized 6D environment action."""

    def __init__(
        self,
        host="0.0.0.0",
        port=9999,
        stale_timeout=0.25,
        recv_timeout=0.1,
        action_sign=None,
    ):
        self.stale_timeout = float(stale_timeout)
        self._action_sign = np.ones(6, dtype=np.float32)
        if action_sign is not None:
            signs = np.asarray(action_sign, dtype=np.float32)
            if signs.shape != (6,):
                raise ValueError("action_sign must have shape (6,)")
            self._action_sign = signs

        self._lock = threading.Lock()
        self._latest_action = np.zeros(6, dtype=np.float32)
        self._latest_payload = None
        self._latest_sender = None
        self._packet_count = 0
        self._last_packet_time = 0.0
        self._running = True

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, int(port)))
        self._socket.settimeout(float(recv_timeout))

        self._thread = threading.Thread(
            target=self._receive_loop,
            name="udp-gamepad-receiver",
            daemon=True,
        )
        self._thread.start()
        print(f"UDP gamepad receiver listening on {host}:{port}")

    @staticmethod
    def _payload_to_action(payload):
        linear = payload.get("linear", {})
        angular = payload.get("angular", {})
        action = np.asarray(
            [
                linear.get("x", 0.0),
                linear.get("y", 0.0),
                linear.get("z", 0.0),
                angular.get("roll", 0.0),
                angular.get("pitch", 0.0),
                angular.get("yaw", 0.0),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

    def _receive_loop(self):
        while self._running:
            try:
                data, sender = self._socket.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                action = self._payload_to_action(payload)
                action = np.clip(action * self._action_sign, -1.0, 1.0)

                with self._lock:
                    self._latest_action = action
                    self._latest_payload = payload
                    self._latest_sender = sender
                    self._packet_count += 1
                    self._last_packet_time = time.monotonic()
            except socket.timeout:
                continue
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                continue

    def get_action(self, _obs=None):
        """Return the latest action, or zeros when the UDP stream is stale."""
        with self._lock:
            action = self._latest_action.copy()
            last_packet_time = self._last_packet_time

        if (
            last_packet_time <= 0.0
            or time.monotonic() - last_packet_time > self.stale_timeout
        ):
            return np.zeros(6, dtype=np.float32)
        return action

    def __call__(self, obs):
        return self.get_action(obs)

    def latest_payload(self):
        with self._lock:
            return None if self._latest_payload is None else dict(self._latest_payload)

    def status(self):
        """Return diagnostics useful for distinguishing UDP from env issues."""
        with self._lock:
            packet_count = self._packet_count
            sender = self._latest_sender
            last_packet_time = self._last_packet_time
            action = self._latest_action.copy()

        age = float("inf") if last_packet_time <= 0 else time.monotonic() - last_packet_time
        return {
            "packet_count": packet_count,
            "sender": sender,
            "age": age,
            "action": action,
        }

    def close(self):
        self._running = False
        self._socket.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


if __name__ == "__main__":
    receiver = UDPGamepadExpert()
    try:
        while True:
            print(receiver.get_action())
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
