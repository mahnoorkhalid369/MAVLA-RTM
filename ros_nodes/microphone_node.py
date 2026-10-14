#!/usr/bin/env python3
"""ROS 2 node: the robot's microphone, the input for spoken instructions.

Isaac Sim does not simulate sound, so this node records the host computer's real microphone
(through ALSA's `arecord`, no extra Python packages) and publishes it. The simulated scene
places the microphone at the `microphone_link` TF frame (see sim/scene.py, MIC_POSITION).

    source /opt/ros/jazzy/setup.bash
    python3 ros_nodes/microphone_node.py                  # default input device
    python3 ros_nodes/microphone_node.py --device hw:0,0  # a specific ALSA device (arecord -L)

Publishes:
    /microphone/audio          std_msgs/Int16MultiArray  mono PCM, 16 kHz, 100 ms chunks
    /microphone/level_db       std_msgs/Float32          loudness of each chunk (dBFS)
    /microphone/speech_active  std_msgs/Bool             True while you are speaking
Speech detection adapts to the room: it is "speech" when the level is --margin-db above the
background noise measured over the last 10 s (or above a fixed --threshold-db if given).

A speech-to-text node is the planned consumer of /microphone/audio; it will publish the
recognised text as instructions on /instruction (std_msgs/String).
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int16MultiArray, MultiArrayDimension

FRAME_ID = "microphone_link"


class MicrophoneNode(Node):
    def __init__(
        self, device: str, rate: int, chunk_ms: int, threshold_db: float | None, margin_db: float, hangover_s: float
    ) -> None:
        super().__init__("microphone")
        self.rate = rate
        self.chunk_samples = rate * chunk_ms // 1000
        self.fixed_threshold = threshold_db
        self.margin_db = margin_db
        self.threshold_db = -40.0 if threshold_db is None else threshold_db
        self.hangover_s = hangover_s
        n_window = int(10_000 / chunk_ms)  # 10 s of chunk levels for the noise floor
        self.recent_levels: collections.deque[float] = collections.deque(maxlen=n_window)
        self.recent_clipping: collections.deque[bool] = collections.deque(maxlen=int(3_000 / chunk_ms))
        self._last_clip_warning = -1e9
        self.audio_pub = self.create_publisher(Int16MultiArray, "microphone/audio", 10)
        self.level_pub = self.create_publisher(Float32, "microphone/level_db", 10)
        self.speech_pub = self.create_publisher(Bool, "microphone/speech_active", 10)
        self.level_db = -120.0
        self.speech_active = False
        self._last_loud = 0.0
        self._stop = threading.Event()

        cmd = ["arecord", "-q", "-D", device, "-f", "S16_LE", "-c", "1", "-r", str(rate), "-t", "raw"]
        # own process group: Ctrl-C stops this node, which then stops arecord itself
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.create_timer(0.25, self._print_meter)
        mode = f"fixed {threshold_db} dBFS" if threshold_db is not None else f"background noise + {margin_db} dB"
        self.get_logger().info(
            f"recording '{device}' at {rate} Hz, {chunk_ms} ms chunks; frame {FRAME_ID}; speech threshold: {mode}"
        )

    def _read_loop(self) -> None:
        n_bytes = self.chunk_samples * 2
        while not self._stop.is_set():
            data = self.proc.stdout.read(n_bytes)
            if len(data) < n_bytes:
                if self._stop.is_set():  # normal shutdown via close()
                    break
                err = self.proc.stderr.read().decode(errors="replace").strip()
                self.get_logger().error(f"microphone stream ended: {err or 'no data from arecord'}")
                self._stop.set()
                break
            self._publish(np.frombuffer(data, dtype=np.int16))

    def _publish(self, samples: np.ndarray) -> None:
        msg = Int16MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="samples", size=samples.size, stride=samples.size)]
        msg.data = samples.tolist()
        self.audio_pub.publish(msg)

        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0
        self.level_db = float(20.0 * np.log10(max(rms, 1e-6)))
        self.level_pub.publish(Float32(data=self.level_db))

        now = time.monotonic()
        self.recent_levels.append(self.level_db)
        if self.fixed_threshold is None and len(self.recent_levels) >= 10:
            noise_floor = float(np.percentile(self.recent_levels, 20))
            self.threshold_db = max(noise_floor + self.margin_db, -60.0)
        self.recent_clipping.append(np.mean(np.abs(samples.astype(np.int32)) >= 32700) > 0.01)
        if (
            len(self.recent_clipping) == self.recent_clipping.maxlen
            and np.mean(self.recent_clipping) > 0.5
            and now - self._last_clip_warning > 30.0
        ):
            self._last_clip_warning = now
            self.get_logger().warning(
                "\nmicrophone input is overloaded (clipping): speech cannot be separated from noise. "
                "Lower the gain, e.g.:  amixer -c 1 sset 'Internal Mic Boost' 1 && amixer -c 1 sset Capture 50%"
            )
        if self.level_db > self.threshold_db:
            self._last_loud = now
        active = now - self._last_loud < self.hangover_s
        if active != self.speech_active:
            self.speech_active = active
            self.get_logger().info("speech started" if active else "speech ended")
        self.speech_pub.publish(Bool(data=active))

    def _print_meter(self) -> None:
        filled = int(np.clip((self.level_db + 80.0) / 80.0, 0.0, 1.0) * 16)
        state = "SPEAKING" if self.speech_active else "quiet   "
        # short enough not to wrap in a narrow terminal (a wrapped line breaks the \r redraw)
        sys.stdout.write(
            f"\r[mic] {self.level_db:6.1f} dB (speech>{self.threshold_db:5.1f}) {'#' * filled}{'.' * (16 - filled)} {state}"
        )
        sys.stdout.flush()
        if self._stop.is_set():
            raise SystemExit(1)

    def close(self) -> None:
        self._stop.set()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="default", help="ALSA capture device (list with: arecord -L)")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--threshold-db", type=float, default=None, help="fixed speech level (default: adaptive)")
    parser.add_argument("--margin-db", type=float, default=12.0, help="adaptive: dB above background noise")
    parser.add_argument("--hangover", type=float, default=0.6, help="seconds speech stays active after the level drops")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MicrophoneNode(args.device, args.rate, args.chunk_ms, args.threshold_db, args.margin_db, args.hangover)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        print()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
