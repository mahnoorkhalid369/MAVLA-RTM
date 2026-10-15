"""EVALUATION ONLY (never imported by agent code): read the robot and simulator state over ROS 2.

    .venv-seg/bin/python -m evaluation.lab_probe snapshot out.json     # one /joint_states + one ground-truth message
    .venv-seg/bin/python -m evaluation.lab_probe record out.jsonl      # stream both as JSON lines until SIGINT/SIGTERM
"""

import json
import signal
import sys
import time


def main(mode: str, path: str) -> int:
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node("lab_probe")
    latest, stop = {}, {"now": False}
    f = open(path, "w") if mode == "record" else None

    def on_joints(m):
        rec = {"kind": "joints", "wall": time.time(), "name": list(m.name), "position": [float(v) for v in m.position]}
        latest["joints"] = rec
        if f:
            f.write(json.dumps(rec) + "\n")

    def on_truth(m):
        rec = {"kind": "truth", "wall": time.time(), **json.loads(m.data)}
        latest["truth"] = rec
        if f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    node.create_subscription(JointState, "joint_states", on_joints, 50)
    node.create_subscription(String, "sim/ground_truth/objects", on_truth, 10)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.update(now=True))
    end = time.time() + 60
    while not stop["now"]:
        rclpy.spin_once(node, timeout_sec=0.1)
        if mode == "snapshot" and ("joints" in latest and "truth" in latest or time.time() > end):
            break
    if mode == "snapshot":
        open(path, "w").write(json.dumps(latest, indent=1))
    if f:
        f.close()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if mode == "record" or len(latest) == 2 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
