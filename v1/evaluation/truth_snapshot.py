"""Save one /sim/ground_truth/objects message (EVALUATION ONLY; never imported by agent code).

    .venv-seg/bin/python -m evaluation.truth_snapshot out.json
"""

import json
import sys
import time


def main(path: str) -> int:
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node("truth_snapshot")
    got = {}
    node.create_subscription(String, "sim/ground_truth/objects", lambda m: got.setdefault("m", json.loads(m.data)), 10)
    end = time.time() + 60
    while "m" not in got and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "m" in got:
        open(path, "w").write(json.dumps(got["m"], indent=1))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if "m" in got else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
