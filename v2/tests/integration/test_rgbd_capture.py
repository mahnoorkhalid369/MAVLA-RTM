"""Optional production ROS capture loopback. Uses isolated domain 87 and never publishes arm commands.

source /opt/ros/jazzy/setup.bash
ROS_LOG_DIR=/tmp/ros-v4-capture ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="src:$PYTHONPATH" .venv-seg/bin/python -m pytest tests/integration/test_rgbd_capture.py -q
"""

import os
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ROS_DOMAIN_ID') != '87', reason='requires isolated ROS domain 87')


def test_real_ros_callbacks_produce_a_synchronized_sensor_bundle():
    rclpy = pytest.importorskip('rclpy')
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from agent.ros_bridge import RosBridge
    from control.robot import ARM_HOME, ARM_JOINTS, GRIPPER_JOINT
    rclpy.init()
    publisher = rclpy.create_node('isolated_capture_test')
    pubs = {key: publisher.create_publisher(kind, topic, 10) for key, kind, topic in [
        ('rgb', Image, 'table_camera/color/image_raw'), ('depth', Image, 'table_camera/depth/image_rect_raw'),
        ('colour_info', CameraInfo, 'table_camera/color/camera_info'), ('depth_info', CameraInfo, 'table_camera/depth/camera_info'),
        ('pose', PoseStamped, 'table_camera/optical_pose'), ('joints', JointState, 'joint_states'), ('clock', Clock, 'clock')]}
    bridge, stop, errors = RosBridge(), threading.Event(), []
    def publish():
        try:
            i = 0
            while not stop.is_set():
                i += 1
                ts = i / 30
                clock = Clock()
                clock.clock.sec, clock.clock.nanosec = int(ts), int((ts - int(ts)) * 1e9)
                pubs['clock'].publish(clock)
                for key in ('rgb', 'depth', 'colour_info', 'depth_info', 'pose', 'joints'):
                    if key in ('rgb', 'depth'):
                        m = Image()
                        m.height, m.width = 24, 32
                        m.encoding = 'rgb8' if key == 'rgb' else '32FC1'
                        a = np.full((24, 32, 3), 127, np.uint8) if key == 'rgb' else np.full((24, 32), 1.2, np.float32)
                        m.step, m.data = a.strides[0], a.tobytes()
                    elif key.endswith('info'):
                        m = CameraInfo()
                        m.height, m.width = 24, 32
                        m.k = [100., 0., 16., 0., 100., 12., 0., 0., 1.]
                    elif key == 'pose':
                        m = PoseStamped()
                        m.pose.orientation.w = 1.
                        m.pose.position.z = 1.5
                    else:
                        m = JointState()
                        m.name, m.position = ARM_JOINTS + [GRIPPER_JOINT], list(map(float, ARM_HOME)) + [0.]
                    m.header.stamp = clock.clock
                    m.header.frame_id = 'world' if key == 'pose' else 'table_camera_optical_frame'
                    pubs[key].publish(m)
                stop.wait(.033)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        assert bridge.ready(timeout=8), errors
        frame = bridge.snapshot(timeout=8)
        assert frame.rgb.shape == (24, 32, 3) and frame.depth.shape == (24, 32)
        assert abs(frame.stamp - frame.depth_stamp) <= .003
        assert np.allclose(frame.q, ARM_HOME) and frame.T_world_camera[2, 3] == pytest.approx(1.5)
        assert not errors
    finally:
        stop.set()
        thread.join(timeout=2)
        bridge.close()
        publisher.destroy_node()
        rclpy.shutdown()
