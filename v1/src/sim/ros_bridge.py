"""ROS 2 publishers for the live simulation, built as one Isaac Sim OmniGraph.

Runs inside Isaac Sim. Topics are only published while the simulation is playing.
Topic names follow the realsense-ros layout so RealSense tooling works unchanged:

    /clock                          rosgraph_msgs/Clock      simulation time
    /camera/color/image_raw         sensor_msgs/Image        rgb8, 1280x720
    /camera/color/camera_info       sensor_msgs/CameraInfo
    /camera/depth/image_rect_raw    sensor_msgs/Image        32FC1, metres along the optical axis
    /camera/depth/camera_info       sensor_msgs/CameraInfo   (same intrinsics as colour)
    /tf                             tf2_msgs/TFMessage       world -> arm and gripper links (every step)
    /tf_static                      tf2_msgs/TFMessage       camera_link -> camera_color_optical_frame,
                                                             world -> microphone_link,
                                                             world -> table_camera_optical_frame
    /table_camera/color/..., /table_camera/depth/...         fixed table camera (same layout)

Import only after creating the SimulationApp and enabling isaacsim.ros2.bridge.
"""

from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import usdrt.Sdf
from pxr import UsdGeom

from sim.scene import CAM_RESOLUTION

GRAPH_PATH = "/World/ROS2Bridge"
CAMERA_FRAME = "camera_color_optical_frame"
TOPICS = {  # wrist camera (eye-in-hand); the table camera uses the same names under table_camera/
    "color": "camera/color/image_raw",
    "color_info": "camera/color/camera_info",
    "depth": "camera/depth/image_rect_raw",
    "depth_info": "camera/depth/camera_info",
}


WORLD_FRAME = "world"  # root frame of the dynamic TF tree (ROS2PublishTransformTree names it "world"); one tree for everything


def _static_transform(parent_frame: str, child_path: str, parent_path: str | None) -> tuple:
    """(parent, child, xyz, quat xyzw) of a fixed frame, read from the USD stage."""
    cache = UsdGeom.XformCache()
    stage = stage_utils.get_current_stage()
    child = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
    if parent_path is not None:
        child = child * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path)).GetInverse()
    t, q = child.ExtractTranslation(), child.ExtractRotationQuat()
    xyz = [float(t[0]), float(t[1]), float(t[2])]
    quat = [float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2]), float(q.GetReal())]
    return parent_frame, child_path.rsplit("/", 1)[-1], xyz, quat


def camera_topics(prefix: str) -> dict[str, str]:
    """realsense-ros style topic names under a namespace, e.g. 'camera' or 'table_camera'."""
    return {
        "color": f"{prefix}/color/image_raw",
        "color_info": f"{prefix}/color/camera_info",
        "depth": f"{prefix}/depth/image_rect_raw",
        "depth_info": f"{prefix}/depth/camera_info",
    }


def create_ros2_bridge(
    cameras: list[tuple[str, str, str]], tf_bodies: list[str], static_frames: list[tuple[str, str | None]]
) -> None:
    """Build the publisher graph.

    cameras: (USD camera prim, topic prefix, optical frame id) per camera, e.g.
             ("/World/.../d435_color", "camera", "camera_color_optical_frame").
             An optional 4th item False publishes colour only (no depth topics).
    tf_bodies: articulation roots / rigid bodies published on /tf every step, relative to World.
    static_frames: (frame prim, parent link prim or None for World) published once on /tf_static.
    """
    keys = og.Controller.Keys
    height, width = CAM_RESOLUTION
    create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ("PublishStaticTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
    ]
    values = [("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(p) for p in tf_bodies])]
    statics = [
        _static_transform(parent.rsplit("/", 1)[-1] if parent else WORLD_FRAME, frame, parent)
        for frame, parent in static_frames
    ]
    values += [
        ("PublishStaticTF.inputs:staticPublisher", True),
        ("PublishStaticTF.inputs:topicName", "tf_static"),
        ("PublishStaticTF.inputs:parentFrames", [s[0] for s in statics]),
        ("PublishStaticTF.inputs:childFrames", [s[1] for s in statics]),
        ("PublishStaticTF.inputs:translations", [s[2] for s in statics]),
        ("PublishStaticTF.inputs:orientations", [s[3] for s in statics]),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishStaticTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishStaticTF.inputs:timeStamp"),
    ]
    for k, (camera_path, prefix, frame, *rest) in enumerate(cameras):
        with_depth = rest[0] if rest else True
        topics = camera_topics(prefix)
        render = f"RenderProduct{k}"
        create.append((render, "isaacsim.core.nodes.IsaacCreateRenderProduct"))
        values += [
            (f"{render}.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
            (f"{render}.inputs:width", width),
            (f"{render}.inputs:height", height),
        ]
        connect.append(("OnPlaybackTick.outputs:tick", f"{render}.inputs:execIn"))
        for kind, node_type, image_type in (
            ("color", "isaacsim.ros2.bridge.ROS2CameraHelper", "rgb"),
            ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper", "depth"),
            ("color_info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper", None),
            ("depth_info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper", None),
        ):
            if kind.startswith("depth") and not with_depth:
                continue
            name = f"{kind.title().replace('_', '')}Publish{k}"
            create.append((name, node_type))
            values += [(f"{name}.inputs:topicName", topics[kind]), (f"{name}.inputs:frameId", frame)]
            if image_type:
                values.append((f"{name}.inputs:type", image_type))
            connect += [
                (f"{render}.outputs:execOut", f"{name}.inputs:execIn"),
                (f"{render}.outputs:renderProductPath", f"{name}.inputs:renderProductPath"),
            ]

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {keys.CREATE_NODES: create, keys.SET_VALUES: values, keys.CONNECT: connect},
    )
