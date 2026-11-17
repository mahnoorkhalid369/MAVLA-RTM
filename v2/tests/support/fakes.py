"""Shared test helpers: measured geometry as perception reports it (no camera, no simulator)."""


def geom(x, y, top, fp=(0.04, 0.04), support=0.0, yaw=0.0):
    h = top - support
    return {"centre": [x, y, support + h / 2], "top_z": top, "height": h, "support_z": support, "yaw_deg": yaw,
            "extents": list(fp), "footprint": [max(fp), min(fp)], "grasp_yaw_deg": yaw, "grasp_width": min(fp),
            "symmetry_deg": 90.0 if abs(fp[0] - fp[1]) < 0.008 else 180.0, "tilt_deg": 0.5}
