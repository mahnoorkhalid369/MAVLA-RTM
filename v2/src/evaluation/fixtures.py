"""Evaluation-only initial scene variations; no agent observations or decisions are modified."""


def apply_setup(scenario, setup):
    items = {o.id:o for o in scenario.objects}
    regions = {r.id:r for r in scenario.regions}
    for key, dest in setup.get("preplace", {}).items():
        o, region = items[key], regions[dest]
        o.pos = (*region.centre[:2], region.centre[2]+.002)
        o.support = dest
    for key, degrees in setup.get("rotate_yaw_deg", {}).items():
        items[key].yaw_deg = (items[key].yaw_deg+float(degrees)) % 360
    if setup.get("preplace") and "serving_tray" in setup["preplace"].values():
        scenario.notes["receiving_tray_initially_empty"] = False
    return scenario
