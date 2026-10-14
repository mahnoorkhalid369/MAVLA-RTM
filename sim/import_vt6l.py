"""Convert the Epson VT6L URDF (assets/vt6l/vt6l.urdf, from the Epson manual) to USD for Isaac Sim 6.x.

Usage:
    python -m sim.import_vt6l
Output:
    assets/vt6l/usd/vt6l/vt6l.usda  (the importer makes a folder named after the URDF file)
"""

from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.urdf")
enable_extension("isaacsim.robot.schema")
simulation_app.update()

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "assets" / "vt6l" / "vt6l.urdf"
USD_DIR = ROOT / "assets" / "vt6l" / "usd"

config = URDFImporterConfig(
    urdf_path=str(URDF),
    usd_path=str(USD_DIR),
    merge_fixed_joints=False,  # keep tool0 as a separate frame
    fix_base=True,  # bolted to the floor
    collision_type="Convex Hull",
    allow_self_collision=False,
    joint_drive_type="force",
    joint_target_type="position",
    # Stiff PD position drives so the arm tracks commanded joint angles.
    # (The Robotiq 2F-85 gripper is a separate asset with its own drive; see sim/scene.py.)
    override_joint_stiffness=1.0e5,
    override_joint_damping=1.0e3,
    robot_type="Manipulator",
)
output_usd = URDFImporter(config).import_urdf()
print(f"[import_vt6l] wrote: {output_usd}")

simulation_app.close()
