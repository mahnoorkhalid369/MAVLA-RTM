# Home kitchen: personalised snack-tray preparation

Scenario name: **`snack_kitchen`**. The meal-box preparation surface is a **kitchen countertop**; this scene uses a freestanding **kitchen island**.

The VT6L stands on a 0.90 m high island with a light peach-cream matte countertop, navy cabinetry, and brass-toned handles. The rear worktop is muted grey-green, providing contrast with the white appliances. Snacks are arranged on the robot's right (the table camera's left); a teal serving tray and a deeper storage basket are on the other side. A refrigerator, microwave, cutting board, white dish tray, cabinetry, tiled backsplash, and window establish the household setting. Background furniture/appliances remain outside the robot's reach. The work surface stays at robot-frame z=0, preserving the planner's coordinates.

![Software-rendered geometry preview](../out/scenarios/snack_kitchen/kitchen_overview.png)

The preview is rendered from the authored USD geometry with software OpenGL. Isaac RTX lighting/materials may differ. The standalone export includes a static HOME-pose robot visual; the live launcher creates the controlled articulation, gripper, RGB-D cameras, and ROS interface.

## Run

From the repository root:

```bash
cd v2
./launch/sim.sh --scenario snack_kitchen --difficulty easy --seed 42
```

This starts the scene without the agent commanding motion. Stop it before starting the full agent:

```bash
./launch/agent.sh --scenario snack_kitchen --difficulty easy --seed 42
```

`src/agent/settings.py` now defaults to `snack_kitchen`, `easy`, seed `42`. Restart the agent and simulator to rebuild the scene; an already running scene keeps its old objects. CLI overrides apply to one run. The kitchen uses the same configured camera mount and empty calibrated-region layout as `workcell`.

Example requests:

> Put the milk and the crackers in the blue serving tray.

> Put the apple and the water bottle in the blue serving tray. Leave the pudding on the counter.

> Prepare the blue serving tray with water, crackers, and a fruit cup.

## Items and difficulties

| Difficulty | Setup |
|---|---|
| Easy (default) | Six items: milk, water bottle, crackers, fruit cup, apple and pudding; 100 mm configured counter clearance. |
| Medium | Adds one soup can; 90 mm counter clearance. |
| Hard | Adds a mustard bottle and varied orientations; eight counter items, 80 mm counter clearance. |

The milk is a small white rectangular pack with blue bands and large `MILK` printing on its top and all four sides. Water is a blue bottle with a broad body, narrower shoulder/neck, blue cap and `WATER` labels. The fruit cup has a `FRUIT` lid. The unlabelled orange drink bottle, look-alike dessert/sugar packages, oversized tuna can and preloaded mug are removed from these layouts. Both receiving containers start empty.

Clearance is applied to collision geometry during placement. Tall/short pairs also receive extra separation for the wrist-camera housing. These spacing rules do not establish grasp success: the offline full-gripper approach/descent/lift checks below and live RGB-D validation remain separate. Difficulty adds distinct items rather than intentionally confusing their identity.

The seed changes object positions. The kitchen owns its item selection and layout generation independently of `workcell`; it shares the project's scenario primitives, camera convention, and YCB source library. Object prim names receive alphabetic prefixes where needed because USD identifiers cannot start with the numeric YCB catalog codes.

The serving tray and storage basket are collision-backed fixtures, perceived through RGB-D. No mat cells or privileged `R<n>` placement IDs are exported to the agent. Names, source data, and sample requests in scenario metadata are for evaluation only.

## NVIDIA USD sources

NVIDIA's official [Residential Assets Pack](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#residential-assets-pack) supplies:

| Model | Archive path under `Assets/ArchVis/Residential/` | Use |
|---|---|---|
| Refrigerator | `Appliances/Refrigerators/fridge.usd` | Background appliance |
| Microwave | `Appliances/Microwaves/microwave.usd` | Background appliance |
| Cutting board | `Kitchen/Kitchenware/CuttingBoards/cutting_board_A.usd` | Back-counter prop |
| White tray | `Kitchen/Kitchenware/Serving/white_tray.usd` | Back-counter dish prop |
| Apple | `Food/Fruit/Apple.usd` | Manipulable fruit, uniformly scaled to 75% |

The package models use centimetres. Small wrappers in `assets/snack_kitchen/models/` convert them to metres and centre their bounds. Original geometry and materials remain referenced. The apple is about 67 mm across, inside the gripper width limit; it uses a rigid cylindrical collision proxy rather than fruit deformation.

The other scanned food packages come from the locally installed NVIDIA Isaac Sim YCB assets, also used in NVIDIA's [official synthetic-data example](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html). The milk pack, labelled water bottle and fruit cup are procedural stand-ins. The island, cabinetry, serving/storage containers, and room are procedural geometry; they are not presented as downloaded NVIDIA furniture.

The Residential archive is approximately 22.9 GB. `scripts/fetch_snack_assets.py` fetches only this subset using HTTP ranges: **70 files, about 122 MB compressed / 145 MB extracted**, including all referenced texture images and the supplied licence notice. Runtime MDL materials `OmniPBR.mdl` and `OmniGlass.mdl` are supplied by Isaac Sim.

```bash
# From v2; needed only when the local subset is absent.
.venv-seg/bin/python scripts/fetch_snack_assets.py
.venv-seg/bin/python scripts/fetch_snack_assets.py --check
```

Downloaded assets stay in the shared ignored directory `data/assets/nvidia_residential/`. The checked-in manifest `assets/snack_kitchen/nvidia_manifest.json` records their source, sizes, and SHA256 checksums. The Residential content is governed by NVIDIA's bundled terms, not the YCB licence; the original notice is preserved in `assets/snack_kitchen/NVIDIA-LICENSE.txt` and the download directory.

## Export and validation

The standalone export works with the existing USD-capable `.venv` without starting Isaac Sim:

```bash
.venv/bin/python scripts/export_snack_kitchen.py --difficulty easy --seed 42
```

It writes `out/scenarios/snack_kitchen/snack_kitchen_easy_s42.usda`, ground-truth metadata, and a static geometry report. USD references are relative to the export; keep the repository asset/data directories available. This is an environment export with standard USD prop physics and a static robot preview, not a replacement for the live ROS-controlled simulator.

Offline planner validation:

```bash
PYTHONPATH=src .venv-seg/bin/python -m scenarios.checks \
  --scenario snack_kitchen --difficulty easy --seed 42 \
  --out out/scenarios/snack_kitchen/easy_s42_design.json
```

Checks cover spawn overlap, background geometry outside reach, camera top visibility, planned grasps, and placement probes. The generic scenario tests cover easy/medium/hard at seeds 0, 42, and 1234, plus seeded reproducibility. Dedicated tests check valid prim names, asset paths, preserved workcell state, wrapper scale/centring, and exported rigid-body/support physics.

Validation record (2026-11-11): **53 targeted tests passed**: seven kitchen tests, seven registry tests, 24 master/worker tests, and 15 scenario-library cases selected for `snack_kitchen`. All nine difficulty/seed combinations passed static geometry/visibility checks. Full offline grasp and placement checks reported zero problems for easy, medium, and hard at seed 42. All 70 downloaded NVIDIA files passed size/SHA256 verification.

A broader scenario-suite attempt encountered hard-layout generation failures in the separately changing `workcell` module and was interrupted. The kitchen now has its own layout generator; full-project suite success is not claimed here.

Live PhysX settling, RGB-D segmentation, semantic grounding, and end-to-end grasp success remain to be tested on the GPU host. The GPU driver was unavailable in the authoring session; a software-rendered preview and offline planning results do not establish those properties.

Scene revision 1.1 validation (2026-11-12): 20 focused kitchen/scenario tests passed; two USD-dependent tests were skipped in the segmentation Python environment. Separate checks in the USD Python environment verified the exported support collider, all six rigid-body masses and product-label textures. All three difficulty layouts at seed 42 passed offline grasp and placement-probe planning with zero reported problems. Static layout checks covered seeds 0, 42 and 1234 at each difficulty. The easy seed-42 scene has a conservative item-to-item footprint clearance of at least 101.9 mm. Both software previews were regenerated and inspected. No live camera/model inference or physical picks were run for this revision.
