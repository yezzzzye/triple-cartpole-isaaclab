# SPDX-License-Identifier: MIT
"""Convert the authored ``triple_cartpole.usd`` into an Isaac Lab ready articulation.

The authored asset is a *scene*, not a *robot*, so it cannot be loaded as an articulation
as-is. This script rewires the physics while leaving all authored geometry, materials,
masses and joint frames untouched. README.md ("How the asset is wired") explains each
change and why it is needed.

Usage::

    ./run.ps1 scripts/prepare_asset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", type=Path, default=REPO_ROOT / "assets" / "triple_cartpole.usd")
parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets" / "triple_cartpole_physics.usd")
parser.add_argument("--root-name", type=str, default="TripleCartpole", help="Name of the generated default prim.")
parser.add_argument("--slider-effort", type=float, default=400.0, help="maxForce written on the slider drive [N].")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

SOURCE_ROOT = "/World"
DROPPED_PRIMS = ("/GroundPlane", "/PhysicsScene", "/Render", "/Environment", "/world_anchor")
RAIL_LINK = "rail"
SLIDER_JOINT = "slider_to_cart"


def remap_relationships(stage: Usd.Stage, old_prefix: str, new_prefix: str) -> int:
    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(new_prefix)):
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            remapped = [
                Sdf.Path(new_prefix + str(t)[len(old_prefix) :]) if str(t).startswith(old_prefix + "/") else t
                for t in targets
            ]
            if remapped != targets:
                rel.SetTargets(remapped)
                count += 1
    return count


def main() -> None:
    if not args_cli.input.is_file():
        raise FileNotFoundError(f"Input asset not found: {args_cli.input}")

    root_path = f"/{args_cli.root_name}"
    print(f"[prepare_asset] reading  {args_cli.input}")

    source_stage = Usd.Stage.Open(str(args_cli.input))
    flattened = source_stage.Flatten()

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.unlink(missing_ok=True)
    out_layer = Sdf.Layer.CreateNew(str(args_cli.output))
    if not Sdf.CopySpec(flattened, Sdf.Path(SOURCE_ROOT), out_layer, Sdf.Path(root_path)):
        raise RuntimeError(f"Failed to copy {SOURCE_ROOT} into {root_path}")

    stage = Usd.Stage.Open(out_layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.GetPrimAtPath(root_path)
    stage.SetDefaultPrim(root_prim)
    for dropped in DROPPED_PRIMS:
        if stage.GetPrimAtPath(dropped):
            stage.RemovePrim(dropped)

    print(f"[prepare_asset] remapped {remap_relationships(stage, SOURCE_ROOT, root_path)} relationship(s)")

    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim or not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        print(f"[prepare_asset] removing ArticulationRootAPI from {prim.GetPath()}")
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        for name in ("physxArticulation:enabledSelfCollisions", "physxArticulation:articulationEnabled"):
            prim.RemoveProperty(name)

    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physx_articulation.CreateEnabledSelfCollisionsAttr(False)
    physx_articulation.CreateSolverPositionIterationCountAttr(16)
    physx_articulation.CreateSolverVelocityIterationCountAttr(1)

    rail_path = f"{root_path}/{RAIL_LINK}"
    rail_prim = UsdGeom.Xform.Define(stage, rail_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(rail_prim)
    rail_mass = UsdPhysics.MassAPI.Apply(rail_prim)
    rail_mass.CreateMassAttr(1.0)
    rail_mass.CreateDiagonalInertiaAttr((0.01, 0.01, 0.01))
    rail_mass.CreateCenterOfMassAttr((0.0, 0.0, 0.0))

    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/root_joint")
    root_joint.CreateBody1Rel().SetTargets([Sdf.Path(rail_path)])
    root_joint.CreateExcludeFromArticulationAttr(False)

    slider = None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == SLIDER_JOINT:
            slider = prim
            break
    if slider is None:
        raise RuntimeError(f"Joint '{SLIDER_JOINT}' not found under {root_path}")

    UsdPhysics.Joint(slider).CreateBody0Rel().SetTargets([Sdf.Path(rail_path)])
    drive = UsdPhysics.DriveAPI.Apply(slider, "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    drive.CreateMaxForceAttr(args_cli.slider_effort)
    drive.CreateTargetPositionAttr(0.0)
    print(f"[prepare_asset] anchored {SLIDER_JOINT} to {rail_path} with a force drive")

    stage.GetRootLayer().Save()
    print(f"[prepare_asset] wrote    {args_cli.output}")

    print("\n[prepare_asset] articulation summary")
    print(f"  default prim      : {stage.GetDefaultPrim().GetPath()}")
    links, joints = [], []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(prim.GetName())
        joint = UsdPhysics.Joint(prim)
        if joint and prim.GetTypeName() != "PhysicsFixedJoint":
            axis = prim.GetAttribute("physics:axis").Get()
            body0 = joint.GetBody0Rel().GetTargets()
            joints.append(f"{prim.GetName()} ({prim.GetTypeName()}, axis {axis}, parent {body0[0].name})")
    print(f"  links   ({len(links)}) : {', '.join(links)}")
    print(f"  dof     ({len(joints)}) :")
    for entry in joints:
        print(f"      - {entry}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
