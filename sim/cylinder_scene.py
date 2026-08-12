"""Minimal Isaac Sim scene: Unitree G1 humanoid + a table + a cylinder, for rendering demo videos.

This is a reference scene (not a full training env). It spawns the public G1 articulation, a plain table,
and a cylinder of roughly the manipulated-object dimensions, sets a camera, and steps the sim. Drive the
arms either by replaying a recorded joint trajectory or by stepping the deployed policy, and capture frames
to build the clips in ../docs/videos/.

Requires Isaac Sim / Isaac Lab. Paths to the G1 USD depend on your Isaac assets install.

  # inside an Isaac Sim python environment
  python sim/cylinder_scene.py --g1-usd /path/to/g1.usd --out ../docs/videos/demo.mp4
"""
import argparse

# Cylinder ~ the manipulated object footprint (metres). Tune to your object.
CYL_RADIUS = 0.045
CYL_HEIGHT = 0.20
TABLE_SIZE = (0.8, 0.6, 0.74)   # x, y, height


def build(args):
    # Imports are inside the function so the file is readable without Isaac installed.
    from isaacsim.simulation_app import SimulationApp  # noqa: F401  (Isaac Sim entrypoint)
    app = SimulationApp({"headless": args.headless})

    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCylinder, FixedCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # table
    world.scene.add(FixedCuboid(prim_path="/World/table", name="table",
                                position=np.array([0.5, 0.0, TABLE_SIZE[2] / 2]),
                                scale=np.array(TABLE_SIZE), color=np.array([0.8, 0.8, 0.8])))
    # cylinder (the manipulated object)
    world.scene.add(DynamicCylinder(prim_path="/World/cylinder", name="cylinder",
                                    position=np.array([0.5, 0.0, TABLE_SIZE[2] + CYL_HEIGHT / 2]),
                                    radius=CYL_RADIUS, height=CYL_HEIGHT, color=np.array([0.2, 0.5, 0.9])))
    # humanoid
    add_reference_to_stage(usd_path=args.g1_usd, prim_path="/World/G1")

    world.reset()
    # ---- drive the arms here: replay a trajectory or step the deployed policy, capturing frames ----
    for _ in range(args.steps):
        world.step(render=not args.headless)
    app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--g1-usd", required=True, help="path to the Unitree G1 USD in your Isaac assets")
    p.add_argument("--out", default="demo.mp4")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--headless", action="store_true")
    build(p.parse_args())
