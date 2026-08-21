"""3D digital twin of the G1-D in Rerun: the robot (meshes posed by forward-kinematics from joint states),
the tables, LiDAR obstacle boxes, and the planned path as a blue 3D line.

This VISUALIZES the real robot's state -- it does not simulate physics. Feed it live data:
  * base pose + joint angles  (from the robot's lowstate/odometry)  -> the arms/body move in 3D
  * obstacle boxes            (from clustering the LiDAR point cloud) -> "what it sees in the way"
  * planned path              (from nav_planner.plan)                -> blue 3D line it wants to follow
Here it's driven by a demo pose + the nav_planner demo scene so the twin is testable offline.

    MUJOCO_GL=glfw python twin_rerun.py --out /tmp/twin.rrd      # then: rerun /tmp/twin.rrd
"""
import argparse
import os
import sys

import mujoco
import rerun as rr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_planner as N

G1_XML = "$REPO/assets/robots/unitree_g1_mjlab/g1.xml"


def log_robot(m, d, root="world/g1"):
    """Log every visual mesh geom, baked into world coordinates at the current FK pose."""
    for gi in range(m.ngeom):
        if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[gi] != 2:
            continue
        mid = m.geom_dataid[gi]
        if mid < 0:
            continue
        va, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[va:va + nv].reshape(-1, 3)
        faces = m.mesh_face[fa:fa + nf].reshape(-1, 3)
        R = d.geom_xmat[gi].reshape(3, 3); p = d.geom_xpos[gi]
        world_v = verts @ R.T + p
        rr.log(f"{root}/geom_{gi}", rr.Mesh3D(vertex_positions=world_v, triangle_indices=faces,
                                              albedo_factor=[0.75, 0.78, 0.82]))


def log_table(name, cx, cy, w, h, top_z=0.74):
    """A table as a box centered at (cx,cy) with top at top_z."""
    rr.log(f"world/tables/{name}", rr.Boxes3D(centers=[[cx, cy, top_z / 2]],
                                              half_sizes=[[w / 2, h / 2, top_z / 2]], colors=[140, 110, 80]))


def log_obstacle_boxes(boxes, z0=0.0, z1=0.8):
    """boxes: list of (cx, cy, w, h) from LiDAR clustering -> red 3D boxes."""
    if not boxes:
        return
    centers = [[cx, cy, (z0 + z1) / 2] for cx, cy, _, _ in boxes]
    halves = [[w / 2, h / 2, (z1 - z0) / 2] for _, _, w, h in boxes]
    rr.log("world/obstacles", rr.Boxes3D(centers=centers, half_sizes=halves, colors=[220, 60, 60]))


def log_path(path_cells, grid, z=0.04):
    """A* cells -> a blue 3D polyline the robot wants to follow."""
    if not path_cells:
        return
    pts = [[*grid.c2w(r, c), z] for r, c in path_cells]
    rr.log("world/plan/path", rr.LineStrips3D([pts], colors=[40, 120, 255], radii=0.02))
    rr.log("world/plan/waypoints", rr.Points3D(pts, colors=[40, 120, 255], radii=0.03))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/twin.rrd")
    args = ap.parse_args()

    rr.init("g1_digital_twin")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- robot model + a demo standing pose (base pose + joints come from the real robot in live use) ---
    m = mujoco.MjModel.from_xml_path(G1_XML)
    d = mujoco.MjData(m)
    d.qpos[:3] = [2.0, -0.2, 0.793]      # base at the pick station (x,y,z)
    d.qpos[3:7] = [1, 0, 0, 0]           # base orientation (wxyz)
    # nudge the arms out so the twin clearly shows arm state (indices depend on model; safe no-op if absent)
    mujoco.mj_forward(m, d)
    log_robot(m, d)

    # --- scene: same layout as the nav_planner demo (tables + obstacle), and the planned path ---
    g = N.OccupancyGrid(size_m=8.0, res=0.05)
    tables = [("pick", 2.0, 0.9, 0.8, 0.8), ("place", -2.0, 0.9, 0.8, 0.8)]
    obstacles = [(0.0, 0.0, 0.6, 1.2)]
    for nm, cx, cy, w, h in tables:
        g.add_box(cx, cy, w, h); log_table(nm, cx, cy, w, h)
    for cx, cy, w, h in obstacles:
        g.add_box(cx, cy, w, h)
    log_obstacle_boxes(obstacles)

    robot_xy, goal_xy = (2.0, -0.2), (-2.0, -0.2)
    path = N.plan(g, robot_xy, goal_xy)
    log_path(path, g)
    rr.log("world/plan/goal", rr.Points3D([[*goal_xy, 0.04]], colors=[255, 180, 0], radii=0.06))

    rr.save(args.out)
    print(f"[twin] path waypoints: {None if path is None else len(path)}")
    print(f"[twin] saved {args.out}  ->  open with:  rerun {args.out}")
