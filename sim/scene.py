"""Mobile pick-place scene for the G1 in MuJoCo -- the digital twin the autonomy FSM is validated in.

The robot is the public Unitree **G1 + Dex3** model (``g1_29dof_with_hand_rev_1_0``) -- the same 29-DoF
body and 2x7-DoF dexterous hands the real G1-D carries, so the two-handed grasp is articulated rather
than implied. The bipedal base stands in for the wheeled one: the FSM drives a kinematic non-holonomic
base, so only ``G1_XML`` changes when the wheeled asset is available. All joint indices are resolved by
NAME (see ``sim/grasp_ik.py``), because this model interleaves each hand after its own arm.

Layout: a front table carrying the target object -- a plain upright cylinder (45 mm radius, 182 mm tall),
the neutral stand-in used throughout this repo -- 5 in from the near edge; a place table 2 m to the left
(rotated 90 deg) holding a 1 in-thick rectangular container with a 6 in hole on four 4 in standoffs; and
box obstacles (5x7x11 in) scattered through the corridor with RANDOMIZED placement. The scene exposes an
occupancy grid so ``nav_planner`` can A* a collision-free path to the drop stance.

Everything here is a MuJoCo primitive or procedurally generated, so the scene builds with no asset
downloads beyond the public G1 model itself.
"""
import os
import sys

import numpy as np
import mujoco
import pathlib
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nav"))
import nav_planner as N

IN = 0.0254
HERE = os.path.dirname(os.path.abspath(__file__))
# Unitree G1 with Dex3 hands -- "g1_29dof_with_hand_rev_1_0", the 29-DoF body + 2x7-DoF hand
# configuration this project runs on. Fetch once with scripts/fetch_g1_assets.py, or set G1_XML
# to your own copy (e.g. the wheeled G1-D base, which is not public).
G1_XML = os.environ.get("G1_XML", os.path.join(HERE, "..", "assets", "g1", "g1_with_hands.xml"))
CONTAINER_OBJ = os.path.join(HERE, "assets", "container.obj")

# Work-surface height. 0.90 m sits inside the measured bimanual reach envelope of BOTH
# supported robots: the legged G1 reaches 0.74-1.07 m, while the taller wheeled G1-D cannot
# bring its palms below ~0.92 m at all, so a standard 0.74 m bench is unreachable for it.
TOP = 0.90
STAND_H = 4 * IN
CYL_R, CYL_H = 0.045, 0.1822                     # target object: upright cylinder
CYL_STAND_QUAT = [1, 0, 0, 0]                    # a MuJoCo cylinder primitive is already +z
FRONT_TBL = (1.0, 0.0)
PLACE_TBL = (0.0, 2.0)
CYL_XY = (0.7 + 5 * IN, 0.0)                    # 5in from the front table's near edge
CONT_XY = (0.0, 1.7 + 7 * IN)                    # 7in from the place table's robot-facing edge
GOAL = (0.0, CONT_XY[1] - 0.55)                  # robot drop stance: centered on the drop, ~0.55 m out
START = (0.0, 0.0)
PKG = [11 * IN / 2, 7 * IN / 2, 5 * IN / 2]
ROBOT_RADIUS = 0.18


_ROBOT_XML = None      # resolved once by _init_robot()
_BASE_Z = None


def floating_base_xml(path):
    """Return an MJCF guaranteed to have a floating base, writing a sibling file if one is needed.

    The wheeled G1-D's URDF roots at ``AGV_link`` through a fixed joint, so the importer folds the
    chassis straight into the worldbody: the robot is bolted to the origin and cannot be driven
    anywhere. Wrap the worldbody's bodies AND its loose geoms in one free-jointed body -- the chassis
    and the outer lift column are worldbody geoms, and leaving them behind strands them at the origin
    while the rest of the robot drives off. No-op when the asset already has a free joint.

    The rewritten file is written beside the source so ``compiler meshdir`` still resolves.
    """
    path = pathlib.Path(path)
    tree = ET.parse(path)
    world = tree.getroot().find("worldbody")
    if world is None or tree.getroot().findall(".//freejoint") or \
            any(j.get("type") == "free" for j in tree.getroot().iter("joint")):
        return path
    movable = [c for c in list(world) if c.tag in ("body", "geom", "site")]
    if not movable:
        return path
    base = ET.Element("body", {"name": "floating_base"})
    ET.SubElement(base, "freejoint", {"name": "base_free"})
    # The chassis inertial was merged into the world, leaving the wrapper massless, which MuJoCo
    # rejects. This twin runs forward kinematics only, so a nominal value never reaches a dynamics
    # computation.
    ET.SubElement(base, "inertial", {"pos": "0 0 0.2", "mass": "40", "diaginertia": "2 2 2"})
    for c in movable:
        world.remove(c)
        base.append(c)
    world.append(base)
    out = path.with_name("_" + path.stem + "_floating.xml")
    tree.write(out)
    return out


def base_stand_height(xml_path):
    """Base height that puts the lowest geometry (wheels or feet) exactly on the floor.

    Compiled from the ROBOT ALONE -- measuring on the assembled scene does not work, because an
    infinite ground plane carries a huge AABB that swamps the minimum and the tables would clamp it
    to zero. Measured from each geom's AABB corners rotated into the world: ``geom_rbound`` is a
    bounding SPHERE and over-estimates by enough to sink a humanoid most of a metre into the floor,
    and the geom frame origin is ~17 cm off on a mesh like the AGV chassis. Starts from the asset's
    own ``qpos0`` so a model that already ships standing is left where its author put it.
    """
    model = mujoco.MjSpec.from_file(str(xml_path)).compile()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    lo = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        pos = data.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = centre + half * np.array([sx, sy, sz], float)
                    lo = min(lo, float((pos + rot @ corner)[2]))
    return float(model.qpos0[2] - lo)


def _init_robot():
    """Resolve the robot MJCF (adding a floating base if needed) and its stand height, once."""
    global _ROBOT_XML, _BASE_Z
    if _ROBOT_XML is None:
        _ROBOT_XML = floating_base_xml(G1_XML)
        _BASE_Z = base_stand_height(_ROBOT_XML)
    return _ROBOT_XML, _BASE_Z


def _make_container(path):
    def rect_pt(th, rx, ry):
        c, s = np.cos(th), np.sin(th)
        return min(rx / abs(c) if abs(c) > 1e-9 else 1e18, ry / abs(s) if abs(s) > 1e-9 else 1e18) * np.array([c, s])
    rx, ry, rh, t, M = 5 * IN, 4 * IN, 3 * IN, 1 * IN, 64
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    outer = [rect_pt(a, rx, ry) for a in ang]; inner = [(rh * np.cos(a), rh * np.sin(a)) for a in ang]
    V, F = [], []
    for x, y in outer: V.append((x, y, t / 2))
    for x, y in inner: V.append((x, y, t / 2))
    for x, y in outer: V.append((x, y, -t / 2))
    for x, y in inner: V.append((x, y, -t / 2))
    def quad(a, b, c, d): F.append((a, b, c)); F.append((a, c, d))
    for i in range(M):
        j = (i + 1) % M
        quad(i, j, M + j, M + i); quad(2 * M + j, 2 * M + i, 3 * M + i, 3 * M + j)
        quad(j, i, 2 * M + i, 2 * M + j); quad(M + i, M + j, 3 * M + j, 3 * M + i)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in V: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for a, b, c in F: f.write(f"f {a+1} {b+1} {c+1}\n")


if not os.path.exists(CONTAINER_OBJ):
    _make_container(CONTAINER_OBJ)


def occupancy(packages):
    g = N.OccupancyGrid(size_m=8.0, res=0.04)
    g.add_box(*FRONT_TBL, 0.6, 0.9)              # front table (rotated) footprint
    g.add_box(*PLACE_TBL, 0.9, 0.6)              # place table footprint
    for px, py, yaw in packages:
        c, s = abs(np.cos(np.radians(yaw))), abs(np.sin(np.radians(yaw)))
        g.add_box(px, py, 11 * IN * c + 7 * IN * s, 11 * IN * s + 7 * IN * c)
    return g


def plan_path(packages):
    N.ROBOT_RADIUS = ROBOT_RADIUS
    return N.plan(occupancy(packages), START, GOAL)


def random_packages(rng, n=3, tries=200):
    """Sample n package poses in the corridor that leave a collision-free A* path start->goal."""
    for _ in range(tries):
        pkgs = []
        for _ in range(n):
            pkgs.append((float(rng.uniform(-0.38, 0.38)), float(rng.uniform(0.45, 1.55)), float(rng.uniform(0, 90))))
        # keep them apart and clear of start/goal
        ok = all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.42 for i, p in enumerate(pkgs) for q in pkgs[i + 1:])
        ok = ok and all(np.hypot(p[0], p[1] - 0.0) > 0.4 and np.hypot(p[0], p[1] - GOAL[1]) > 0.4 for p in pkgs)
        if ok and plan_path(pkgs) is not None:
            return pkgs
    return [(0.32, 0.6, 20), (-0.34, 1.05, -30), (0.34, 1.55, 15)]   # fallback (known solvable)


def quat_z(deg):
    r = np.radians(deg) / 2.0
    return [np.cos(r), 0, 0, np.sin(r)]


def build(packages, path=None, intruder=None):
    """Build the MjModel. Returns (model, info). cylinder has a freejoint (kinematic control in the animation).
    `intruder` (x,y) adds a red obstacle that is NOT in the planned occupancy -- used to demo the failsafe."""
    robot_xml, base_z = _init_robot()
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[12, 12, 0.1], rgba=[0.3, 0.32, 0.35, 1])
    wb.add_light(pos=[0.5, 1, 4], dir=[-0.1, -0.1, -1], type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL))
    wb.add_light(pos=[0, 1, 4], dir=[0, 0, -1])
    spec.add_mesh(name="container", file=CONTAINER_OBJ)

    def box(name, pos, half, rgba, quat=(1, 0, 0, 0)):
        b = wb.add_body(name=name, pos=pos, quat=quat)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, rgba=rgba)

    q90 = quat_z(90)
    box("front_table", [*FRONT_TBL, TOP / 2], [0.45, 0.3, TOP / 2], [0.55, 0.4, 0.25, 1], quat=q90)
    box("place_table", [*PLACE_TBL, TOP / 2], [0.45, 0.3, TOP / 2], [0.55, 0.4, 0.25, 1])
    # container on standoffs
    for sx in (-5 * IN, 5 * IN):
        for sy in (-4 * IN, 4 * IN):
            b = wb.add_body(name=f"stand_{sx:.3f}_{sy:.3f}", pos=[CONT_XY[0] + sx, CONT_XY[1] + sy, TOP + STAND_H / 2])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.008, STAND_H / 2, 0], rgba=[0.3, 0.3, 0.32, 1])
    cb = wb.add_body(name="container", pos=[CONT_XY[0], CONT_XY[1], TOP + STAND_H + 0.5 * IN])
    cb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="container", rgba=[0.72, 0.72, 0.75, 1])
    # target object: a free body, kinematically posed by the animation while carried
    cylinder = wb.add_body(name="cylinder", pos=[*CYL_XY, TOP + CYL_H / 2], quat=CYL_STAND_QUAT)
    cylinder.add_freejoint()
    cylinder.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CYL_R, CYL_H / 2, 0],
                      rgba=[0.30, 0.55, 0.85, 1])
    # packages
    for i, (px, py, yaw) in enumerate(packages):
        b = wb.add_body(name=f"pkg{i}", pos=[px, py, PKG[2]], quat=quat_z(yaw))
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=PKG, rgba=[0.78, 0.62, 0.42, 1])
    # unexpected intruder (red) -- deliberately NOT added to occupancy() so the planner doesn't route around it;
    # the LiDAR failsafe must catch it at runtime.
    if intruder is not None:
        ib = wb.add_body(name="intruder", pos=[intruder[0], intruder[1], 0.15])
        ib.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.12, 0.12, 0.15], rgba=[0.9, 0.1, 0.1, 1])

    # optional planned-path polyline (blue capsules at 4in)
    if path:
        grid = occupancy(packages)
        pts = [grid.c2w(r, c) for r, c in path]
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            seg = wb.add_body(name=f"path{i}")
            seg.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x0, y0, 4 * IN, x1, y1, 4 * IN],
                         size=[0.015, 0, 0], rgba=[0.2, 0.5, 1.0, 1], contype=0, conaffinity=0)

    # The scene is lit from above, so at a low camera elevation the table sides and the
    # robot's own front face read almost black. Ambient fill fixes that without washing out
    # the directional shadows that make the depth legible.
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    model = spec.compile()
    gb = model.body("cylinder")
    info = {"cyl_qadr": int(model.jnt_qposadr[gb.jntadr[0]]), "packages": packages,
            "base_z": base_z}
    return model, info
