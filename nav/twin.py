"""Live Rerun digital twin for the pick-place FSM. Mirrors a MuJoCo (model, data) each control tick: robot +
scene geoms (posed via the current qpos), the planned path, and the live LiDAR obstacle points. Works for the
sim backend (pass its model/data) and, on the robot, a G1 model whose qpos is set from lowstate/odometry.

    Twin(model, data, save_path="run.rrd")   # then twin.log_path(pts); twin.update(lidar_xy) each tick
    ... open with:  rerun run.rrd     (or spawn=True for a live window)
"""
import numpy as np
import mujoco
import rerun as rr


class Twin:
    def __init__(self, model, data, save_path=None, spawn=False):
        self.m, self.d = model, data
        rr.init("g1_digital_twin", spawn=spawn)
        if save_path:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.mesh_geoms, self.box_geoms = [], []
        for gi in range(model.ngeom):
            if model.geom_group[gi] == 3:                       # skip collision geoms
                continue
            if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gi]
                if mid < 0:
                    continue
                va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                V = model.mesh_vert[va:va + nv].reshape(-1, 3)
                Fc = model.mesh_face[fa:fa + nf].reshape(-1, 3)
                ent = f"world/geom_{gi}"
                rr.log(ent, rr.Mesh3D(vertex_positions=V, triangle_indices=Fc,
                                      albedo_factor=model.geom_rgba[gi][:3].tolist()), static=True)
                self.mesh_geoms.append((gi, ent))
            elif model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
                self.box_geoms.append(gi)
        self.t = 0

    def log_path(self, pts, z=0.1):
        rr.log("world/plan/path", rr.LineStrips3D([[[x, y, z] for x, y in pts]], colors=[40, 120, 255], radii=0.02),
               static=True)

    def update(self, lidar_xy=None):
        """Caller has already set self.d.qpos; log all geom transforms + LiDAR at this tick."""
        mujoco.mj_forward(self.m, self.d)
        rr.set_time("tick", sequence=self.t); self.t += 1
        for gi, ent in self.mesh_geoms:
            rr.log(ent, rr.Transform3D(translation=self.d.geom_xpos[gi], mat3x3=self.d.geom_xmat[gi].reshape(3, 3)))
        for gi in self.box_geoms:
            ent = f"world/box_{gi}"
            rr.log(ent, rr.Boxes3D(half_sizes=[self.m.geom_size[gi][:3].tolist()],
                                   colors=[(self.m.geom_rgba[gi][:3] * 255).astype(int).tolist()]))
            rr.log(ent, rr.Transform3D(translation=self.d.geom_xpos[gi], mat3x3=self.d.geom_xmat[gi].reshape(3, 3)))
        if lidar_xy is not None and len(lidar_xy):
            pts = np.column_stack([np.asarray(lidar_xy), np.full(len(lidar_xy), 0.12)])
            rr.log("world/lidar", rr.Points3D(pts, colors=[255, 80, 80], radii=0.03))
