"""2D navigation core for the G1-D mobile pick-place: occupancy grid + A* + top-down digital-twin view with a
human ACCEPT gate. This is the backbone of the supervised-autonomy loop:

    LiDAR points -> occupancy grid (inflated by robot radius) -> A* path to the goal -> render a top-down
    "digital twin" (robot, obstacles, goal, planned path) -> user reviews and presses ACCEPT -> execute.

Frame: metric world (meters), x forward, y left, yaw about z. On the real robot the occupancy comes from the
LiDAR HeightMap/PointCloud2 and the robot pose from odometry; here the same API is driven by synthetic
obstacles so the planner + view + accept gate are fully testable offline.

Offline self-test (renders a plan to a PNG, no robot):
    python nav_planner.py --demo --out /tmp/plan.png
"""
import argparse
import heapq

import cv2
import numpy as np


class OccupancyGrid:
    def __init__(self, size_m=8.0, res=0.05, origin=(-4.0, -4.0)):
        self.res = res
        self.origin = np.array(origin, float)          # world coord of cell (0,0)
        self.n = int(round(size_m / res))
        self.occ = np.zeros((self.n, self.n), np.uint8)  # [row=y, col=x]; 1 = blocked

    def w2c(self, x, y):
        c = int((x - self.origin[0]) / self.res)
        r = int((y - self.origin[1]) / self.res)
        return r, c

    def c2w(self, r, c):
        return self.origin[0] + (c + 0.5) * self.res, self.origin[1] + (r + 0.5) * self.res

    def in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def add_box(self, cx, cy, w, h):
        r0, c0 = self.w2c(cx - w / 2, cy - h / 2)
        r1, c1 = self.w2c(cx + w / 2, cy + h / 2)
        self.occ[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = 1

    def add_points(self, pts_xy, z=None, z_min=0.05, z_max=1.6):
        """Mark cells hit by LiDAR points (pts_xy: Nx2 in world). Optional z filter drops floor/ceiling."""
        p = np.asarray(pts_xy, float)
        if z is not None:
            p = p[(np.asarray(z) > z_min) & (np.asarray(z) < z_max)]
        for x, y in p:
            r, c = self.w2c(x, y)
            if self.in_bounds(r, c):
                self.occ[r, c] = 1

    def inflate(self, radius_m):
        k = int(round(radius_m / self.res))
        if k <= 0:
            return self.occ.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        return cv2.dilate(self.occ, kernel)


def astar(grid_occ, start_rc, goal_rc):
    """8-connected A* on a binary occupancy array (1=blocked). Returns list of (r,c) cells or None."""
    n = grid_occ.shape[0]
    (sr, sc), (gr, gc) = start_rc, goal_rc
    if not (0 <= gr < n and 0 <= gc < n) or grid_occ[gr, gc]:
        return None
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    h = lambda r, c: np.hypot(r - gr, c - gc)
    openq = [(h(sr, sc), 0.0, (sr, sc))]
    came, gcost = {}, {(sr, sc): 0.0}
    while openq:
        _, g, (r, c) = heapq.heappop(openq)
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]
                path.append((r, c))
            return path[::-1]
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n) or grid_occ[nr, nc]:
                continue
            if dr and dc and (grid_occ[r, nc] or grid_occ[nr, c]):   # no corner cutting
                continue
            ng = g + np.hypot(dr, dc)
            if ng < gcost.get((nr, nc), 1e18):
                gcost[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(openq, (ng + h(nr, nc), ng, (nr, nc)))
    return None


def simplify(grid_occ, path):
    """Line-of-sight shortcut: drop intermediate cells you can reach in a straight, collision-free line."""
    if not path or len(path) < 3:
        return path
    def clear(a, b):
        for t in np.linspace(0, 1, int(np.hypot(b[0] - a[0], b[1] - a[1])) + 1):
            r = int(round(a[0] + t * (b[0] - a[0]))); c = int(round(a[1] + t * (b[1] - a[1])))
            if grid_occ[r, c]:
                return False
        return True
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not clear(path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return out


def render_topdown(grid, robot_xy, robot_yaw, goal_xy, path_cells, accepted=None, scale=6):
    """Top-down digital-twin image: obstacles (gray), inflated margin (light), robot+heading, goal, path."""
    inflated = grid.inflate(ROBOT_RADIUS)
    img = np.full((grid.n, grid.n, 3), 40, np.uint8)
    img[inflated > 0] = (70, 70, 90)          # inflated margin
    img[grid.occ > 0] = (120, 120, 140)       # hard obstacle
    img = cv2.resize(img, (grid.n * scale, grid.n * scale), interpolation=cv2.INTER_NEAREST)
    img = cv2.flip(img, 0)                     # y up

    def px(x, y):
        r, c = grid.w2c(x, y)
        return int(c * scale + scale / 2), int((grid.n - 1 - r) * scale + scale / 2)

    if path_cells:
        pts = [px(*grid.c2w(r, c)) for r, c in path_cells]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 230, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 180, 0), -1)
    gx, gy = px(*goal_xy); cv2.drawMarker(img, (gx, gy), (0, 180, 255), cv2.MARKER_STAR, 18, 2)
    rx, ry = px(*robot_xy)
    cv2.circle(img, (rx, ry), 8, (255, 200, 0), -1)
    hx, hy = px(robot_xy[0] + 0.3 * np.cos(robot_yaw), robot_xy[1] + 0.3 * np.sin(robot_yaw))
    cv2.line(img, (rx, ry), (hx, hy), (255, 200, 0), 2)
    banner = "ACCEPT plan?  [a]=accept  [r]=reject" if accepted is None else ("ACCEPTED" if accepted else "REJECTED")
    col = (0, 230, 230) if accepted is None else ((0, 230, 0) if accepted else (0, 0, 230))
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (25, 25, 25), -1)
    cv2.putText(img, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    return img


ROBOT_RADIUS = 0.35   # G1-D footprint half-width for inflation (m)


def plan(grid, robot_xy, goal_xy):
    inflated = grid.inflate(ROBOT_RADIUS)
    s = grid.w2c(*robot_xy); g = grid.w2c(*goal_xy)
    raw = astar(inflated, s, g)
    return simplify(inflated, raw) if raw else None


def plan_and_confirm(grid, robot_xy, robot_yaw, goal_xy, window="nav plan"):
    """Show the digital-twin plan and block until the user accepts (a) or rejects (r/ESC). Returns (path, ok)."""
    path = plan(grid, robot_xy, goal_xy)
    img = render_topdown(grid, robot_xy, robot_yaw, goal_xy, path)
    if path is None:
        cv2.putText(img, "NO PATH", (8, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow(window, img)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("a") and path is not None:
            cv2.imshow(window, render_topdown(grid, robot_xy, robot_yaw, goal_xy, path, accepted=True)); cv2.waitKey(400)
            return path, True
        if k in (ord("r"), 27):
            return path, False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="/tmp/plan.png")
    a = ap.parse_args()
    if a.demo:
        g = OccupancyGrid(size_m=8.0, res=0.05)
        g.add_box(2.0, 0.9, 0.8, 0.8)      # pick table  (robot stands in front, y<0.5)
        g.add_box(-2.0, 0.9, 0.8, 0.8)     # place table
        g.add_box(0.0, 0.0, 0.6, 1.2)      # obstacle between them
        robot = (2.0, -0.2); goal = (-2.0, -0.2)   # in front of each table, clear of inflation
        path = plan(g, robot, goal)
        print("path cells:", None if path is None else len(path))
        img = render_topdown(g, robot, np.pi, goal, path)
        cv2.imwrite(a.out, img)
        print("wrote", a.out)
