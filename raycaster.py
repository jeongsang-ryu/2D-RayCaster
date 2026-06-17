#!/usr/bin/env python3
"""
RaycastEngine — one 2D LiDAR raycaster for BOTH the simulator and the particle filter.

Backends
  'glt' | 'pcddt' | 'cddt' | 'rm'  : range_libc C++ (fast, needs the .so built)
  'lut'                            : precomputed numpy table (NO range_libc at
                                     query time → portable to NUC / Mac mini;
                                     load a .npz and query with numpy indexing)

Consumers
  Simulator      : scan(pose_xyt, num_beams, fov)          -> ranges[num_beams]  (meters)
  Particle filter: calc_range_repeat_angles(particles, angles) -> ranges[M*K]    (meters)

Coordinate convention (ROS): occupancy[row, col] with row 0 at world y = origin_y
(bottom-origin). world -> pixel: col = (x-ox)/res, row = (y-oy)/res. Use
RaycastEngine.load_map_yaml() to load an F1TENTH map image (it flips top-bottom).
"""
import os
import numpy as np

_RL_BACKENDS = ("glt", "pcddt", "cddt", "rm", "bl")


class RaycastEngine:
    def __init__(self, backend="pcddt", max_range_m=10.0, theta_disc=112):
        self.backend = backend
        self.max_range_m = float(max_range_m)
        self.theta_disc = int(theta_disc)
        self.res = self.ox = self.oy = None
        self.W = self.H = None
        self._rl = None          # range_libc method instance
        self._lut = None         # numpy LUT [W, H, theta_disc] (uint16, pixel ranges)
        self._dtheta = 2 * np.pi / self.theta_disc

    # ---------- map ----------
    def set_map(self, occupancy, resolution, origin_xy):
        """occupancy: bool/uint8 [H,W] True=occupied, row0 at world y=origin_y.
        Builds the chosen backend. For 'lut', materializes the table (see save_lut)."""
        occ = np.ascontiguousarray(occupancy).astype(bool)
        self.H, self.W = occ.shape
        self.res = float(resolution); self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        max_range_px = self.max_range_m / self.res
        if self.backend in _RL_BACKENDS:
            import range_libc
            omap = range_libc.PyOMap(occ)
            if self.backend == "glt":
                self._rl = range_libc.PyGiantLUTCast(omap, max_range_px, self.theta_disc)
            elif self.backend in ("cddt", "pcddt"):
                self._rl = range_libc.PyCDDTCast(omap, max_range_px, self.theta_disc)
                if self.backend == "pcddt":
                    self._rl.prune()
            elif self.backend == "rm":
                self._rl = range_libc.PyRayMarching(omap, max_range_px)
            elif self.backend == "bl":
                self._rl = range_libc.PyBresenhamsLine(omap, max_range_px)
        elif self.backend == "lut":
            self._materialize_lut(occ)
        else:
            raise ValueError(f"unknown backend {self.backend}")
        return self

    # ---------- core query (pixel space) ----------
    def _query_px(self, qxyt):
        """qxyt float32 [N,3] in pixels (col,row,theta) -> ranges_px [N]."""
        out = np.zeros(qxyt.shape[0], dtype=np.float32)
        if self._rl is not None:
            self._rl.calc_range_many(np.ascontiguousarray(qxyt, np.float32), out)
            return out
        # LUT path
        xi = np.clip(qxyt[:, 0].astype(np.int64), 0, self.W - 1)
        yi = np.clip(qxyt[:, 1].astype(np.int64), 0, self.H - 1)
        ki = np.mod(np.rint(qxyt[:, 2] / self._dtheta).astype(np.int64), self.theta_disc)
        return self._lut[xi, yi, ki].astype(np.float32)

    def _w2p(self, x, y):
        return (x - self.ox) / self.res, (y - self.oy) / self.res

    # ---------- SIMULATOR API ----------
    def scan(self, pose_xyt, num_beams, fov, max_range=None):
        """pose [x,y,theta] world meters -> ranges [num_beams] meters."""
        px, py = self._w2p(pose_xyt[0], pose_xyt[1])
        off = np.linspace(-fov / 2, fov / 2, num_beams)
        q = np.empty((num_beams, 3), np.float32)
        q[:, 0] = px; q[:, 1] = py; q[:, 2] = pose_xyt[2] + off
        r = self._query_px(q) * self.res
        return np.minimum(r, max_range or self.max_range_m)

    # ---------- PARTICLE FILTER API ----------
    def calc_range_repeat_angles(self, particles_xyt, angles):
        """particles [M,3] world, angles [K] -> ranges [M*K] meters (row-major per particle)."""
        M, K = particles_xyt.shape[0], angles.shape[0]
        px, py = self._w2p(particles_xyt[:, 0], particles_xyt[:, 1])
        if self._rl is not None and self.backend in ("glt", "pcddt", "cddt", "rm", "bl"):
            parts = np.empty((M, 3), np.float32)
            parts[:, 0] = px; parts[:, 1] = py; parts[:, 2] = particles_xyt[:, 2]
            out = np.zeros(M * K, np.float32)
            self._rl.calc_range_repeat_angles(parts, np.ascontiguousarray(angles, np.float32), out)
            return out * self.res
        # LUT / generic: vectorized gather
        ang = particles_xyt[:, 2][:, None] + angles[None, :]          # [M,K]
        xi = np.clip(px.astype(np.int64), 0, self.W - 1)[:, None]
        yi = np.clip(py.astype(np.int64), 0, self.H - 1)[:, None]
        ki = np.mod(np.rint(ang / self._dtheta).astype(np.int64), self.theta_disc)
        return (self._lut[xi, yi, ki] * self.res).reshape(M * K)

    # ---------- precomputed LUT: build / save / load ----------
    def _materialize_lut(self, occ):
        """Build the [W,H,theta_disc] pixel-range LUT once. Oracle = pruned CDDT
        (GLT/RM return 0 on some high-occupancy maps; CDDT is reliable)."""
        import range_libc
        omap = range_libc.PyOMap(occ)
        oracle = range_libc.PyCDDTCast(omap, self.max_range_m / self.res, self.theta_disc)
        oracle.prune()
        lut = np.zeros((self.W, self.H, self.theta_disc), np.uint16)
        xs = np.repeat(np.arange(self.W, dtype=np.float32), self.H)
        ys = np.tile(np.arange(self.H, dtype=np.float32), self.W)
        q = np.zeros((self.W * self.H, 3), np.float32); q[:, 0] = xs; q[:, 1] = ys
        out = np.zeros(self.W * self.H, np.float32)
        cap = self.max_range_m / self.res
        for k in range(self.theta_disc):
            q[:, 2] = k * self._dtheta
            oracle.calc_range_many(q, out)
            lut[:, :, k] = np.minimum(out, cap).reshape(self.W, self.H).astype(np.uint16)
        self._lut = lut

    def save_lut(self, path):
        if self._lut is None:
            raise RuntimeError("no LUT to save (use backend='lut' and set_map first)")
        np.savez_compressed(path, lut=self._lut, resolution=self.res,
                            origin=np.array([self.ox, self.oy], np.float64),
                            max_range_m=self.max_range_m, theta_disc=self.theta_disc)

    @classmethod
    def load_lut(cls, path):
        """Fast path: load a precomputed LUT — NO range_libc needed."""
        z = np.load(path)
        e = cls(backend="lut", max_range_m=float(z["max_range_m"]), theta_disc=int(z["theta_disc"]))
        e._lut = z["lut"]; e.W, e.H = e._lut.shape[0], e._lut.shape[1]
        e.res = float(z["resolution"]); e.ox, e.oy = [float(v) for v in z["origin"]]
        return e

    # ---------- helper: load an F1TENTH map yaml ----------
    @staticmethod
    def load_map_yaml(yaml_path):
        """-> (occupancy[H,W] bool row0=bottom, resolution, (origin_x, origin_y))."""
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_path))
        img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta["image"])))
        if img.ndim == 3:
            img = img[..., 0]
        occ = np.flipud(img) <= 128          # flip top-bottom -> row0 = world bottom
        return occ, float(meta["resolution"]), (meta["origin"][0], meta["origin"][1])


if __name__ == "__main__":
    import sys, time
    CAC = os.environ.get("CAC_DIR", "/home/js/unicorn_racing_stack/src/creating_autonomous_car")
    sys.path.insert(0, CAC + "/slam/range_libc/pywrapper")
    occ, res, origin = RaycastEngine.load_map_yaml(CAC + "/stack_master/maps/f/f.yaml")
    # a free world pose
    ys, xs = np.where(~occ); i = len(xs) // 2
    pose = np.array([origin[0] + xs[i] * res, origin[1] + ys[i] * res, 0.3])
    print(f"map {occ.shape}, pose {pose.round(2)}")
    ref = None
    for be in ("pcddt", "cddt", "lut"):
        e = RaycastEngine(backend=be, max_range_m=10.0).set_map(occ, res, origin)
        s = e.scan(pose, 1080, 4.7)
        if ref is None: ref = s
        parts = np.tile(pose, (4000, 1)).astype(np.float32)
        ang = np.linspace(-4.7/2, 4.7/2, 100).astype(np.float32)
        t = time.perf_counter()
        for _ in range(10): e.calc_range_repeat_angles(parts, ang)
        pf = (time.perf_counter() - t) / 10 * 1e3
        err = np.abs(s - ref).mean() * 100
        print(f"  {be:6s}: scan mean {s.mean():.2f} m | PF 4000x100 {pf:.2f} ms ({25/pf:.1f}x rt) | vs pcddt {err:.1f} cm")
    # save/load roundtrip
    e = RaycastEngine(backend="lut", max_range_m=10.0).set_map(occ, res, origin)
    e.save_lut("/tmp/f_lut.npz")
    e2 = RaycastEngine.load_lut("/tmp/f_lut.npz")
    print(f"  load_lut roundtrip: scan mean {e2.scan(pose,1080,4.7).mean():.2f} m  (no range_libc needed)")
