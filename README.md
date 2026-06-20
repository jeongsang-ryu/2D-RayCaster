# 2D-RayCaster

A fast, **standalone 2D LiDAR ray-caster** for occupancy grids — usable with or
without ROS. The core is the header-only C++ `range_libc` (Bresenham / ray-march
/ CDDT / GiantLUT, optional CUDA) wrapped to Python with **pybind11** (no Cython,
no numpy at build), plus a unified [`RaycastEngine`](raycaster.py) that the
F1TENTH simulator and particle filter share.

```python
import numpy as np, range_libc
occ = np.load("map.npy")                       # bool [H, W], True = occupied
omap = range_libc.PyOMap(occ, resolution=0.05, origin_x=0.0, origin_y=0.0)
rm   = range_libc.PyRayMarching(omap, 10.0/0.05)
out  = np.zeros(1080, np.float32)
rm.calc_range_many(queries_xyT, out)           # queries [N,3] in world meters
```

No ROS, numba, or Cython is required to build or run the CPU library.

---

## Install

### 1. Basic — CPU (any OS / arch)

Needs only a C++17 compiler and Python ≥ 3.8. `pybind11` is the single build
dependency (header-only; it carries the numpy C-API itself, so numpy is **not**
needed at build time).

```bash
git clone --recursive https://github.com/jeongsang-ryu/2D-RayCaster.git
cd 2D-RayCaster/range_libc/pywrapper
pip install .                       # or: pip install -e .   (editable)
```

That gives the CPU backends: `bl`, `rm`, `cddt`, `pcddt`, `GLUT`. This path
builds identically on **Linux (x86_64 / aarch64), macOS (Intel / Apple Silicon),
and Windows** — it is the recommended default and what the benchmarks below use.

> Conda/RoboStack tip: install with `pip install --no-build-isolation .` after
> `conda install -c conda-forge pybind11` to build entirely from conda with zero
> PyPI fetch.

### 2. NVIDIA — CUDA (`rmgpu`)

`range_libc` ships CUDA kernels ([`includes/kernels.cu`](range_libc/includes/kernels.cu),
`CudaRangeLib.h`) for a GPU ray-marcher `PyRayMarchingGPU` (`rmgpu`) — the
throughput winner for large batches (particle filter, RL rollouts).

**Status:** the pybind11 module currently builds **CPU-only**; `PyRayMarchingGPU`
imports but is a stub that warns (so CPU code always works). Enabling the real
CUDA backend is a one-flag build that compiles `kernels.cu` with `nvcc` and
defines `USE_CUDA` — wiring it into this pybind `setup.py` is the next step:

```bash
# intended interface (needs the CUDA toolkit matching your driver, nvidia-smi):
cd range_libc/pywrapper
WITH_CUDA=1 pip install .
python -c "import range_libc; print('CUDA:', range_libc.SHOULD_USE_CUDA)"   # -> True
```

```python
gpu = range_libc.PyRayMarchingGPU(omap, 10.0/0.05)
gpu.calc_range_repeat_angles(particles, angles, out)     # 4000×100 batch on the GPU
```

> Until then, GPU throughput on NVIDIA is available via the **jax** backend
> (`jax[cuda12]`, section below) — same DT algorithm — and the CPU `GLUT` backend
> already gives 13× real-time headroom on the particle-filter batch (see benchmark).

### 3. Intel GPU

`range_libc`'s GPU kernels are **CUDA-only**, so there is no native Intel-GPU
`rmgpu`. Two options:

- **CPU build** (section 1) runs natively on Intel x86 and is already fast enough
  for the simulator and (with `GLUT`/`pcddt`) the particle filter — see benchmarks.
- **Intel-GPU acceleration** comes through the **jax** ray-caster (same DT
  sphere-tracing algorithm), via Intel's oneAPI/XPU plugin:
  ```bash
  pip install jax "intel-extension-for-openxla"   # experimental, Arc-focused
  JAX_PLATFORMS=xpu python benchmark/bench_jax.py
  ```
  (As of writing this is experimental; the CPU `range_libc` build is the reliable
  choice on Intel NUCs.)

### 4. Apple Silicon

- **CPU build** (section 1) compiles natively on arm64 macOS via clang/`libc++` —
  this is the recommended path on a Mac mini / MacBook.
- **Apple-GPU acceleration** is **not** in `range_libc` (CUDA-only kernels); use
  the **jax** ray-caster with `jax-metal`:
  ```bash
  pip install jax-metal
  python benchmark/bench_jax.py        # runs on the Metal GPU
  ```
  `jax-metal` is experimental; for reliable results on macOS prefer the CPU
  `range_libc` build (`GLUT` for the heavy particle-filter workload).

### Backend × platform summary

| backend | Linux/Win x86 | Linux aarch64 (Orin) | macOS (Apple) | how |
|---|:--:|:--:|:--:|---|
| `bl/rm/cddt/pcddt/GLUT` (CPU) | ✅ | ✅ | ✅ | section 1 |
| `rmgpu` (CUDA) | ✅ (NVIDIA) | ✅ (Orin) | ❌ | section 2 |
| `jax-rm` (GPU) | ✅ cuda | ✅ cuda | ✅ metal | sections 3–4 |

---

## Benchmark

`benchmark/run_bench.py` measures the two real workloads on whatever backends the
host supports and **tags each result with the machine it ran on**, so numbers from
the 4080, an Orin, a laptop, etc. stay comparable. See
[`benchmark/README.md`](benchmark/README.md) for methodology and the full matrix.

```bash
cd benchmark && python run_bench.py        # writes results/<device>.md
```

- **SIM** = time for **one** LiDAR scan (1080 beams) — latency.
- **PF** = time for the **4000-particle × 100-beam** batch — throughput.
- `headroom = 25 ms budget / time` ⇒ survives ~N× slower hardware.

### Results — RTX 4080 SUPER (Ryzen 9 7950X, `test` map, pybind11/cpu)

| backend | SIM ms | SIM headroom | PF ms | PF headroom | note |
|---|--:|--:|--:|--:|---|
| `GLUT (cpu)`  | 0.008 | 3247× | **1.88** | **13×** | fastest; big init + LUT memory |
| `pcddt (cpu)` | 0.015 | 1641× | 8.03 | 3.1× | **balanced — PF default** |
| `cddt (cpu)`  | 0.015 | 1682× | 8.47 | 3.0× | fast init |
| `rm (cpu)`    | 0.021 | 1214× | 21.2 | 1.2× | simple/robust; sim default |
| `bl (cpu)`    | 0.338 |   74× | 125  | 0.2× | slow, avoid |

`rmgpu (cuda)`, `cpu-omp`, and the `jax-rm` rows are filled in by re-running on a
CUDA / OpenMP / jax host — see [`benchmark/results/`](benchmark/results/).

**Takeaways**
- **SIM** is latency-bound; every backend has 100×+ headroom → pick the engine for
  *code unity* (one raycaster for sim + PF), not speed.
- **PF** is the real constraint. `GLUT` (13×) is safe even on weak hardware (cost:
  ~0.7 s init + LUT memory); `pcddt`/`cddt` (~3×) are fine on a desktop; `rm` is
  desktop-only for the full batch; `bl` fails.

---

## Algorithms

| backend | principle |
|---|---|
| `bl` Bresenham / DDA | integer cell-stepping along the grid |
| `rm` ray-marching (= sphere tracing) | jump by the precomputed distance-transform each step (adaptive) |
| `cddt` / `pcddt` | discretize directions, compressed LUT of obstacle distances; `p` = pruned |
| `GLUT` Giant LUT | full (x, y, θ) → range table, O(1) lookup; biggest memory/init |
| `rmgpu` | `rm` on CUDA — wins on large batches |
| `jax-rm` | the same DT sphere-tracing on XLA (CUDA / Metal / XPU / CPU) |

`rm`, `rmgpu`, `jax-rm`, and numba are the **same** DT sphere-tracing algorithm;
`cddt` / `GLUT` precompute past it. All CPU backends agree on measured ranges
(`rm ≈ cddt ≈ GLUT`, validated in the pywrapper cross-checks).

## Library API ([`raycaster.py`](raycaster.py))

`RaycastEngine` is a single raycaster for both consumers, with a pure-numpy `lut`
backend that needs **no** native build at query time (precompute once → save
`.npz` → load anywhere):

```python
from raycaster import RaycastEngine
e = RaycastEngine(backend="rm", max_range_m=10.0, theta_disc=720).set_map(occ, res, origin)
ranges = e.scan([x, y, theta], num_beams=1080, fov=4.7)        # simulator
ranges = e.calc_range_repeat_angles(particles_xyT, angles)      # particle filter
```

## References
- range_libc / CDDT: Walsh & Karaman, *"CDDT: Fast Approximate 2D Ray Casting…"*, ICRA 2018, [arXiv:1705.01167](https://arxiv.org/abs/1705.01167) · [kctess5/range_libc](https://github.com/kctess5/range_libc)
- Sphere tracing: Hart, *The Visual Computer*, 1996 · Bresenham: IBM Sys. J., 1965
- JAX: [google/jax](https://github.com/google/jax) · F1TENTH gym: O'Kelly et al., NeurIPS 2019
