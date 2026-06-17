# raycaster — 2D LiDAR ray-casting benchmark & comparison

UNICORN racing-stack의 2D 라이다 **raycaster(광선 투사)** 구현들을 정리·비교·검증하는 도구.
시뮬레이터와 파티클 필터의 성능은 사실상 raycaster 효율이 좌우하므로, 어떤 방법이 어떤
워크로드(특히 저사양 하드웨어)에 적합한지 정량적으로 측정한다.

> ROS 패키지가 아니라 **연구/벤치마크 도구**다 (`COLCON_IGNORE` 포함, colcon 빌드 제외).
> 향후 NUC·Mac mini 등에서 재측정해 `results/`에 날짜별로 누적한다.

## 구조
```
tools/raycaster/
├── README.md                 # 이 문서 (분석·결론·알고리즘·참고)
├── SETUP.md                  # 재현 환경 구축 (venv, range_libc 빌드, jax)
├── benchmarks/
│   ├── bench_raycast.py       # numba + range_libc(bl/rm/cddt/pcddt/glt) + segment
│   ├── bench_jax.py           # jax DT sphere-tracing (single/batch, GPU/CPU)
│   └── bench_workload.py      # 실시간 타당성: SIM vs PF 워크로드 검증
└── results/
    ├── 2026-06-17_ryzen7950x_rtx4080super.md   # 머신별 원시 결과(날짜+환경)
    └── ray_casting_methods_comparison.html      # 인터랙티브 시각화
```

## 핵심 결과 (Ryzen 7950X + RTX 4080S, 2026-06-17 — 전체는 `results/`)

`1 scan = 1080-beam 프레임 1개`. 단일 스캔 처리량:

| Raycaster | Backend | scans/s | vs numba | 한 줄 평 |
|---|---|--:|--:|---|
| jax DT-march, batched | GPU | 155,300 | 9.7× | 대량 병렬 최강(NVIDIA 필요) |
| range_libc **GLT** | CPU | 147,700 | 9.2× | 단일 최고속, init·메모리 큼 |
| range_libc **PCDDT/CDDT** | CPU | ~52,000 | ~3× | **균형 — 추천** |
| range_libc RM | CPU | 23,500 | 1.5× | 단순·견고 |
| **numba f1tenth_gym** *(현재 sim)* | CPU | 16,100 | 1.0× | 기준선 |
| range_libc BL (Bresenham) | CPU | 3,600 | 0.2× | 느림 |

### 실시간 타당성 — Simulator vs Particle Filter (저사양 핵심)

| 워크로드 | 패턴 | 요구량 |
|---|---|--:|
| Simulator | 2 lidar × 40 Hz × 2200 beams | 176 k rays/s (latency) |
| Particle Filter | 4000 particles × 100 beams × 40 Hz | 16 M rays/s (throughput) |

25 ms/cycle 예산, headroom = 25 ms / 측정시간 (≈ N배 느린 HW까지 실시간 가능):

| 방법 | SIM ↑ | PF ↑ | 판정 |
|---|--:|--:|---|
| numba / jax-CPU | 117× / 40× | **0.7× / 0.4×** | **PF 실패(데스크탑서도)** |
| range_libc cddt/pcddt | ~300× | ~3× | PF OK(약한 랩탑선 위험) |
| **range_libc glt** | 1048× | **16×** | **PF 안전(저사양도)** |
| jax-GPU | 25× | 9× | PF OK(NVIDIA GPU 한정) |

**결론**
- **Simulator**: latency-bound, 전부 100×+ 여유 → 저사양 문제없음. *코드 통일* 기준으로 선택.
- **Particle Filter**: throughput-bound, 진짜 제약. numba·jax-CPU는 실시간 실패.
  저사양/휴대형까지 보장하려면 **PF = range_libc GLT(CPU)**. 부담 시 PCDDT + 파티클/beam 축소.

### jax-GPU 플랫폼 가용성

| 플랫폼 | jax GPU | PF 가능? |
|---|---|---|
| NVIDIA dGPU | ✅ `jax[cuda]` | ◎ |
| Intel iGPU (NUC) | ❌ CPU 폴백 (`intel-extension-for-openxla`는 실험적/Arc 위주) | ❌ |
| Apple Silicon (Mac mini) | △ `jax-metal` 실험적·미완성 | ✗ 신뢰불가 |
| GPU 없음 | ❌ | ❌ |

→ NUC·Mac mini 등에선 jax-GPU에 의존 불가. **range_libc(CPU)** 가 이식성·성능 모두 정답
(C++라 x86/ARM 어디서나 컴파일). range_libc `rmgpu`도 CUDA 전용이라 같은 한계.

## 알고리즘 한눈에 ("0.01씩 전진"은 naive 버전)

| 방법 | 원리 |
|---|---|
| **Bresenham/DDA** (`bl`) | 격자를 셀 단위 정수 증분으로 이동하며 점유 검사 (고정 스텝의 정수판) |
| **Ray marching = sphere tracing** (`rm`, numba, jax) | **거리 변환(DT)** 으로 "가장 가까운 장애물 거리"만큼 한 번에 점프 → 적응형. ("0.01 step"의 똑똑한 버전) |
| **CDDT / PCDDT** (`cddt`) | 방향을 이산화해 거리 LUT를 압축 저장, 쿼리는 lookup+보간. PCDDT는 가지치기 |
| **Giant LUT** (`glt`) | (x,y,θ)→거리 전체 precompute, O(1) 조회. 최고속·고메모리 |
| **Segment analytic** | 광선-선분 교차 해석 계산. 격자 무관·정확 (2D z-buffer로 가속 가능) |

> numba·range_libc `rm`·jax는 **모두 같은 DT sphere-tracing 알고리즘**, 구현(numba/C++/XLA)만 다름.

## range_libc 전체 스택 통일 전략

particle_filter는 이미 range_libc(`pcddt`)를 쓴다. 시뮬레이터(`f1tenth_gym_ros`)의 numba도
range_libc 백엔드로 교체하면 **sim·localization·(옵션)planner가 단일 raycast 엔진**을 공유 →
맵 로딩·좌표 규약·거리 정의 일원화, 유지보수 단순. 한 번 빌드한 `range_libc.so` 공유.
권장: sim 백엔드를 PCDDT(또는 저사양 대비 GLT)로 교체 → numba와 출력 일치 검증 → 통일.

## 재현
[`SETUP.md`](SETUP.md) 참고. 요약:
```bash
# 의존: creating_autonomous_car 의 f110_gym / slam/range_libc / stack_master/maps
export CAC_DIR=/path/to/creating_autonomous_car      # 기본값은 HMCL 데스크탑 경로
$VENV/python benchmarks/bench_raycast.py             # MAP=test|f
$VENV/python benchmarks/bench_jax.py                 # JAX_PLATFORMS=cpu 로 CPU
$VENV/python benchmarks/bench_workload.py            # SIM vs PF 실시간 검증
```

## References
- range_libc / CDDT: C. Walsh, S. Karaman, *"CDDT: Fast Approximate 2D Ray Casting for Accelerated Localization,"* ICRA 2018. [arXiv:1705.01167](https://arxiv.org/abs/1705.01167) · [kctess5/range_libc](https://github.com/kctess5/range_libc) · [f1tenth/range_libc](https://github.com/f1tenth/range_libc)
- F1TENTH gym: O'Kelly et al., NeurIPS 2019. [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) · [f1tenth/f1tenth_gym_jax](https://github.com/f1tenth/f1tenth_gym_jax)
- JAX: Bradbury et al., 2018. [google/jax](https://github.com/google/jax)
- Sphere tracing: J. C. Hart, *The Visual Computer*, 1996. · Bresenham: IBM Sys. J., 1965. · Z-buffer: Catmull, 1974.
