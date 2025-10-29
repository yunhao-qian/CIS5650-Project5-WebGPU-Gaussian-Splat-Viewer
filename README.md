# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Yunhao Qian
* Tested on: (TODO) Google Chrome 141.0 on
  * Windows 11, 24H2
  * 13th Gen Intel(R) Core(TM) i7-13700 (2.10 GHz)
  * NVIDIA GeForce RTX 4090

## Live Demo

[![Live demo](img/thumb.png)](https://yunhao-qian.github.io/showcase/gaussian-splatting/)

## Demo Video

https://github.com/user-attachments/assets/5a173f00-dfd2-4566-b4ca-4921d0ac81bb

# Implementation & Performance Report

## Implementation

### Point Cloud Renderer

- Renders all points as unit yellow dots (fixed size), regardless of distance to camera.
- Color is a uniform solid yellow.

### Gaussian Renderer

- Renders each point as a splat (disk/billboard) whose:
  - Color comes from the source PLY model (viewed from different angles you’ll perceive different color mixes due to overlapping/transparency).
  - Size and opacity falloff are driven by the point’s covariance: high opacity at the center; opacity decreases smoothly toward the edges.

## Performance Analysis

Test assets: `bonsai.ply` and `bicycle.ply`.

### Point Cloud vs. Gaussian Renderers

Visuals:

- Point cloud: fixed-size pixels/points, always solid yellow, no depth-related size attenuation.
- Gaussian: true surface colors from PLY, splat size respects scale, opacity smoothly decays from each Gaussian’s center.

Throughput:

- On my setup, both renderers are similar in frame rate in uncongested views, generally ~56-60 FPS.

### Workgroup Sizes (Preprocess Step)

Measured total preprocess time while sweeping a single workgroup-size parameter:

| Workgroup Size | Time (ms) | Relative Speedup vs. 32× |
|---:|---:|---:|
| 32  | 4,627 | 1.00× |
| 64  | 4,734 | 0.98× |
| 128 | 5,651 | 0.82× |
| 256 | 4,365 | 1.06× |
| 512 | 3,305 | 1.40× |
| 1,024 | 4,846 | 0.95× |
| 2,048 | 5,139 | 0.90× |

Analysis:

* 512 is the clear sweet spot here (≈1.40× faster than 32; ~30% faster than 256).
* Very small groups (32) underutilize the GPU; very large ones (≥1,024) likely reduce scheduling flexibility.

> Sorting Kernel (for Gaussian Splatting): Attempted to change sorting workgroup size; program misbehaved.

### View-Frustum Culling

* Outside scene (both `bonsai.ply` and `bicycle.ply`): ~56 FPS with or without culling — the scenes are small vs. GPU capability, so savings are masked by fixed overhead.
* Heavy view on `bicycle.ply` (set Gaussian multiplier to max (1.5), zoomed in):
  * ~10 FPS without culling -> ~40 FPS with culling (~4× speedup).
* Conclusion: Culling removes gaussians from sorting and from the graphics pipeline, so it can reduce time roughly proportionally to the number of discarded splats. Gains become obvious in dense, overdraw-heavy views.

### Number of Gaussians (Scaling)

Preprocess timings

* `bicycle.ply`: 1,063,091 pts in 24,684 ms -> ~43.07 pts/ms (≈ 43.1 k pts/s).
* `bonsai.ply`: 272,956 pts in 4,365 ms -> ~62.53 pts/ms (≈ 62.5 k pts/s).

Average throughput: ~52.8 pts/ms (≈ 52.8 k pts/s).

Analysis:

* Preprocess time scales roughly linearly with the number of points, indicating the GPU is well saturated.
* The per-scene variation (43k-62k pts/s) likely comes from:
  * Different covariance distributions (branching, math intensity).
  * Cache locality and memory access patterns (attribute stride/packing).
  * Different proportions of filtered/culled points during preprocess.

Display stage:

* In typical views for both assets, display holds near ~56-60 FPS; large improvements appear only when enabling culling in dense views.

## Credits

* [Vite](https://vitejs.dev/)
* [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
* [stats.js](https://github.com/mrdoob/stats.js)
* [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
* Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
